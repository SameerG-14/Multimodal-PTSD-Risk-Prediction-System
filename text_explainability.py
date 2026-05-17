"""
Fine-tuned Mental-RoBERTa text explainer for PTSD vs NO PTSD attribution.
Loaded at inference for token-level gradient saliency (separate from fusion prediction).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("ptsd.text_explain")

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    _STOPWORDS: FrozenSet[str] = frozenset(w.lower() for w in ENGLISH_STOP_WORDS)
except ImportError:  # pragma: no cover
    _STOPWORDS = frozenset(
        "a an the and or but if in on at to for of as by with from up down out "
        "so such no nor not only own same than too very just also about into "
        "is are was were been being be have has had having do does did doing "
        "it its this that these those i you he she they we what which who whom "
        "when where why how all both each few more most other some such".split()
    )


def _clean_transcript_for_explainer(text: str) -> str:
    """Strip accidental markup (e.g. literal <s>...</s>) so RoBERTa sees normal words."""
    t = re.sub(r"</?s\s*>", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


def _normalize_wordpiece(tok: str) -> str:
    return tok.replace("Ġ", "").replace("##", "").strip().lower()


def _is_stoplike(tok: str) -> bool:
    """Gradient salience often spikes on function words; hide strong tint for display."""
    w = _normalize_wordpiece(tok)
    if len(w) <= 2:
        return True
    return w in _STOPWORDS


class TextExplainerModel:
    def __init__(self, model_path: Optional[str], device: torch.device):
        self.model_path = Path(model_path).resolve() if model_path else None
        self.device = device
        self._tokenizer = None
        self._model: Optional[nn.Module] = None
        self._loaded = False

    def is_available(self) -> bool:
        if not self.model_path:
            return False
        cfg = self.model_path / "config.json"
        return cfg.is_file()

    def load(self) -> bool:
        if self._loaded:
            return self._model is not None
        self._loaded = True
        if not self.is_available():
            log.info("Text explainer checkpoint not found — skip linguistic attribution.")
            return False
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path)
            ).to(self.device)
            self._model.eval()
            log.info(f"Text explainer loaded from {self.model_path}")
            return True
        except Exception as e:
            log.warning(f"Text explainer load failed: {e}")
            self._model = None
            self._tokenizer = None
            return False

    def explain(self, transcript: str, max_length: int = 512) -> Optional[Dict[str, Any]]:
        if not transcript or not transcript.strip():
            return None
        if not self.load() or self._model is None or self._tokenizer is None:
            return None

        transcript = _clean_transcript_for_explainer(transcript)
        if len(transcript) < 8:
            return None

        enc = self._tokenizer(
            transcript,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        with torch.no_grad():
            base_out = self._model(**enc)
            probs = torch.softmax(base_out.logits, dim=-1)[0].cpu().numpy()
        # Expect id2label 0 = NO PTSD, 1 = PTSD
        pt_idx, no_idx = 1, 0
        if probs.shape[0] < 2:
            return None

        pt_prob = float(probs[pt_idx])
        label = "PTSD" if pt_prob >= 0.5 else "NO PTSD"

        backbone = getattr(self._model, "roberta", None) or getattr(
            self._model, "bert", None
        )
        if backbone is None:
            return {
                "available": True,
                "label": label,
                "ptsd_probability": pt_prob,
                "no_ptsd_probability": float(probs[no_idx]),
                "token_attributions": [],
                "disclaimer": "Unsupported backbone for gradient attribution.",
            }

        emb_layer = backbone.embeddings

        def grad_wrt_class(cls_idx: int) -> Optional[torch.Tensor]:
            embedded = emb_layer(input_ids).detach().clone().requires_grad_(True)
            self._model.zero_grad(set_to_none=True)
            out = self._model(inputs_embeds=embedded, attention_mask=attention_mask)
            out.logits[0, cls_idx].backward(retain_graph=False)
            return embedded.grad.detach().clone() if embedded.grad is not None else None

        g_pt = grad_wrt_class(pt_idx)
        g_no = grad_wrt_class(no_idx)

        if g_pt is None or g_no is None:
            return {
                "available": True,
                "label": label,
                "ptsd_probability": pt_prob,
                "no_ptsd_probability": float(probs[no_idx]),
                "token_attributions": [],
                "disclaimer": "Token gradients unavailable for this input.",
            }

        mag_pt = g_pt.abs().sum(dim=-1)[0].float().cpu().numpy()
        mag_no = g_no.abs().sum(dim=-1)[0].float().cpu().numpy()
        mag_pt = mag_pt / (mag_pt.max() + 1e-9)
        mag_no = mag_no / (mag_no.max() + 1e-9)

        ids = input_ids[0].tolist()
        tokens = self._tokenizer.convert_ids_to_tokens(ids)
        special_ids: Set[int] = set(getattr(self._tokenizer, "all_special_ids", []) or [])

        rows: List[Dict[str, Any]] = []
        for i, tok in enumerate(tokens):
            if attention_mask[0, i].item() == 0:
                continue
            if ids[i] in special_ids:
                continue
            mp, mn = float(mag_pt[i]), float(mag_no[i])
            diff = mp - mn
            rows.append({"token": tok, "score_ptsd": mp, "score_no": mn, "delta": diff})

        if not rows:
            return {
                "available": True,
                "label": label,
                "ptsd_probability": round(pt_prob, 6),
                "no_ptsd_probability": round(float(probs[no_idx]), 6),
                "token_attributions": [],
                "disclaimer": "No word pieces to attribute after skipping special tokens.",
            }

        mps = np.array([float(r["score_ptsd"]) for r in rows], dtype=np.float64)
        mns = np.array([float(r["score_no"]) for r in rows], dtype=np.float64)
        peak_joint = float(max(mps.max(), mns.max())) + 1e-9
        # Dual channels: compare salience toward PTSD logit vs NO PTSD logit separately.
        # Using only (mp−mn) biases color toward the predicted class (often all green).
        thr_mp = float(np.percentile(mps, 72.0))
        thr_mn = float(np.percentile(mns, 72.0))

        attributions: List[Dict[str, Any]] = []
        for r in rows:
            mp = float(r["score_ptsd"])
            mn = float(r["score_no"])
            stop = _is_stoplike(r["token"])

            cand_pt = mp >= thr_mp and mp >= mn * 1.08
            cand_mn = mn >= thr_mn and mn >= mp * 1.08

            if stop:
                lean = "neutral"
                intensity = min(0.14, max(mp, mn) / peak_joint * 0.25)
            elif cand_pt and cand_mn:
                lean = "PTSD" if mp >= mn else "NO PTSD"
                intensity = max(mp, mn) / peak_joint
            elif cand_pt:
                lean = "PTSD"
                intensity = mp / peak_joint
            elif cand_mn:
                lean = "NO PTSD"
                intensity = mn / peak_joint
            else:
                lean = "neutral"
                intensity = (max(mp, mn) / peak_joint) * 0.32

            attributions.append(
                {
                    "token": r["token"],
                    "score_ptsd": round(mp, 4),
                    "score_no_ptsd": round(mn, 4),
                    "lean": lean,
                    "intensity": round(float(min(1.0, intensity)), 4),
                }
            )

        return {
            "available": True,
            "label": label,
            "ptsd_probability": round(pt_prob, 6),
            "no_ptsd_probability": round(float(probs[no_idx]), 6),
            "token_attributions": attributions,
            "disclaimer": (
                "Gradient salience toward each class logit (not semantic keywords). "
                "Strong tint is suppressed on common English stopwords. "
                "Fine-tuned on sorted_multimodal text; independent of multimodal fusion."
            ),
        }
