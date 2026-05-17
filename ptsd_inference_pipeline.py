"""
PTSD Multimodal Inference Pipeline  ·  Production Edition v3
=============================================================
New in v3:
  27. Modality contribution scoring  — ablation-based, normalized, stable
  28. SHAP explainability             — KernelSHAP on fusion head inputs
  29. Attention visualization         — attn weights from CrossModalAttn
  30. Top-feature report              — per-modality influential dimensions
  31. ExplainabilityResult dataclass  — clean output structure
  32. explain_prediction()            — one-call explainability API
"""

from __future__ import annotations

import asyncio
import base64
import gc
import hashlib
import io
import logging
import os
import pickle
import queue
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from PIL import Image
from sklearn.preprocessing import normalize
from torch.cuda.amp import autocast
from transformers import AutoModel, AutoTokenizer

from text_explainability import TextExplainerModel

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ptsd")

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    _PROM = True
    _REQ_TOTAL = Counter("ptsd_requests_total",  "Total prediction requests")
    _REQ_ERR   = Counter("ptsd_errors_total",     "Failed prediction requests")
    _LATENCY   = Histogram("ptsd_latency_seconds","End-to-end latency",
                           buckets=[1,2,5,10,20,30,60,120])
    _STAGE_LAT = Histogram("ptsd_stage_latency_seconds","Per-stage latency",
                           ["stage"],buckets=[0.5,1,2,5,10,20,30])
    _GPU_MEM   = Gauge("ptsd_gpu_memory_bytes","GPU memory allocated")
    _QUEUE_LEN = Gauge("ptsd_queue_length","Pending requests in queue")
except ImportError:
    _PROM = False
    class _Noop:
        def inc(self,*a,**kw): pass
        def observe(self,*a,**kw): pass
        def set(self,*a,**kw): pass
        def labels(self,*a,**kw): return self
    _REQ_TOTAL=_REQ_ERR=_LATENCY=_STAGE_LAT=_GPU_MEM=_QUEUE_LEN=_Noop()


# ══════════════════════════════════════════════════════════════════════════════
# 1 · CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineCfg:
    text_dim:   int = 768
    audio_dim:  int = 1792
    video_dim:  int = 1792
    hidden_dim: int = 256
    num_heads:  int = 4

    primary_model: str = "early_fusion"
    load_models: List[str] = field(
        default_factory=lambda: [
            "early_fusion",
            "hybrid_fusion",
            "late_fusion",
            "attn_fusion",
        ]
    )

    text_maxlen:       int   = 512
    audio_sr:          int   = 16_000
    audio_chunk_s:     float = 30.0
    n_fft:             int   = 2048
    hop_length:        int   = 512
    n_mels:            int   = 128
    spec_h:            int   = 224
    spec_w:            int   = 224

    frames_per_minute: int   = 30
    max_frames:        int   = 240

    max_video_bytes:   int   = 2 * 1024**3
    max_video_seconds: float = 600.0
    min_audio_rms:     float = 1e-4
    min_transcript_chars: int = 10
    min_audio_seconds: float = 3.0
    min_audio_peak:    float = 0.01
    min_audio_active_ratio: float = 0.1
    audio_activity_window_s: float = 0.5
    require_audio:     bool  = True
    require_face:      bool  = True
    min_face_frames_ratio: float = 0.3
    min_face_frames:      int   = 3
    min_face_area_ratio:   float = 0.02
    min_face_size_ratio:   float = 0.08
    face_check_stride:     int   = 4
    min_face_sharpness:    float = 50.0

    mc_passes:     int  = 30
    use_amp:       bool = True
    quantize_int8: bool = False
    whisper_model: str  = "base"

    max_concurrent: int = 4
    queue_maxsize:  int = 20
    cpu_workers:    int = 4

    timeout_extraction: float = 120.0
    timeout_embedding:  float = 60.0
    timeout_fusion:     float = 10.0

    embedding_cache_size: int = 64

    model_dir:          str = "saved_models"
    scalers_path:       str = "embeddings/scalers.pkl"
    label_encoder_path: str = "embeddings/label_encoder.pkl"

    # ── Explainability config ─────────────────────────────────────────────────
    # Ablation contribution: number of times to repeat ablation per modality
    # (higher = more stable, slower)
    ablation_repeats:   int = 10
    # SHAP: number of background samples for KernelSHAP
    shap_background_n:  int = 50
    # Whether to run SHAP (requires shap package; slower but more detailed)
    run_shap:           bool = False

    # Linguistic explainability (fine-tuned RoBERTa; see train_text_explainer.py)
    text_explainer_path: str = "saved_models/text_ptsd_explainer"
    # Thumbnails for full visual audit JSON payload (all spectrograms + frames)
    visual_audit_thumb_max: int = 128
    visual_audit_jpeg_quality: int = 72
    # Mel chunks for UI only (embedding still uses audio_chunk_s). Shorter → more spectrograms.
    visual_audit_chunk_s: float = 5.0


# ══════════════════════════════════════════════════════════════════════════════
# 2 · TELEMETRY
# ══════════════════════════════════════════════════════════════════════════════

class StageTimer:
    def __init__(self, name: str, metrics: Dict[str, float]):
        self.name=name; self.metrics=metrics; self._t0=0.0
    def __enter__(self): self._t0=time.perf_counter(); return self
    def __exit__(self,*_):
        e=time.perf_counter()-self._t0
        self.metrics[self.name]=round(e,4)
        _STAGE_LAT.labels(stage=self.name).observe(e)
        log.debug(f"  [{self.name}] {e:.3f}s")

def _gpu_stats()->str:
    if not torch.cuda.is_available(): return "CPU"
    alloc=torch.cuda.memory_allocated()/1e6
    resv =torch.cuda.memory_reserved()/1e6
    _GPU_MEM.set(torch.cuda.memory_allocated())
    return f"GPU alloc={alloc:.0f}MB  reserved={resv:.0f}MB"


# ══════════════════════════════════════════════════════════════════════════════
# 3 · LRU EMBEDDING CACHE
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingCache:
    def __init__(self, maxsize:int=64):
        self._store=OrderedDict(); self._maxsize=maxsize; self._lock=threading.Lock()
    @staticmethod
    def hash_bytes(data:bytes)->str: return hashlib.sha256(data).hexdigest()
    def get(self,key:str)->Optional[Tuple]:
        if self._maxsize==0: return None
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key); return self._store[key]
        return None
    def put(self,key:str,value:Tuple):
        if self._maxsize==0: return
        with self._lock:
            self._store[key]=value; self._store.move_to_end(key)
            while len(self._store)>self._maxsize: self._store.popitem(last=False)


# ══════════════════════════════════════════════════════════════════════════════
# 4 · VIDEO INGESTOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RawMedia:
    waveform:   np.ndarray
    frame_pil:  List[Image.Image]
    duration_s: float
    file_hash:  str


class VideoIngestor:
    def __init__(self,cfg:PipelineCfg):
        self.cfg=cfg; self._check_ffmpeg()

    @staticmethod
    def _check_ffmpeg():
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            raise EnvironmentError("ffmpeg/ffprobe not found on PATH.")

    def probe_duration(self,video_path:Path)->float:
        cmd=["ffprobe","-v","quiet","-show_entries","format=duration",
             "-of","csv=p=0",str(video_path)]
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
        try: return float(r.stdout.strip())
        except ValueError: return 0.0

    def _target_frames(self,duration_s:float)->int:
        n=int((duration_s/60.0)*self.cfg.frames_per_minute)
        return max(1,min(n,self.cfg.max_frames))

    @staticmethod
    def _fast_hash(path:Path)->str:
        h=hashlib.sha256()
        with open(path,"rb") as f: h.update(f.read(1024*1024))
        return h.hexdigest()

    def ingest(self,video_path:Path,file_bytes_hint:Optional[bytes]=None)->RawMedia:
        duration=self.probe_duration(video_path)
        if duration<=0: raise ValueError("Could not determine video duration.")
        if duration>self.cfg.max_video_seconds:
            raise ValueError(f"Video too long: {duration:.0f}s")
        file_hash=self._fast_hash(video_path)
        n_frames=self._target_frames(duration)
        log.info(f"Ingest: duration={duration:.1f}s  target_frames={n_frames}")
        waveform=self._extract_audio_pipe(video_path)
        fps_rate=n_frames/duration
        frames=self._extract_frames_pipe(video_path,fps_rate,n_frames)
        log.info(f"Ingest done: audio={len(waveform)/self.cfg.audio_sr:.1f}s frames={len(frames)}")
        return RawMedia(waveform=waveform,frame_pil=frames,
                        duration_s=duration,file_hash=file_hash)

    def _extract_audio_pipe(self,video_path:Path)->np.ndarray:
        cmd=["ffmpeg","-y","-i",str(video_path),"-vn","-acodec","pcm_f32le",
             "-ar",str(self.cfg.audio_sr),"-ac","1","-f","f32le","pipe:1"]
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                               bufsize=4*1024*1024)
        chunks=[]
        while True:
            data=proc.stdout.read(256*1024)
            if not data: break
            chunks.append(np.frombuffer(data,dtype=np.float32))
        proc.stdout.close(); proc.wait()
        if not chunks: return np.zeros(self.cfg.audio_sr,dtype=np.float32)
        return np.concatenate(chunks)

    def _extract_frames_pipe(self,video_path:Path,fps_rate:float,max_n:int)->List[Image.Image]:
        W,H=380,380
        cmd=["ffmpeg","-y","-i",str(video_path),
             "-vf",f"fps={fps_rate:.6f},scale={W}:{H}",
             "-an","-f","rawvideo","-pix_fmt","rgb24","pipe:1"]
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        frame_bytes=W*H*3
        frames=[]
        while len(frames)<max_n:
            raw=proc.stdout.read(frame_bytes)
            if len(raw)<frame_bytes: break
            arr=np.frombuffer(raw,dtype=np.uint8).reshape(H,W,3)
            frames.append(Image.fromarray(arr))
        proc.stdout.close(); proc.wait()
        return frames


# ══════════════════════════════════════════════════════════════════════════════
# 4.5 · INPUT QUALITY GUARDS
# ══════════════════════════════════════════════════════════════════════════════

def _audio_rms(waveform: np.ndarray) -> float:
    if waveform.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(waveform ** 2)))


def _audio_activity_ratio(
    waveform: np.ndarray,
    sr: int,
    window_s: float,
    rms_threshold: float,
) -> float:
    if waveform.size == 0 or sr <= 0:
        return 0.0
    window = max(1, int(window_s * sr))
    total = 0
    active = 0
    for i in range(0, len(waveform), window):
        seg = waveform[i:i + window]
        if seg.size < window * 0.5:
            break
        total += 1
        if _audio_rms(seg) >= rms_threshold:
            active += 1
    if total == 0:
        return 0.0
    return active / float(total)


def _face_presence_stats(frames: List[Image.Image], cfg: PipelineCfg) -> Tuple[float, int, int]:
    if not frames:
        return 0.0, 0, 0
    try:
        import cv2
    except Exception as e:
        raise ValueError(
            "Face guard requires opencv-python. Install opencv-python or set REQUIRE_FACE=0."
        ) from e

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise ValueError(
            "Face guard could not load Haar cascade. Check opencv install or disable REQUIRE_FACE."
        )

    stride = max(1, cfg.face_check_stride)
    total = 0
    hits = 0
    min_size = None

    for idx in range(0, len(frames), stride):
        frame = frames[idx]
        gray = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        if min_size is None:
            min_side = max(16, int(min(h, w) * cfg.min_face_size_ratio))
            min_size = (min_side, min_side)

        if cfg.min_face_sharpness > 0:
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            if sharpness < cfg.min_face_sharpness:
                total += 1
                continue

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=min_size,
        )
        total += 1
        if len(faces) == 0:
            continue
        max_area = max([fw * fh for (_, _, fw, fh) in faces])
        if (max_area / float(h * w)) >= cfg.min_face_area_ratio:
            hits += 1

    if total == 0:
        return 0.0, hits, total
    return hits / float(total), hits, total


# ══════════════════════════════════════════════════════════════════════════════
# 5 · CPU PREPROCESSORS
# ══════════════════════════════════════════════════════════════════════════════

class AudioPreprocessor:
    def __init__(self,cfg:PipelineCfg):
        self.cfg=cfg
        self.chunk_samp=int(cfg.audio_chunk_s*cfg.audio_sr)
        import torchvision.transforms as T
        self._transform=tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1.transforms()

    def _waveform_to_tensor(self,wav:np.ndarray)->torch.Tensor:
        try:
            import torchaudio.transforms as AT
            wav_t=torch.from_numpy(wav).unsqueeze(0)
            mel=AT.MelSpectrogram(
                sample_rate=self.cfg.audio_sr,n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,n_mels=self.cfg.n_mels)(wav_t)
            mel_db=AT.AmplitudeToDB()(mel)
            mel_norm=(mel_db-mel_db.mean())/(mel_db.std()+1e-8)
            img=mel_norm.squeeze(0).numpy()
            img=((img-img.min())/(img.max()-img.min()+1e-8)*255).astype(np.uint8)
            pil=Image.fromarray(img).convert("RGB").resize(
                (self.cfg.spec_w,self.cfg.spec_h))
            return self._transform(pil)
        except Exception as e:
            log.warning(f"Spectrogram error: {e}")
            return torch.zeros(3,self.cfg.spec_h,self.cfg.spec_w)

    def process(self,waveform:np.ndarray)->List[torch.Tensor]:
        rms=float(np.sqrt(np.mean(waveform**2)))
        if rms<self.cfg.min_audio_rms:
            log.warning(f"Audio RMS={rms:.2e} below threshold")
        chunks=[waveform[i:i+self.chunk_samp]
                for i in range(0,max(len(waveform),1),self.chunk_samp)
                if len(waveform[i:i+self.chunk_samp])>self.cfg.audio_sr//4
               ] or [waveform[:self.chunk_samp]]
        return [self._waveform_to_tensor(c) for c in chunks]


class VideoPreprocessor:
    def __init__(self,cfg:PipelineCfg):
        self.cfg=cfg
        self._transform=tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1.transforms()
    def process(self,frames:List[Image.Image])->List[torch.Tensor]:
        out=[]
        for img in frames:
            try: out.append(self._transform(img.convert("RGB")))
            except Exception as e: log.warning(f"Frame transform error: {e}")
        return out


class TextPreprocessor:
    def __init__(self,cfg:PipelineCfg):
        self.cfg=cfg; self.tok=None
    def load(self):
        if self.tok is not None: return
        for name in ["mental/mental-roberta-base","roberta-base"]:
            try:
                self.tok=AutoTokenizer.from_pretrained(name)
                self._model_name=name
                log.info(f"Tokenizer loaded: {name}"); return
            except Exception as e: log.warning(f"Tokenizer {name} failed: {e}")
        raise RuntimeError("No tokenizer available")
    def process(self,transcript:str)->Dict[str,torch.Tensor]:
        self.load()
        if not transcript or len(transcript.strip())<self.cfg.min_transcript_chars:
            transcript="interview audio"
        enc=self.tok(transcript,return_tensors="pt",truncation=True,
                     padding="max_length",max_length=self.cfg.text_maxlen)
        return {k:v.squeeze(0) for k,v in enc.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 6 · GPU EMBEDDING EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════════

class TextEmbedder:
    def __init__(self,cfg:PipelineCfg,device:torch.device):
        self.cfg=cfg; self.device=device
        self.use_amp=cfg.use_amp and device.type=="cuda"
        self.model=None; self.name=None
    def load(self):
        if self.model is not None: return
        for name in ["mental/mental-roberta-base","roberta-base"]:
            try:
                self.model=AutoModel.from_pretrained(name).to(self.device).eval()
                if self.cfg.quantize_int8 and self.device.type=="cpu":
                    self.model=torch.quantization.quantize_dynamic(
                        self.model,{nn.Linear},dtype=torch.qint8)
                self.name=name; log.info(f"Text model loaded: {name}"); return
            except Exception as e: log.warning(f"Text model {name} failed: {e}")
        raise RuntimeError("No text model available")
    @torch.no_grad()
    def embed(self,token_dict:Dict[str,torch.Tensor])->np.ndarray:
        self.load()
        batch={k:v.unsqueeze(0).to(self.device) for k,v in token_dict.items()}
        with autocast(enabled=self.use_amp):
            out=self.model(**batch)
        mask=batch["attention_mask"].unsqueeze(-1).float()
        emb=(out.last_hidden_state*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
        return emb.float().squeeze(0).cpu().numpy()


class VisualEmbedder:
    def __init__(self,cfg:PipelineCfg,device:torch.device,name:str="visual"):
        self.cfg=cfg; self.device=device
        self.use_amp=cfg.use_amp and device.type=="cuda"
        self.name=name; self.model=None
    def load(self):
        if self.model is not None: return
        weights=tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1
        m=tv_models.efficientnet_b4(weights=weights)
        m.classifier=nn.Identity()
        self.model=m.to(self.device).eval()
        if self.cfg.quantize_int8 and self.device.type=="cpu":
            self.model=torch.quantization.quantize_dynamic(
                self.model,{nn.Conv2d,nn.Linear},dtype=torch.qint8)
        log.info(f"EfficientNet-B4 loaded ({self.name})")
    @torch.no_grad()
    def embed(self,tensors:List[torch.Tensor])->np.ndarray:
        self.load()
        if not tensors: return np.zeros(1792,dtype=np.float32)
        embs=[]
        for i in range(0,len(tensors),32):
            chunk=torch.stack(tensors[i:i+32]).to(self.device)
            with autocast(enabled=self.use_amp):
                emb=self.model(chunk).float().cpu().numpy()
            embs.append(emb)
        return np.vstack(embs).mean(axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# 7 · MODEL ARCHITECTURE  (exact mirror of training)
# ══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    def __init__(self,in_dim,out_dim,dropout=0.3):
        super().__init__()
        self.norm=nn.LayerNorm(in_dim)
        self.ff=nn.Sequential(nn.Linear(in_dim,out_dim),nn.SiLU(),nn.Dropout(dropout))
        self.skip=nn.Linear(in_dim,out_dim) if in_dim!=out_dim else nn.Identity()
    def forward(self,x): return self.ff(self.norm(x))+self.skip(x)

class ModalityEncoder(nn.Module):
    def __init__(self,raw_dim,h=256,dropout=0.3):
        super().__init__()
        self.bn=nn.BatchNorm1d(raw_dim)
        self.block1=ResidualBlock(raw_dim,h,dropout=dropout)
        self.block2=ResidualBlock(h,h,dropout=dropout*0.7)
    def forward(self,x):
        x=x.float(); return self.block2(self.block1(self.bn(x)))

class EarlyFusionModel(nn.Module):
    def __init__(self,h,enc_t,enc_a,enc_v):
        super().__init__()
        self._et,self._ea,self._ev=enc_t,enc_a,enc_v
        self.head=nn.Sequential(nn.Linear(h*3,h),nn.LayerNorm(h),nn.SiLU(),nn.Linear(h,2))
    def forward(self,t,a,v):
        return self.head(torch.cat([self._et(t),self._ea(a),self._ev(v)],dim=1))
    def forward_encoded(self,et,ea,ev):
        return self.head(torch.cat([et,ea,ev],dim=1))

class HybridFusionModel(nn.Module):
    def __init__(self,h,enc_t,enc_a,enc_v):
        super().__init__()
        self._et,self._ea,self._ev=enc_t,enc_a,enc_v
        self.gate=nn.Sequential(nn.Linear(h*3,3),nn.Softmax(dim=1))
        self.head=nn.Sequential(nn.Linear(h,h),nn.LayerNorm(h),nn.SiLU(),nn.Linear(h,2))
    def forward(self,t,a,v):
        et,ea,ev=self._et(t),self._ea(a),self._ev(v)
        w=self.gate(torch.cat([et,ea,ev],dim=1))
        return self.head(w[:,0:1]*et+w[:,1:2]*ea+w[:,2:3]*ev)
    def forward_encoded(self,et,ea,ev):
        w=self.gate(torch.cat([et,ea,ev],dim=1))
        return self.head(w[:,0:1]*et+w[:,1:2]*ea+w[:,2:3]*ev)
    def get_gate_weights(self,et,ea,ev)->np.ndarray:
        """Return the 3-way softmax gate weights for this sample. (B,3) -> (3,)"""
        with torch.no_grad():
            w=self.gate(torch.cat([et,ea,ev],dim=1))
        return w.squeeze(0).cpu().numpy()

class LateFusionModel(nn.Module):
    def __init__(self,h,enc_t,enc_a,enc_v):
        super().__init__()
        self._et,self._ea,self._ev=enc_t,enc_a,enc_v
        def branch(): return nn.Sequential(nn.Linear(h,h),nn.LayerNorm(h),nn.SiLU(),nn.Linear(h,2))
        self.tb,self.ab,self.vb=branch(),branch(),branch()
        self.wts=nn.Parameter(torch.ones(3)/3)
    def forward(self,t,a,v):
        et,ea,ev=self._et(t),self._ea(a),self._ev(v)
        w=torch.softmax(self.wts,0)
        return w[0]*self.tb(et)+w[1]*self.ab(ea)+w[2]*self.vb(ev)
    def forward_encoded(self,et,ea,ev):
        w=torch.softmax(self.wts,0)
        return w[0]*self.tb(et)+w[1]*self.ab(ea)+w[2]*self.vb(ev)
    def get_learned_weights(self)->np.ndarray:
        """Return the learned modality weights (normalized). (3,)"""
        return torch.softmax(self.wts,0).detach().cpu().numpy()

class CrossModalAttentionFusion(nn.Module):
    def __init__(self,td,ad,vd,h=256,nh=4):
        super().__init__()
        self.proj_t=nn.Linear(td,h); self.norm_t=nn.LayerNorm(h)
        self.proj_a=nn.Linear(ad,h); self.norm_a=nn.LayerNorm(h)
        self.proj_v=nn.Linear(vd,h); self.norm_v=nn.LayerNorm(h)
        self.cls_token=nn.Parameter(torch.zeros(1,1,h))
        self.pos_emb=nn.Parameter(torch.zeros(1,4,h))
        enc_l=nn.TransformerEncoderLayer(d_model=h,nhead=nh,dim_feedforward=h*4,
                                         dropout=0.1,batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(enc_l,num_layers=2)
        self.norm=nn.LayerNorm(h)
        self.head=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,2))
        self._attn_weights=None  # populated during forward when capture=True
    def forward(self,t,a,v,capture_attn:bool=False):
        et=self.norm_t(self.proj_t(t))
        ea=self.norm_a(self.proj_a(a))
        ev=self.norm_v(self.proj_v(v))
        return self.forward_encoded(et,ea,ev,capture_attn=capture_attn)
    def forward_encoded(self,et,ea,ev,capture_attn:bool=False):
        B=et.size(0)
        seq=torch.cat([self.cls_token.expand(B,-1,-1),
                       torch.stack([et,ea,ev],dim=1)],dim=1)+self.pos_emb
        if capture_attn:
            # Extract attention from last layer manually
            layer=self.transformer.layers[-1]
            with torch.no_grad():
                attn_out,weights=layer.self_attn(
                    self.transformer.layers[-2](seq) if len(self.transformer.layers)>1 else seq,
                    self.transformer.layers[-2](seq) if len(self.transformer.layers)>1 else seq,
                    self.transformer.layers[-2](seq) if len(self.transformer.layers)>1 else seq,
                    need_weights=True,average_attn_weights=True)
            self._attn_weights=weights.detach().cpu().numpy()  # (B,4,4)
        out=self.transformer(seq)
        return self.head(self.norm(out[:,0]))


# ══════════════════════════════════════════════════════════════════════════════
# 8 · MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    CONSTRUCTORS={
        "early_fusion":  EarlyFusionModel,
        "hybrid_fusion": HybridFusionModel,
        "late_fusion":   LateFusionModel,
        "attn_fusion":   CrossModalAttentionFusion,
    }

    def __init__(self,cfg:PipelineCfg,device:torch.device):
        self.cfg=cfg; self.device=device
        self.models:Dict[str,nn.Module]={}

    def _build_encoders(self,ckpt_encoders:dict)->tuple:
        td,ad,vd,h=(self.cfg.text_dim,self.cfg.audio_dim,
                    self.cfg.video_dim,self.cfg.hidden_dim)
        et=ModalityEncoder(td,h,0.3); et.load_state_dict(ckpt_encoders["text"])
        ea=ModalityEncoder(ad,h,0.3); ea.load_state_dict(ckpt_encoders["audio"])
        ev=ModalityEncoder(vd,h,0.3); ev.load_state_dict(ckpt_encoders["video"])
        return et,ea,ev

    def load(self):
        td,ad,vd,h,nh=(self.cfg.text_dim,self.cfg.audio_dim,
                       self.cfg.video_dim,self.cfg.hidden_dim,self.cfg.num_heads)
        for name in self.cfg.load_models:
            ckpt_path=Path(self.cfg.model_dir)/f"{name}_best.pt"
            if not ckpt_path.exists():
                log.warning(f"Checkpoint missing: {ckpt_path}"); continue
            try:
                ckpt=torch.load(ckpt_path,map_location=self.device,weights_only=False)
                et,ea,ev=self._build_encoders(ckpt["encoders"])
                if name=="attn_fusion":
                    model=CrossModalAttentionFusion(td,ad,vd,h,nh)
                else:
                    cls=self.CONSTRUCTORS[name]
                    model=cls(h,et,ea,ev)
                model.load_state_dict(ckpt["fusion"],strict=False)
                model._enc_t=et.to(self.device).eval()
                model._enc_a=ea.to(self.device).eval()
                model._enc_v=ev.to(self.device).eval()
                self.models[name]=model.to(self.device).eval()
                log.info(f"Loaded {name}  ({_gpu_stats()})")
            except Exception as e:
                log.error(f"Failed to load {name}: {e}")
        if not self.models:
            raise RuntimeError("No models loaded — check model_dir")

    def get(self,name:str)->nn.Module:
        if name not in self.models:
            raise KeyError(f"Model '{name}' not loaded. loaded={list(self.models)}")
        return self.models[name]

    def release(self):
        self.models.clear(); gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# 9 · WHISPER SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

class WhisperTranscriber:
    _instance:Optional["WhisperTranscriber"]=None
    _lock=threading.Lock()

    def __init__(self,model_name:str):
        self._name=model_name; self._model=None
        self._model_lock=threading.Lock()

    @classmethod
    def get_instance(cls,model_name:str)->"WhisperTranscriber":
        with cls._lock:
            if cls._instance is None or cls._instance._name!=model_name:
                cls._instance=cls(model_name)
        return cls._instance

    def _load(self):
        with self._model_lock:
            if self._model is None:
                try:
                    import whisper
                    self._model=whisper.load_model(self._name)
                    log.info(f"Whisper '{self._name}' loaded")
                except ImportError:
                    log.warning("openai-whisper not installed")
                except Exception as e:
                    log.warning(f"Whisper load failed: {e}")

    def transcribe(self,waveform:np.ndarray,sr:int)->str:
        self._load()
        if self._model is None: return ""
        try:
            import whisper
            audio=waveform.astype(np.float32)
            if sr!=16_000:
                import librosa
                audio=librosa.resample(audio,orig_sr=sr,target_sr=16_000)
            audio=whisper.pad_or_trim(audio)
            result=self._model.transcribe(audio,language="en",fp16=False)
            return result["text"].strip()
        except Exception as e:
            log.warning(f"Transcription error: {e}"); return ""


# ══════════════════════════════════════════════════════════════════════════════
# 10 · EXPLAINABILITY  ← NEW
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AblationResult:
    normalized: Dict[str, float]
    signed: Dict[str, float]
    raw_signed: Dict[str, float]
    raw_abs: Dict[str, float]


@dataclass
class ExplainabilityResult:
    """
    Structured output from explain_prediction().

    modality_contribution : dict
        Normalized ablation-based contribution of each modality.
        {"text": 0.35, "audio": 0.30, "video": 0.35}
        Stable — averaged over ablation_repeats runs.

    attention_weights : dict | None
        Only for CrossModalAttentionFusion.
        {"cls_to_text": 0.4, "cls_to_audio": 0.3, "cls_to_video": 0.3}
        Derived from the last Transformer layer's attention matrix.

    gate_weights : dict | None
        Only for HybridFusionModel — the learned 3-way softmax gate.
        {"text": 0.35, "audio": 0.30, "video": 0.35}

    learned_weights : dict | None
        Only for LateFusionModel — the trained per-modality scalar weights.

    shap_values : dict | None
        KernelSHAP values per modality (requires run_shap=True in cfg).
        {"text": float, "audio": float, "video": float}
        Positive = pushes toward PTSD. Negative = pushes toward NO PTSD.

    top_text_dims : list[tuple] | None
        Top-10 most influential text embedding dimensions with their
        contribution direction. [(dim_idx, shap_val), ...]
        Requires run_shap=True.

    top_audio_dims : list[tuple] | None
        Same for audio embedding.

    top_video_dims : list[tuple] | None
        Same for video embedding.

    method : str
        Which explainability method was used as primary source for
        modality_contribution. "ablation" | "gate" | "attention" | "learned"
    """
    modality_contribution: Dict[str, float]
    attention_weights:     Optional[Dict[str, float]] = None
    gate_weights:          Optional[Dict[str, float]] = None
    learned_weights:       Optional[Dict[str, float]] = None
    shap_values:           Optional[Dict[str, float]] = None
    top_text_dims:         Optional[List[Tuple[int, float]]] = None
    top_audio_dims:        Optional[List[Tuple[int, float]]] = None
    top_video_dims:        Optional[List[Tuple[int, float]]] = None
    ablation_contribution: Optional[Dict[str, float]] = None
    directional_impact:    Optional[Dict[str, float]] = None
    method:                str = "ablation"

    def to_dict(self) -> dict:
        return {
            "modality_contribution": self.modality_contribution,
            "attention_weights":     self.attention_weights,
            "gate_weights":          self.gate_weights,
            "learned_weights":       self.learned_weights,
            "shap_values":           self.shap_values,
            "top_text_dims":         self.top_text_dims,
            "top_audio_dims":        self.top_audio_dims,
            "top_video_dims":        self.top_video_dims,
            "ablation_contribution": self.ablation_contribution,
            "directional_impact":    self.directional_impact,
            "method":                self.method,
        }


class ModalityExplainer:
    """
    Computes modality contributions and feature-level explanations.

    Method 1 — Ablation (all models, primary method):
        Run the model 4 times:
          full(t,a,v) → baseline PTSD probability
          null(0,a,v) → PTSD prob without text
          null(t,0,v) → PTSD prob without audio
          null(t,a,0) → PTSD prob without video

        Raw contribution of modality m:
          delta_m = |P_full - P_null_m|

        Normalized so contributions sum to 1.0.
        Repeated `ablation_repeats` times with different zero-noise
        perturbations for stability, then averaged.

    Method 2 — Gate weights (HybridFusion only):
        Direct readout of the learned 3-way softmax gate weights.
        Most interpretable for HybridFusion.

    Method 3 — Attention weights (CrossModalAttn only):
        CLS token attention to each modality token from the last
        Transformer layer. Normalized to sum to 1.0.

    Method 4 — Learned weights (LateFusion only):
        Direct readout of the trained nn.Parameter scalar weights.

    Method 5 — KernelSHAP (optional, all models):
        Model-agnostic Shapley values computed on the concatenated
        [text | audio | video] embedding vector.
        Aggregated per modality by summing SHAP values for each
        modality's dimensions.
        Requires: pip install shap
    """

    def __init__(self, cfg: PipelineCfg, device: torch.device):
        self.cfg    = cfg
        self.device = device

    def _ptsd_prob(
        self, model: nn.Module,
        t: torch.Tensor, a: torch.Tensor, v: torch.Tensor
    ) -> float:
        model.eval()
        with torch.no_grad():
            logits = model(t, a, v)
        return torch.softmax(logits, dim=1)[0, 1].item()

    # ── Method 1: Ablation ────────────────────────────────────────────────────

    def ablation_contribution(
        self,
        model: nn.Module,
        t: torch.Tensor,
        a: torch.Tensor,
        v: torch.Tensor,
    ) -> AblationResult:
        """
        Stable ablation over cfg.ablation_repeats runs.
        Each run adds tiny Gaussian noise to zero vectors so the BN
        layers don't receive a perfectly-zero batch (which can cause
        BN instability). Noise scale is 1e-4 — imperceptible to the model.
        """
        n  = self.cfg.ablation_repeats
        td = t.shape[-1]; ad = a.shape[-1]; vd = v.shape[-1]
        noise = 1e-4

        deltas_abs = {"text": [], "audio": [], "video": []}
        deltas_signed = {"text": [], "audio": [], "video": []}
        for _ in range(n):
            p_full  = self._ptsd_prob(model, t, a, v)
            zero_t  = torch.randn(1, td, device=self.device) * noise
            zero_a  = torch.randn(1, ad, device=self.device) * noise
            zero_v  = torch.randn(1, vd, device=self.device) * noise
            delta_t = p_full - self._ptsd_prob(model, zero_t, a, v)
            delta_a = p_full - self._ptsd_prob(model, t, zero_a, v)
            delta_v = p_full - self._ptsd_prob(model, t, a, zero_v)
            deltas_signed["text"].append(delta_t)
            deltas_signed["audio"].append(delta_a)
            deltas_signed["video"].append(delta_v)
            deltas_abs["text"].append(abs(delta_t))
            deltas_abs["audio"].append(abs(delta_a))
            deltas_abs["video"].append(abs(delta_v))

        raw_abs = {k: float(np.mean(v)) for k, v in deltas_abs.items()}
        raw_signed = {k: float(np.mean(v)) for k, v in deltas_signed.items()}
        total_abs = sum(raw_abs.values()) + 1e-9
        normalized = {k: round(v / total_abs, 4) for k, v in raw_abs.items()}
        signed = {k: round(raw_signed[k] / total_abs, 6) for k in raw_signed}
        return AblationResult(
            normalized=normalized,
            signed=signed,
            raw_signed=raw_signed,
            raw_abs=raw_abs,
        )

    # ── Method 2: Gate weights (HybridFusion) ────────────────────────────────

    def gate_weights(
        self,
        model: HybridFusionModel,
        t: torch.Tensor, a: torch.Tensor, v: torch.Tensor,
    ) -> Dict[str, float]:
        model.eval()
        et = model._enc_t(t)
        ea = model._enc_a(a)
        ev = model._enc_v(v)
        w = model.get_gate_weights(et, ea, ev)
        return {"text": round(float(w[0]),4),
                "audio": round(float(w[1]),4),
                "video": round(float(w[2]),4)}

    # ── Method 3: Attention weights (CrossModalAttn) ──────────────────────────

    def attention_weights(
        self,
        model: CrossModalAttentionFusion,
        t: torch.Tensor, a: torch.Tensor, v: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Extract CLS token row of last-layer attention matrix.
        Sequence order: [CLS(0), text(1), audio(2), video(3)]
        CLS attention to each modality token = model's weighting.
        """
        model.eval()
        model(t, a, v, capture_attn=True)  # triggers attention capture
        if model._attn_weights is None:
            # Fallback to ablation if capture failed
            return self.ablation_contribution(model, t, a, v).normalized
        # attn: (B, 4, 4) — average heads already done
        attn = model._attn_weights[0]  # (4, 4)
        cls_row = attn[0, 1:]          # CLS attention to [text, audio, video]
        cls_row = np.clip(cls_row, 0, None)
        total = cls_row.sum() + 1e-9
        return {
            "text":  round(float(cls_row[0]/total), 4),
            "audio": round(float(cls_row[1]/total), 4),
            "video": round(float(cls_row[2]/total), 4),
        }

    # ── Method 4: Learned weights (LateFusion) ────────────────────────────────

    def learned_weights(self, model: LateFusionModel) -> Dict[str, float]:
        w = model.get_learned_weights()
        return {"text": round(float(w[0]),4),
                "audio": round(float(w[1]),4),
                "video": round(float(w[2]),4)}

    # ── Method 5: KernelSHAP (optional) ──────────────────────────────────────

    def shap_contribution(
        self,
        model: nn.Module,
        t: torch.Tensor, a: torch.Tensor, v: torch.Tensor,
        background_t: Optional[np.ndarray] = None,
        background_a: Optional[np.ndarray] = None,
        background_v: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[Dict[str,float]], Optional[List], Optional[List], Optional[List]]:
        """
        KernelSHAP on concatenated [text|audio|video] embedding.
        Returns (per_modality_shap, top_text_dims, top_audio_dims, top_video_dims)
        or (None, None, None, None) if shap not installed.

        top_X_dims: list of (dim_idx, shap_value) for top 10 dims.
        Positive shap_value → that feature dimension pushes toward PTSD.
        """
        try:
            import shap
        except ImportError:
            log.warning("shap not installed — skipping KernelSHAP. pip install shap")
            return None, None, None, None

        td = t.shape[-1]; ad = a.shape[-1]; vd = v.shape[-1]

        # Build background dataset (random noise if no background provided)
        n_bg = self.cfg.shap_background_n
        bg_t = background_t if background_t is not None else                np.random.randn(n_bg, td).astype(np.float32) * 0.1
        bg_a = background_a if background_a is not None else                np.random.randn(n_bg, ad).astype(np.float32) * 0.1
        bg_v = background_v if background_v is not None else                np.random.randn(n_bg, vd).astype(np.float32) * 0.1
        bg   = np.hstack([bg_t, bg_a, bg_v])  # (n_bg, td+ad+vd)

        # Predict function for SHAP: takes (N, td+ad+vd) → (N,) PTSD prob
        def predict_fn(x: np.ndarray) -> np.ndarray:
            model.eval()
            results = []
            for row in x:
                t_ = torch.tensor(row[:td],     dtype=torch.float32).unsqueeze(0).to(self.device)
                a_ = torch.tensor(row[td:td+ad],dtype=torch.float32).unsqueeze(0).to(self.device)
                v_ = torch.tensor(row[td+ad:],  dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = model(t_, a_, v_)
                results.append(torch.softmax(logits,dim=1)[0,1].item())
            return np.array(results)

        x_input = np.hstack([
            t.cpu().numpy(), a.cpu().numpy(), v.cpu().numpy()
        ])  # (1, td+ad+vd)

        explainer  = shap.KernelExplainer(predict_fn, shap.kmeans(bg, 10))
        shap_vals  = explainer.shap_values(x_input, nsamples=100)[0]  # (td+ad+vd,)

        # Aggregate per modality: sum of absolute SHAP values within each block
        sv_t = shap_vals[:td]
        sv_a = shap_vals[td:td+ad]
        sv_v = shap_vals[td+ad:]

        raw = {"text": float(np.abs(sv_t).sum()),
               "audio": float(np.abs(sv_a).sum()),
               "video": float(np.abs(sv_v).sum())}
        total = sum(raw.values()) + 1e-9
        per_modality = {k: round(v/total, 4) for k, v in raw.items()}

        # Top-10 dims per modality (by absolute SHAP value)
        def top10(sv):
            idx = np.argsort(np.abs(sv))[::-1][:10]
            return [(int(i), round(float(sv[i]),6)) for i in idx]

        return per_modality, top10(sv_t), top10(sv_a), top10(sv_v)

    # ── Master explain() ──────────────────────────────────────────────────────

    def explain(
        self,
        model_name: str,
        model: nn.Module,
        t: torch.Tensor, a: torch.Tensor, v: torch.Tensor,
    ) -> ExplainabilityResult:
        """
        Run all applicable explainability methods for the given model type
        and return a unified ExplainabilityResult.
        """
        model.eval()
        attn_w   = None
        gate_w   = None
        learned_w = None
        shap_v   = None
        top_t    = None
        top_a    = None
        top_v    = None
        method   = "ablation"

        # Ablation is always computed (stable, model-agnostic)
        ablation = self.ablation_contribution(model, t, a, v)

        # Model-specific richer methods
        if model_name == "attn_fusion" and isinstance(model, CrossModalAttentionFusion):
            attn_w = self.attention_weights(model, t, a, v)
            method = "attention"
            # Use attention as primary contribution (more principled than ablation for attn models)
            primary = attn_w

        elif model_name == "hybrid_fusion" and isinstance(model, HybridFusionModel):
            gate_w = self.gate_weights(model, t, a, v)
            method = "gate"
            primary = gate_w

        elif model_name == "late_fusion" and isinstance(model, LateFusionModel):
            learned_w = self.learned_weights(model)
            method = "learned"
            primary = learned_w

        else:
            # EarlyFusion or any other: ablation only
            primary = ablation.normalized

        # SHAP (optional, slow)
        if self.cfg.run_shap:
            shap_v, top_t, top_a, top_v = self.shap_contribution(model, t, a, v)

        return ExplainabilityResult(
            modality_contribution = primary,
            attention_weights     = attn_w,
            gate_weights          = gate_w,
            learned_weights       = learned_w,
            shap_values           = shap_v,
            top_text_dims         = top_t,
            top_audio_dims        = top_a,
            top_video_dims        = top_v,
            ablation_contribution = ablation.normalized,
            directional_impact    = ablation.raw_signed,
            method                = method,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 10.5 · EXPLANATION NARRATIVE
# ══════════════════════════════════════════════════════════════════════════════

_PTSD_KEYWORDS = {
    "nightmare", "flashback", "intrusive", "avoid", "avoidance", "hypervigilance",
    "panic", "anxious", "anxiety", "fear", "trauma", "trigger", "sleep",
    "insomnia", "irritable", "angry", "startle", "guilt", "shame",
    "detached", "numb", "sad", "depressed", "hopeless", "unsafe",
    "memory", "concentration", "focus", "fatigue", "stress",
}

_NO_PTSD_KEYWORDS = {
    "calm", "safe", "stable", "hopeful", "positive", "okay", "good", "better",
    "improving", "support", "coping", "resilient", "relaxed", "rested",
    "motivated", "confident", "comfortable", "secure", "happy", "content",
    "grateful", "excited", "sleeping", "sleep", "energy",
}


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    clean = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+", clean)
    return [p.strip() for p in parts if p.strip()]


def _extract_text_evidence(transcript: str, label: str, top_k: int = 3) -> Dict:
    sentences = _split_sentences(transcript)
    if not sentences:
        return {"sentences": [], "keywords": []}

    is_ptsd = label.strip().upper() == "PTSD"
    keyword_set = _PTSD_KEYWORDS if is_ptsd else _NO_PTSD_KEYWORDS
    negation_tokens = {"no", "not", "never", "without"}

    scored = []
    for sent in sentences:
        lowered = sent.lower()
        words = re.findall(r"[a-z']+", lowered)
        direct_matches = [w for w in words if w in keyword_set]

        negated_matches: List[str] = []
        if not is_ptsd:
            for kw in _PTSD_KEYWORDS:
                pattern = rf"\\b({'|'.join(negation_tokens)})\\s+{re.escape(kw)}\\b"
                match = re.search(pattern, lowered)
                if match:
                    negated_matches.append(match.group(0))

        matches = sorted(set(direct_matches + negated_matches))
        score = len(matches)
        scored.append((score, len(sent), sent, matches))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = scored[:top_k]
    keywords = sorted({k for _, _, _, matches in top for k in matches})

    return {
        "sentences": [
            {"text": sent, "score": score, "keywords": matches}
            for score, _, sent, matches in top
        ],
        "keywords": keywords,
    }


def _image_to_base64(
    image: Image.Image,
    max_size: int = 256,
    fmt: str = "JPEG",
    jpeg_quality: int = 82,
) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")
    if max(image.size) > max_size:
        image = image.copy()
        image.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    if fmt.upper() == "JPEG":
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    else:
        image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _audio_chunk_segments(
    waveform: np.ndarray,
    cfg: PipelineCfg,
    chunk_seconds: Optional[float] = None,
) -> List[Tuple[int, int]]:
    sec = cfg.audio_chunk_s if chunk_seconds is None else chunk_seconds
    chunk_samp = int(sec * cfg.audio_sr)
    segments: List[Tuple[int, int]] = []
    for i in range(0, max(len(waveform), 1), chunk_samp):
        seg = waveform[i : i + chunk_samp]
        if len(seg) > cfg.audio_sr // 4:
            segments.append((i, i + len(seg)))
    if not segments and waveform.size > 0:
        end = min(chunk_samp, len(waveform))
        segments.append((0, end))
    return segments


def _build_visual_audit(media: RawMedia, cfg: PipelineCfg) -> Dict:
    sr = cfg.audio_sr
    spectrogram_chunks = []
    for ci, (a, b) in enumerate(
        _audio_chunk_segments(media.waveform, cfg, cfg.visual_audit_chunk_s)
    ):
        patch = _spectrogram_patch(media.waveform, sr, a / sr, b / sr, cfg)
        spectrogram_chunks.append(
            {
                "chunk_index": ci,
                "start_s": round(a / sr, 3),
                "end_s": round(b / sr, 3),
                "image_b64": (
                    _image_to_base64(
                        patch,
                        max_size=cfg.visual_audit_thumb_max,
                        fmt="PNG",
                    )
                    if patch
                    else None
                ),
            }
        )

    fps_rate = len(media.frame_pil) / media.duration_s if media.duration_s > 0 else 0.0
    video_frames = []
    for fi, pil in enumerate(media.frame_pil):
        ts = round(fi / fps_rate, 3) if fps_rate > 0 else 0.0
        video_frames.append(
            {
                "frame_index": fi,
                "timestamp_s": ts,
                "image_b64": _image_to_base64(
                    pil,
                    max_size=cfg.visual_audit_thumb_max,
                    fmt="JPEG",
                    jpeg_quality=cfg.visual_audit_jpeg_quality,
                ),
            }
        )

    return {
        "spectrogram_chunks": spectrogram_chunks,
        "video_frames": video_frames,
    }


def _spectrogram_patch(
    waveform: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    cfg: PipelineCfg,
) -> Optional[Image.Image]:
    try:
        import torchaudio.transforms as AT
        start = max(0, int(start_s * sr))
        end = max(start + 1, int(end_s * sr))
        seg = waveform[start:end]
        if seg.size < sr // 2:
            return None
        wav_t = torch.from_numpy(seg).unsqueeze(0)
        mel = AT.MelSpectrogram(
            sample_rate=sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
        )(wav_t)
        mel_db = AT.AmplitudeToDB()(mel)
        mel_np = mel_db.squeeze(0).numpy()
        mel_np = (mel_np - mel_np.min()) / (mel_np.max() - mel_np.min() + 1e-8)
        img = (mel_np * 255).astype(np.uint8)
        return Image.fromarray(img).convert("RGB").resize((cfg.spec_w, cfg.spec_h))
    except Exception as e:
        log.warning(f"Spectrogram patch error: {e}")
        return None


def _audio_energy_peaks(
    waveform: np.ndarray, sr: int, window_s: float = 5.0, top_k: int = 3
) -> List[Dict]:
    if waveform.size == 0 or sr <= 0:
        return []
    window = int(sr * window_s)
    if window <= 0:
        return []

    peaks = []
    for i in range(0, len(waveform), window):
        seg = waveform[i:i + window]
        if seg.size < window * 0.5:
            break
        rms = float(np.sqrt(np.mean(seg ** 2)))
        peaks.append({
            "start_s": round(i / sr, 2),
            "end_s": round((i + seg.size) / sr, 2),
            "rms": round(rms, 6),
        })

    peaks.sort(key=lambda x: x["rms"], reverse=True)
    return sorted(peaks[:top_k], key=lambda x: x["start_s"])


def _audio_band_energy(waveform: np.ndarray, sr: int) -> Optional[Dict[str, float]]:
    if waveform.size == 0 or sr <= 0:
        return None
    max_samples = int(sr * 10)
    segment = waveform[:max_samples] if waveform.size > max_samples else waveform
    if segment.size < sr:
        return None

    window = np.hanning(segment.size)
    spectrum = np.fft.rfft(segment * window)
    freqs = np.fft.rfftfreq(segment.size, 1 / sr)
    power = np.abs(spectrum) ** 2

    low = power[(freqs > 0) & (freqs <= 300)].sum()
    mid = power[(freqs > 300) & (freqs <= 3000)].sum()
    high = power[(freqs > 3000) & (freqs <= 8000)].sum()
    total = low + mid + high + 1e-9

    return {
        "low": round(float(low / total), 4),
        "mid": round(float(mid / total), 4),
        "high": round(float(high / total), 4),
    }


def _video_motion_peaks(
    frames: List[Image.Image], duration_s: float, top_k: int = 3
) -> List[Dict]:
    if not frames or len(frames) < 2:
        return []
    scores: List[Tuple[float, int]] = []
    prev = None
    for idx, frame in enumerate(frames):
        arr = np.asarray(frame.convert("L"), dtype=np.float32)
        if prev is not None:
            diff = float(np.mean(np.abs(arr - prev)))
            scores.append((diff, idx))
        prev = arr

    scores.sort(reverse=True, key=lambda x: x[0])
    fps_rate = len(frames) / duration_s if duration_s > 0 else 0.0
    peaks = []
    for score, idx in scores[:top_k]:
        timestamp = round(idx / fps_rate, 2) if fps_rate > 0 else 0.0
        peaks.append({
            "timestamp_s": timestamp,
            "score": round(score, 4),
            "frame_index": int(idx),
        })
    return sorted(peaks, key=lambda x: x["timestamp_s"])


def _build_explanation_narrative(
    label: str,
    ptsd_prob: float,
    confidence: str,
    uncertainty: float,
    explainability: Optional[ExplainabilityResult],
    transcript: str,
    media: RawMedia,
    cfg: PipelineCfg,
    visual_audit: Optional[Dict] = None,
    text_model: Optional[Dict] = None,
) -> Dict:
    contribution = explainability.modality_contribution if explainability else {}
    top_modality = None
    if contribution:
        top_modality = max(contribution.items(), key=lambda x: x[1])

    text_evidence = _extract_text_evidence(transcript, label)
    audio_peaks = _audio_energy_peaks(media.waveform, cfg.audio_sr)
    band_energy = _audio_band_energy(media.waveform, cfg.audio_sr)
    video_peaks = _video_motion_peaks(media.frame_pil, media.duration_s)

    spectrogram_patches = []
    for peak in audio_peaks:
        patch = _spectrogram_patch(
            media.waveform,
            cfg.audio_sr,
            peak["start_s"],
            peak["end_s"],
            cfg,
        )
        spectrogram_patches.append({
            **peak,
            "image_b64": _image_to_base64(patch, max_size=240, fmt="PNG") if patch else None,
        })

    key_frames = []
    for peak in video_peaks:
        idx = min(max(peak.get("frame_index", 0), 0), len(media.frame_pil) - 1)
        frame = media.frame_pil[idx] if media.frame_pil else None
        key_frames.append({
            "timestamp_s": peak["timestamp_s"],
            "score": peak["score"],
            "image_b64": _image_to_base64(frame, max_size=240, fmt="JPEG") if frame else None,
        })

    top_name = top_modality[0] if top_modality else "N/A"
    top_pct = round(top_modality[1] * 100, 1) if top_modality else 0.0

    summary = (
        f"Prediction: {label} with {ptsd_prob * 100:.1f}% PTSD probability. "
        f"Top modality: {top_name} at {top_pct:.1f}% contribution. "
        f"Confidence: {confidence} (uncertainty {uncertainty:.3f})."
    )

    payload = {
        "summary": summary,
        "disclaimer": (
            "Evidence highlights are heuristic signals derived from the input and "
            "are not direct feature attributions."
        ),
        "text_evidence": text_evidence,
        "audio_evidence": {
            "energy_peaks": audio_peaks,
            "band_energy": band_energy,
            "spectrogram_patches": spectrogram_patches,
        },
        "video_evidence": {
            "motion_peaks": video_peaks,
            "key_frames": key_frames,
            "frame_count": len(media.frame_pil),
        },
    }
    if visual_audit:
        payload["visual_audit"] = visual_audit
    if text_model:
        payload["text_model"] = text_model
    return payload


# ══════════════════════════════════════════════════════════════════════════════
# 11 · RESULT TYPES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PTSDResult:
    label:            str
    ptsd_probability: float
    confidence:       str
    uncertainty:      float
    ci_lower:         float
    ci_upper:         float
    per_model:        Dict[str, Dict]
    stage_latency:    Dict[str, float]
    explainability:   Optional[ExplainabilityResult] = None   # ← NEW
    explanation:      Optional[Dict] = None
    transcript:       Optional[str] = None
    metadata:         Dict = field(default_factory=dict)

    def to_api_dict(self, include_transcript: bool = False) -> dict:
        d = {
            "label":            self.label,
            "ptsd_probability": self.ptsd_probability,
            "confidence":       self.confidence,
            "uncertainty":      self.uncertainty,
            "ci_lower":         self.ci_lower,
            "ci_upper":         self.ci_upper,
            "per_model":        self.per_model,
            "stage_latency":    self.stage_latency,
            "metadata":         self.metadata,
            "explainability":   self.explainability.to_dict()                                 if self.explainability else None,
            "explanation":      self.explanation,
        }
        if include_transcript:
            d["transcript"] = self.transcript or ""
        return d

    def summary(self) -> str:
        lines = [
            "="*60, "PTSD Risk Prediction", "="*60,
            f"Label           : {self.label}",
            f"P(PTSD)         : {self.ptsd_probability:.4f}",
            f"95% CI          : [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"Confidence      : {self.confidence}",
            f"MC Uncertainty  : {self.uncertainty:.4f}",
        ]
        if self.explainability:
            e = self.explainability
            lines += [
                "-"*60,
                f"Explainability  (method: {e.method})",
                f"  Modality contributions:",
                f"    text  : {e.modality_contribution.get('text',0):.3f}",
                f"    audio : {e.modality_contribution.get('audio',0):.3f}",
                f"    video : {e.modality_contribution.get('video',0):.3f}",
            ]
            if e.shap_values:
                lines.append("  SHAP values (toward PTSD):")
                for k,v in e.shap_values.items():
                    lines.append(f"    {k:6s}: {v:+.4f}")
        lines += ["-"*60, "Stage latency (seconds):"]
        for stage, t in self.stage_latency.items():
            lines.append(f"  {stage:25s}: {t:.3f}s")
        lines.append("="*60)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 12 · CORE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class InferencePipeline:
    def __init__(self, cfg: PipelineCfg):
        self.cfg     = cfg
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = cfg.use_amp and self.device.type == "cuda"

        self.ingestor    = VideoIngestor(cfg)
        self.audio_pre   = AudioPreprocessor(cfg)
        self.video_pre   = VideoPreprocessor(cfg)
        self.text_pre    = TextPreprocessor(cfg)
        self.text_emb    = TextEmbedder(cfg, self.device)
        self.audio_emb   = VisualEmbedder(cfg, self.device, "audio")
        self.video_emb   = VisualEmbedder(cfg, self.device, "video")
        self.registry    = ModelRegistry(cfg, self.device)
        self.transcriber = WhisperTranscriber.get_instance(cfg.whisper_model)
        self.cache       = EmbeddingCache(cfg.embedding_cache_size)
        self.explainer   = ModalityExplainer(cfg, self.device)  # ← NEW
        self.text_explainer = TextExplainerModel(cfg.text_explainer_path, self.device)

        self._scalers   = None
        self._label_enc = None
        self._cpu_pool  = ThreadPoolExecutor(
            max_workers=cfg.cpu_workers, thread_name_prefix="ptsd_cpu")

    @classmethod
    def build(cls, cfg: Optional[PipelineCfg] = None) -> "InferencePipeline":
        cfg = cfg or PipelineCfg()
        p   = cls(cfg)
        p._load_artifacts()
        p.registry.load()
        p.text_pre.load()
        p.text_emb.load()
        p.audio_emb.load()
        p.video_emb.load()
        log.info(f"Pipeline ready on {p.device}  {_gpu_stats()}")
        return p

    def _load_artifacts(self):
        with open(self.cfg.scalers_path, "rb") as f:
            self._scalers = pickle.load(f)
        with open(self.cfg.label_encoder_path, "rb") as f:
            self._label_enc = pickle.load(f)
        log.info(f"Scalers + label encoder loaded — classes: {self._label_enc.classes_}")

    def _validate_media(self, media: RawMedia):
        if self.cfg.require_audio:
            min_samples = int(self.cfg.min_audio_seconds * self.cfg.audio_sr)
            if media.waveform.size < min_samples:
                raise ValueError(
                    "Audio too short or missing. Please include clear spoken audio."
                )
            rms = _audio_rms(media.waveform)
            peak = float(np.max(np.abs(media.waveform))) if media.waveform.size else 0.0
            active_ratio = _audio_activity_ratio(
                media.waveform,
                self.cfg.audio_sr,
                self.cfg.audio_activity_window_s,
                self.cfg.min_audio_rms,
            )
            log.info(
                "Audio gate: seconds=%.2f rms=%.4f peak=%.4f active_ratio=%.2f",
                media.waveform.size / float(self.cfg.audio_sr),
                rms,
                peak,
                active_ratio,
            )
            if rms < self.cfg.min_audio_rms or peak < self.cfg.min_audio_peak:
                raise ValueError(
                    "Audio is too quiet or missing. Please record with clear audio."
                )
            if active_ratio < self.cfg.min_audio_active_ratio:
                raise ValueError(
                    "Audio is mostly silent. Please include clear spoken audio."
                )

        if self.cfg.require_face:
            ratio, hits, total = _face_presence_stats(media.frame_pil, self.cfg)
            log.info(
                "Face gate: ratio=%.2f hits=%d total=%d",
                ratio,
                hits,
                total,
            )
            if hits < self.cfg.min_face_frames or ratio < self.cfg.min_face_frames_ratio:
                raise ValueError(
                    "No clear face detected in enough frames. Please ensure a front-facing, well-lit interview."
                )

    def _scale_normalize(self, raw: np.ndarray, modality: str) -> np.ndarray:
        v = self._scalers[modality].transform(raw.reshape(1, -1))
        return normalize(v).astype(np.float32)

    @torch.no_grad()
    def _mc_predict(
        self, model: nn.Module,
        t: torch.Tensor, a: torch.Tensor, v: torch.Tensor
    ) -> Tuple[float, float]:
        model.eval()
        def enable_dropout(m):
            if isinstance(m, nn.Dropout): m.train()
        model.apply(enable_dropout)
        with torch.no_grad():
            logits_stack = torch.stack([model(t,a,v)
                                        for _ in range(self.cfg.mc_passes)])
        probs = torch.softmax(logits_stack, dim=-1)[:, 0, 1]
        return probs.mean().item(), probs.std().item()

    def predict_video(
        self,
        video_path: Path,
        include_transcript: bool = False,
        explain: bool = True,           # ← NEW flag
        include_full_visual_audit: bool = True,
    ) -> PTSDResult:
        latency: Dict[str, float] = {}
        t_total = time.perf_counter()
        _REQ_TOTAL.inc()

        fsize = video_path.stat().st_size
        if fsize > self.cfg.max_video_bytes:
            raise ValueError(f"File too large: {fsize/1e9:.1f} GB > limit")

        with StageTimer("ingest", latency):
            media = self.ingestor.ingest(video_path)

        with StageTimer("quality_gate", latency):
            self._validate_media(media)

        cached = self.cache.get(media.file_hash)
        if cached:
            log.info("Cache hit — skipping embedding extraction")
            text_raw, audio_raw, video_raw, transcript = cached
        else:
            with StageTimer("preprocess", latency):
                fut_audio = self._cpu_pool.submit(self.audio_pre.process, media.waveform)
                fut_video = self._cpu_pool.submit(self.video_pre.process, media.frame_pil)
                fut_tx    = self._cpu_pool.submit(
                    self.transcriber.transcribe, media.waveform, self.cfg.audio_sr)
                audio_tensors = fut_audio.result(timeout=self.cfg.timeout_extraction)
                video_tensors = fut_video.result(timeout=self.cfg.timeout_extraction)
                transcript    = fut_tx.result(timeout=self.cfg.timeout_extraction)
                token_dict    = self.text_pre.process(transcript)

            with StageTimer("embed_text",  latency): text_raw  = self.text_emb.embed(token_dict)
            with StageTimer("embed_audio", latency): audio_raw = self.audio_emb.embed(audio_tensors)
            with StageTimer("embed_video", latency): video_raw = self.video_emb.embed(video_tensors)
            self.cache.put(media.file_hash, (text_raw, audio_raw, video_raw, transcript))

        n_audio_chunks = (
            len(audio_tensors)
            if not cached
            else len(_audio_chunk_segments(media.waveform, self.cfg))
        )

        t_np = self._scale_normalize(text_raw,  "text")
        a_np = self._scale_normalize(audio_raw, "audio")
        v_np = self._scale_normalize(video_raw, "video")

        t = torch.tensor(t_np, dtype=torch.float32).to(self.device)
        a = torch.tensor(a_np, dtype=torch.float32).to(self.device)
        v = torch.tensor(v_np, dtype=torch.float32).to(self.device)

        per_model: Dict[str, Dict] = {}
        with StageTimer("fusion", latency):
            for name, model in self.registry.models.items():
                with torch.no_grad(), autocast(enabled=False):
                    logits = model(t, a, v)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                per_model[name] = {
                    "prob":  round(prob, 6),
                    "label": self._label_enc.inverse_transform([int(prob >= 0.5)])[0],
                }

        primary_name = self.cfg.primary_model
        if primary_name not in per_model:
            primary_name = list(self.registry.models)[0]

        with StageTimer("mc_dropout", latency):
            mc_mean, mc_std = self._mc_predict(
                self.registry.get(primary_name), t, a, v)

        per_model[primary_name]["mc_prob"] = round(mc_mean, 6)
        per_model[primary_name]["mc_std"]  = round(mc_std, 6)

        # ── Explainability ────────────────────────────────────────────────────
        explain_result: Optional[ExplainabilityResult] = None
        if explain:
            with StageTimer("explain", latency):
                try:
                    explain_result = self.explainer.explain(
                        primary_name,
                        self.registry.get(primary_name),
                        t, a, v,
                    )
                    # Attach contribution to per_model entry for easy access
                    per_model[primary_name]["modality_contribution"] =                         explain_result.modality_contribution
                except Exception as e:
                    log.warning(f"Explainability failed (non-fatal): {e}")

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        ptsd_prob  = mc_mean
        label      = self._label_enc.inverse_transform([int(ptsd_prob >= 0.5)])[0]
        ci_lower   = float(np.clip(ptsd_prob - 1.96*mc_std, 0, 1))
        ci_upper   = float(np.clip(ptsd_prob + 1.96*mc_std, 0, 1))
        confidence = ("High"   if mc_std < 0.05
                      else "Medium" if mc_std < 0.15 else "Low")

        visual_audit_payload: Optional[Dict] = None
        if explain and include_full_visual_audit:
            try:
                visual_audit_payload = _build_visual_audit(media, self.cfg)
            except Exception as e:
                log.warning(f"Visual audit failed (non-fatal): {e}")

        text_model_payload: Optional[Dict] = None
        if explain:
            try:
                text_model_payload = self.text_explainer.explain(transcript or "")
            except Exception as e:
                log.warning(f"Text explainer failed (non-fatal): {e}")
                text_model_payload = None
            if text_model_payload is None:
                tx = (transcript or "").strip()
                if len(tx) < self.cfg.min_transcript_chars:
                    text_model_payload = {
                        "available": False,
                        "message": "Transcript too short or missing for text attribution.",
                    }
                elif not self.text_explainer.is_available():
                    text_model_payload = {
                        "available": False,
                        "message": (
                            "Fine-tuned text checkpoint not found. Run: "
                            "python train_text_explainer.py --data-root sorted_multimodal/text"
                        ),
                    }
                else:
                    text_model_payload = {
                        "available": False,
                        "message": "Text attribution failed for this input.",
                    }

        explanation_payload: Optional[Dict] = None
        if explain:
            try:
                explanation_payload = _build_explanation_narrative(
                    label=label,
                    ptsd_prob=ptsd_prob,
                    confidence=confidence,
                    uncertainty=mc_std,
                    explainability=explain_result,
                    transcript=transcript or "",
                    media=media,
                    cfg=self.cfg,
                    visual_audit=visual_audit_payload,
                    text_model=text_model_payload,
                )
            except Exception as e:
                log.warning(f"Explanation narrative failed (non-fatal): {e}")

        total_lat = time.perf_counter() - t_total
        latency["TOTAL"] = round(total_lat, 4)
        _LATENCY.observe(total_lat)
        log.info(f"Prediction done in {total_lat:.2f}s  label={label}  P={ptsd_prob:.4f}")

        return PTSDResult(
            label=label,
            ptsd_probability=round(ptsd_prob, 6),
            confidence=confidence,
            uncertainty=round(mc_std, 6),
            ci_lower=round(ci_lower, 6),
            ci_upper=round(ci_upper, 6),
            per_model=per_model,
            stage_latency=latency,
            explainability=explain_result,
            explanation=explanation_payload,
            transcript=transcript if include_transcript else None,
            metadata={
                "video":          str(video_path),
                "duration_s":     round(media.duration_s, 1),
                "n_frames":       len(media.frame_pil),
                "n_audio_chunks": n_audio_chunks,
                "primary_model":  primary_name,
                "device":         str(self.device),
                "cache_hit":      cached is not None,
            },
        )

    async def predict_video_async(
        self, video_path: Path,
        include_transcript: bool = False,
        explain: bool = True,
        include_full_visual_audit: bool = True,
        timeout: float = 240.0,
    ) -> PTSDResult:
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: self.predict_video(
                    video_path,
                    include_transcript,
                    explain,
                    include_full_visual_audit,
                )),
            timeout=timeout,
        )

    def shutdown(self):
        self._cpu_pool.shutdown(wait=True)
        self.registry.release()
        log.info("Pipeline shut down.")


# ══════════════════════════════════════════════════════════════════════════════
# 13 · INFERENCE WORKER
# ══════════════════════════════════════════════════════════════════════════════

class InferenceWorker:
    def __init__(self, pipeline: InferencePipeline):
        self.pipeline = pipeline
        self._queue: asyncio.Queue = asyncio.Queue(
            maxsize=pipeline.cfg.queue_maxsize)
        self._sem     = asyncio.Semaphore(pipeline.cfg.max_concurrent)
        self._running = False

    async def submit(
        self, video_path: Path,
        include_transcript: bool = False,
        explain: bool = True,
        include_full_visual_audit: bool = True,
    ) -> PTSDResult:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        try:
            self._queue.put_nowait(
                (video_path, include_transcript, explain, include_full_visual_audit, fut)
            )
            _QUEUE_LEN.set(self._queue.qsize())
        except asyncio.QueueFull:
            raise RuntimeError(
                f"Server overloaded — queue full ({self.pipeline.cfg.queue_maxsize} slots).")
        return await fut

    async def run(self):
        self._running = True
        log.info("InferenceWorker started")
        while self._running:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            video_path, inc_tx, do_explain, full_audit, fut = item
            async with self._sem:
                _QUEUE_LEN.set(self._queue.qsize())
                try:
                    result = await self.pipeline.predict_video_async(
                        video_path,
                        inc_tx,
                        do_explain,
                        full_audit,
                        timeout=(self.pipeline.cfg.timeout_extraction +
                                 self.pipeline.cfg.timeout_embedding +
                                 self.pipeline.cfg.timeout_fusion + 180),
                    )
                    fut.set_result(result)
                except Exception as e:
                    _REQ_ERR.inc(); fut.set_exception(e)
                finally:
                    self._queue.task_done()

    def stop(self): self._running = False


# ══════════════════════════════════════════════════════════════════════════════
# 14 · CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, json as _json

    ap = argparse.ArgumentParser(description="PTSD Multimodal Inference v3")
    ap.add_argument("video")
    ap.add_argument("--model-dir",   default="saved_models")
    ap.add_argument("--scalers",     default="embeddings/scalers.pkl")
    ap.add_argument("--label-enc",   default="embeddings/label_encoder.pkl")
    ap.add_argument("--primary",     default="early_fusion")
    ap.add_argument(
        "--load-models",
        default="early_fusion,hybrid_fusion,late_fusion,attn_fusion",
    )
    ap.add_argument("--whisper",     default="base")
    ap.add_argument("--mc-passes",   type=int, default=30)
    ap.add_argument("--transcript",  action="store_true")
    ap.add_argument("--no-explain",  action="store_true")
    ap.add_argument(
        "--no-full-visual-audit",
        action="store_true",
        help="Omit per-chunk spectrograms and all frames from explanation payload",
    )
    ap.add_argument("--shap",        action="store_true",
                    help="Run KernelSHAP (slow, requires pip install shap)")
    ap.add_argument("--json",        action="store_true")
    ap.add_argument("--no-amp",      action="store_true")
    ap.add_argument("--int8",        action="store_true")
    ap.add_argument("--prom-port",   type=int, default=0)
    args = ap.parse_args()

    if args.prom_port and _PROM:
        start_http_server(args.prom_port)

    cfg = PipelineCfg(
        model_dir=args.model_dir,
        scalers_path=args.scalers,
        label_encoder_path=args.label_enc,
        primary_model=args.primary,
        load_models=args.load_models.split(","),
        whisper_model=args.whisper,
        mc_passes=args.mc_passes,
        use_amp=not args.no_amp,
        quantize_int8=args.int8,
        run_shap=args.shap,
    )

    pipe   = InferencePipeline.build(cfg)
    result = pipe.predict_video(
        Path(args.video),
        include_transcript=args.transcript,
        explain=not args.no_explain,
        include_full_visual_audit=not args.no_full_visual_audit,
    )

    if args.json:
        print(_json.dumps(result.to_api_dict(args.transcript), indent=2))
    else:
        print(result.summary())

    pipe.shutdown()