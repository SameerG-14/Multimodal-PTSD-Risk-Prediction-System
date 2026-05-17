"""
FastAPI Production Web Integration v3 — PTSD Multimodal Inference
=================================================================
New in v3:
  - /predict accepts explain=true query param
  - /explain endpoint for post-hoc explanation of a cached prediction
  - ExplainabilityResult surfaced in all prediction responses
  - include_explanation flag in InferenceWorker.submit()
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from ptsd_inference_pipeline import (
    InferencePipeline,
    InferenceWorker,
    PipelineCfg,
)

log = logging.getLogger("ptsd.web")


def _cfg_from_env() -> PipelineCfg:
    return PipelineCfg(
        model_dir          = os.getenv("MODEL_DIR",       "saved_models"),
        scalers_path       = os.getenv("SCALERS_PATH",    "embeddings/scalers.pkl"),
        label_encoder_path = os.getenv("LABEL_ENC_PATH",  "embeddings/label_encoder.pkl"),
        primary_model      = os.getenv("PRIMARY_MODEL",   "early_fusion"),
        load_models        = os.getenv(
            "LOAD_MODELS",
            "early_fusion,hybrid_fusion,late_fusion,attn_fusion",
        ).split(","),
        whisper_model      = os.getenv("WHISPER_MODEL",   "base"),
        max_concurrent     = int(os.getenv("MAX_CONCURRENT",    "4")),
        queue_maxsize      = int(os.getenv("QUEUE_MAXSIZE",     "20")),
        cpu_workers        = int(os.getenv("CPU_WORKERS",       "4")),
        mc_passes          = int(os.getenv("MC_PASSES",         "30")),
        ablation_repeats   = int(os.getenv("ABLATION_REPEATS",  "10")),
        run_shap           = os.getenv("RUN_SHAP", "0") == "1",
        require_audio      = os.getenv("REQUIRE_AUDIO", "1") == "1",
        require_face       = os.getenv("REQUIRE_FACE", "1") == "1",
        min_audio_rms      = float(os.getenv("MIN_AUDIO_RMS", "1e-4")),
        min_audio_seconds  = float(os.getenv("MIN_AUDIO_SECONDS", "3.0")),
        min_audio_peak     = float(os.getenv("MIN_AUDIO_PEAK", "0.01")),
        min_audio_active_ratio = float(os.getenv("MIN_AUDIO_ACTIVE_RATIO", "0.1")),
        audio_activity_window_s = float(os.getenv("AUDIO_ACTIVITY_WINDOW_S", "0.5")),
        min_face_frames_ratio = float(os.getenv("MIN_FACE_FRAMES_RATIO", "0.3")),
        min_face_frames    = int(os.getenv("MIN_FACE_FRAMES", "3")),
        min_face_area_ratio   = float(os.getenv("MIN_FACE_AREA_RATIO", "0.02")),
        min_face_size_ratio   = float(os.getenv("MIN_FACE_SIZE_RATIO", "0.08")),
        face_check_stride     = int(os.getenv("FACE_CHECK_STRIDE", "4")),
        min_face_sharpness    = float(os.getenv("MIN_FACE_SHARPNESS", "50.0")),
        text_explainer_path   = os.getenv(
            "TEXT_EXPLAINER_PATH", "saved_models/text_ptsd_explainer"),
        visual_audit_thumb_max = int(os.getenv("VISUAL_AUDIT_THUMB_MAX", "128")),
        visual_audit_jpeg_quality = int(os.getenv("VISUAL_AUDIT_JPEG_QUALITY", "72")),
        visual_audit_chunk_s = float(os.getenv("VISUAL_AUDIT_CHUNK_S", "5.0")),
    )


ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

_pipeline: Optional[InferencePipeline] = None
_worker:   Optional[InferenceWorker]   = None
_worker_task: Optional[asyncio.Task]   = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _worker, _worker_task

    cfg       = _cfg_from_env()
    _pipeline = InferencePipeline.build(cfg)
    _worker   = InferenceWorker(_pipeline)
    _worker_task = asyncio.create_task(_worker.run(), name="inference_worker")
    log.info("InferenceWorker task started")

    prom_port = int(os.getenv("PROM_PORT", "0"))
    if prom_port:
        try:
            from prometheus_client import start_http_server
            start_http_server(prom_port)
            log.info(f"Prometheus metrics on :{prom_port}")
        except ImportError:
            log.warning("prometheus_client not installed")

    yield

    _worker.stop()
    if _worker_task:
        _worker_task.cancel()
    _pipeline.shutdown()
    log.info("Shutdown complete")


app = FastAPI(
    title="PTSD Risk Prediction API",
    description="Multimodal PTSD risk inference with modality contribution explainability.",
    version="3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _validate_extension(filename: str):
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400, f"Unsupported format '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")
    return ext


async def _stream_to_temp(video: UploadFile, ext: str) -> Path:
    """Stream-write upload to a temp file. Returns path."""
    tmp_dir  = tempfile.mkdtemp(prefix="ptsd_req_")
    tmp_path = Path(tmp_dir) / f"input{ext}"
    CHUNK    = 1 * 1024 * 1024
    total    = 0
    max_b    = _pipeline.cfg.max_video_bytes
    async with aiofiles.open(tmp_path, "wb") as out:
        while chunk := await video.read(CHUNK):
            total += len(chunk)
            if total > max_b:
                raise HTTPException(
                    413,
                    f"File too large (>{max_b/1e9:.1f} GB limit). "
                    "Trim the interview or use a shorter clip.")
            await out.write(chunk)
    log.info(f"Upload saved: {total/1e6:.1f} MB → {tmp_path}")
    return tmp_path


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["ops"])
async def health():
    """Liveness probe."""
    loaded = list(_pipeline.registry.models.keys()) if _pipeline else []
    return {"status": "ok", "loaded_models": loaded}


@app.get("/ready", tags=["ops"])
async def readiness():
    """Readiness probe."""
    if _pipeline is None or _worker is None:
        raise HTTPException(503, "Pipeline not ready")
    return {"status": "ready",
            "queue_depth": _worker._queue.qsize() if _worker else -1}


@app.post("/predict", tags=["inference"])
async def predict(
    video:                   UploadFile = File(..., description="Interview video file"),
    include_transcript:      bool       = Form(False),
    include_explanation:     bool       = Form(True,
        description="Include modality contribution + explainability in response"),
    include_full_visual_audit: bool    = Form(True,
        description="Include every mel spectrogram chunk and sampled frame (large payload)"),
    request:                 Request    = None,
):
    """
    Upload an interview video and receive a PTSD risk prediction.

    Response includes:
    - label, ptsd_probability, confidence, uncertainty, ci_lower, ci_upper
    - per_model scores
    - explainability.modality_contribution: how much each modality contributed
    - explainability.method: which method was used (ablation/gate/attention/learned)
    - explainability.gate_weights / attention_weights / learned_weights if applicable
    - explainability.shap_values if run_shap=True in server config
    - stage_latency breakdown
    """
    if _worker is None:
        raise HTTPException(503, "Service not ready")

    ext = _validate_extension(video.filename)
    tmp_path: Optional[Path] = None

    try:
        tmp_path = await _stream_to_temp(video, ext)

        try:
            result = await _worker.submit(
                tmp_path,
                include_transcript,
                include_explanation,
                include_full_visual_audit,
            )
        except RuntimeError as e:
            raise HTTPException(503, str(e))

        return JSONResponse(result.to_api_dict(include_transcript))

    except HTTPException:
        raise
    except asyncio.TimeoutError:
        raise HTTPException(504, "Inference timed out.")
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        log.exception("Unhandled inference error")
        raise HTTPException(500, f"Inference failed: {type(e).__name__}: {e}")
    finally:
        if tmp_path is not None:
            shutil.rmtree(tmp_path.parent, ignore_errors=True)


@app.get("/metrics/summary", tags=["ops"])
async def metrics_summary():
    """Quick stats snapshot."""
    import gc, torch
    stats = {
        "queue_depth":    _worker._queue.qsize() if _worker else -1,
        "loaded_models":  list(_pipeline.registry.models.keys()) if _pipeline else [],
        "cache_size":     len(_pipeline.cache._store) if _pipeline else 0,
        "python_gc":      gc.get_count(),
    }
    if torch.cuda.is_available():
        stats["gpu_allocated_mb"] = round(torch.cuda.memory_allocated()/1e6, 1)
        stats["gpu_reserved_mb"]  = round(torch.cuda.memory_reserved()/1e6,  1)
    return stats
