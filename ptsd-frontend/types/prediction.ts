export interface ExplainabilityResult {
  modality_contribution: Record<string, number>;
  attention_weights?: Record<string, number> | null;
  gate_weights?: Record<string, number> | null;
  learned_weights?: Record<string, number> | null;
  shap_values?: Record<string, number> | null;
  top_text_dims?: [number, number][] | null;
  top_audio_dims?: [number, number][] | null;
  top_video_dims?: [number, number][] | null;
  ablation_contribution?: Record<string, number> | null;
  directional_impact?: Record<string, number> | null;
  method: string;
}

export interface ModelBreakdown {
  prob: number;
  label: string;
  mc_prob?: number;
  mc_std?: number;
  modality_contribution?: Record<string, number>;
}

export interface PredictionMetadata {
  video: string;
  duration_s: number;
  n_frames: number;
  n_audio_chunks: number | "cached";
  primary_model: string;
  device: string;
  cache_hit: boolean;
}

export interface EvidenceSentence {
  text: string;
  score: number;
  keywords: string[];
}

export interface VisualAuditSpectrogramChunk {
  chunk_index: number;
  start_s: number;
  end_s: number;
  image_b64?: string | null;
}

export interface VisualAuditVideoFrame {
  frame_index: number;
  timestamp_s: number;
  image_b64?: string | null;
}

export interface VisualAuditPayload {
  spectrogram_chunks: VisualAuditSpectrogramChunk[];
  video_frames: VisualAuditVideoFrame[];
}

export interface TextModelTokenAttr {
  token: string;
  score_ptsd: number;
  score_no_ptsd: number;
  lean: "PTSD" | "NO PTSD" | "neutral";
  /** 0–1 salience for background tint */
  intensity?: number;
}

export interface TextModelExplanation {
  available?: boolean;
  message?: string;
  label?: string;
  ptsd_probability?: number;
  no_ptsd_probability?: number;
  token_attributions?: TextModelTokenAttr[];
  disclaimer?: string;
}

export interface ExplanationNarrative {
  summary: string;
  disclaimer?: string;
  text_evidence: {
    sentences: EvidenceSentence[];
    keywords: string[];
  };
  audio_evidence: {
    energy_peaks: Array<{ start_s: number; end_s: number; rms: number }>;
    band_energy?: { low: number; mid: number; high: number } | null;
    spectrogram_patches?: Array<{
      start_s: number;
      end_s: number;
      rms: number;
      image_b64?: string | null;
    }>;
  };
  video_evidence: {
    motion_peaks: Array<{ timestamp_s: number; score: number }>;
    key_frames?: Array<{
      timestamp_s: number;
      score: number;
      image_b64?: string | null;
    }>;
    frame_count: number;
  };
  visual_audit?: VisualAuditPayload | null;
  text_model?: TextModelExplanation | null;
}

export interface PredictionResponse {
  label: "PTSD" | "NO PTSD";
  ptsd_probability: number;
  confidence: string;
  uncertainty: number;
  ci_lower: number;
  ci_upper: number;
  transcript?: string;
  stage_latency: Record<string, number>;
  per_model: Record<string, ModelBreakdown>;
  metadata?: PredictionMetadata;
  explainability?: ExplainabilityResult | null;
  explanation?: ExplanationNarrative | null;
}

export type AppState = "idle" | "uploading" | "processing" | "results" | "error";

export interface UploadedFile {
  file: File;
  name: string;
  sizeBytes: number;
}
