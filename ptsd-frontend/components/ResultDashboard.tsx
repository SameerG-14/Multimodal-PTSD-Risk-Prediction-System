"use client";

import React, { useEffect, useRef, useState } from "react";
import { PredictionResponse, TextModelExplanation, TextModelTokenAttr } from "@/types/prediction";
import { formatLatency, formatStageName } from "@/lib/api";
import {
  IconShield,
  IconInfo,
  IconClock,
  IconRefreshCw,
  IconActivity,
  IconTrendingUp,
  IconBrain,
  IconCpu,
  IconX,
} from "./Icons";

// ─── Tooltip ──────────────────────────────────────────────────────────────────
function Tooltip({ children, text }: { children: React.ReactNode; text: string }) {
  return (
    <div className="tooltip-wrapper">
      {children}
      <div className="tooltip-content">{text}</div>
    </div>
  );
}

function formatTokenizerPiece(tok: string): string {
  if (!tok) return "";
  return tok.replace(/Ġ/g, " ").replace(/##/g, "");
}

function tokenLeanClass(lean: string): string {
  if (lean === "PTSD") return "token-lean-ptsd";
  if (lean === "NO PTSD") return "token-lean-no";
  return "token-lean-neutral";
}

function tokenHighlightStyle(a: TextModelTokenAttr): React.CSSProperties {
  const i = Math.min(1, Math.max(0, a.intensity ?? 0.35));
  if (a.lean === "PTSD") {
    return { backgroundColor: `rgba(239, 68, 68, ${0.14 + 0.52 * i})` };
  }
  if (a.lean === "NO PTSD") {
    return { backgroundColor: `rgba(16, 185, 129, ${0.14 + 0.52 * i})` };
  }
  return { backgroundColor: `rgba(99, 134, 194, ${0.07 + 0.18 * i})` };
}

function TextModelTokenStrip({ model }: { model: TextModelExplanation }) {
  const attrs = model.token_attributions ?? [];
  if (attrs.length === 0) return null;
  return (
    <div className="token-strip" dir="ltr">
      {attrs.map((a, idx) => (
        <span
          key={`${idx}-${a.token}`}
          className={`token-piece ${tokenLeanClass(a.lean)}`}
          style={tokenHighlightStyle(a)}
          title={`PTSD salience ${a.score_ptsd.toFixed(2)} · NO PTSD salience ${a.score_no_ptsd.toFixed(2)}`}
        >
          {formatTokenizerPiece(a.token)}
        </span>
      ))}
    </div>
  );
}

// ─── Gauge SVG ────────────────────────────────────────────────────────────────
function ProbabilityGauge({ value, isElevated }: { value: number; isElevated: boolean }) {
  const r = 70;
  const cx = 90;
  const cy = 90;
  const strokeW = 10;
  const circumference = Math.PI * r; // half arc
  const offset = circumference * (1 - value);

  const trackColor = "rgba(99,134,194,0.12)";
  const fillColor = isElevated
    ? "url(#gaugeRed)"
    : "url(#gaugeGreen)";

  return (
    <svg
      width="180"
      height="110"
      viewBox="0 0 180 110"
      style={{ display: "block", margin: "0 auto" }}
    >
      <defs>
        <linearGradient id="gaugeRed" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#dc2626" />
          <stop offset="100%" stopColor="#ef4444" />
        </linearGradient>
        <linearGradient id="gaugeGreen" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#059669" />
          <stop offset="100%" stopColor="#10b981" />
        </linearGradient>
      </defs>
      {/* Track */}
      <circle
        cx={cx}
        cy={cy}
        r={r}
        fill="none"
        stroke={trackColor}
        strokeWidth={strokeW}
        strokeDasharray={`${circumference} ${circumference}`}
        strokeDashoffset={0}
        strokeLinecap="round"
        transform="rotate(180 90 90)"
      />
      {/* Fill */}
      <circle
        cx={cx}
        cy={cy}
        r={r}
        fill="none"
        stroke={fillColor}
        strokeWidth={strokeW}
        strokeDasharray={`${circumference} ${circumference}`}
        strokeDashoffset={offset}
        strokeLinecap="round"
        transform="rotate(180 90 90)"
        style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1)" }}
      />
      {/* Percentage text */}
      <text
        x={cx}
        y={cy - 8}
        textAnchor="middle"
        fill={isElevated ? "#f87171" : "#34d399"}
        fontSize="22"
        fontWeight="700"
        fontFamily="Inter, sans-serif"
      >
        {Math.round(value * 100)}%
      </text>
      <text
        x={cx}
        y={cy + 14}
        textAnchor="middle"
        fill="#4a6080"
        fontSize="11"
        fontFamily="Inter, sans-serif"
      >
        RISK PROBABILITY
      </text>
    </svg>
  );
}

// ─── Confidence Badge ─────────────────────────────────────────────────────────
function ConfidenceBadge({ level }: { level: string }) {
  const lower = level.toLowerCase();
  let cls = "badge ";
  if (lower.includes("high")) cls += "badge-high";
  else if (lower.includes("medium") || lower.includes("mod")) cls += "badge-medium";
  else cls += "badge-low";

  return <span className={cls}>{level}</span>;
}

// ─── Main Dashboard ───────────────────────────────────────────────────────────
interface ResultDashboardProps {
  result: PredictionResponse;
  onReset: () => void;
}

export default function ResultDashboard({ result, onReset }: ResultDashboardProps) {
  const isElevated = result.ptsd_probability >= 0.5;
  const riskLabel = isElevated ? "Elevated risk signal" : "Lower risk signal";
  const isPtsd = isElevated;
  const probabilityPct = Math.round(result.ptsd_probability * 100);
  const explainability = result.explainability ?? undefined;
  const explanation = result.explanation ?? undefined;
  const visualAudit = explanation?.visual_audit;
  const textModelExpl = explanation?.text_model;
  const auditSpectrograms = visualAudit?.spectrogram_chunks ?? [];
  const auditFrames = visualAudit?.video_frames ?? [];
  const primaryModel = result.metadata?.primary_model ?? Object.keys(result.per_model ?? {})[0];
  const modelEntries = Object.entries(result.per_model ?? {});
  const contribution =
    explainability?.modality_contribution ??
    (primaryModel ? result.per_model?.[primaryModel]?.modality_contribution : undefined);
  const contributionEntries = contribution ? Object.entries(contribution) : [];
  const [isExplainOpen, setIsExplainOpen] = useState(false);

  const formatPercent = (value?: number) => {
    if (value === undefined || value === null || Number.isNaN(value)) return "N/A";
    return `${Math.round(value * 100)}%`;
  };

  const formatSigned = (value?: number) => {
    if (value === undefined || value === null || Number.isNaN(value)) return "N/A";
    const sign = value > 0 ? "+" : "";
    return `${sign}${value.toFixed(3)}`;
  };

  const formatPp = (value?: number) => {
    if (value === undefined || value === null || Number.isNaN(value)) return "N/A";
    const sign = value > 0 ? "+" : "";
    return `${sign}${(value * 100).toFixed(1)} pp`;
  };

  const formatSeconds = (value?: number) => {
    if (value === undefined || value === null || Number.isNaN(value)) return "N/A";
    const total = Math.max(0, value);
    const m = Math.floor(total / 60);
    const s = Math.floor(total % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  const normalizeSummary = (text?: string) => {
    if (!text) return "";
    return text
      .replace(/NO PTSD/gi, "lower risk")
      .replace(/\bPTSD\b/gi, "higher risk");
  };

  const narrativeSummary = normalizeSummary(explanation?.summary) ||
    `Risk signal: ${riskLabel}. Estimated risk probability ${formatPercent(result.ptsd_probability)}.`;

  const highlightVariant = isElevated ? "cue-highlight-ptsd" : "cue-highlight-safe";

  const highlightText = (text: string, terms: string[]) => {
    const cleaned = (terms || []).map((term) => term.trim()).filter(Boolean);
    if (cleaned.length === 0) return text;
    const escaped = cleaned
      .slice()
      .sort((a, b) => b.length - a.length)
      .map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    if (escaped.length === 0) return text;
    const pattern = escaped.join("|");
    const splitRegex = new RegExp(`(${pattern})`, "ig");
    const matchRegex = new RegExp(`^(${pattern})$`, "i");
    const parts = text.split(splitRegex);
    return parts.map((part, idx) =>
      matchRegex.test(part) ? (
        <span key={`${part}-${idx}`} className={`cue-highlight ${highlightVariant}`}>
          {part}
        </span>
      ) : (
        <React.Fragment key={`${part}-${idx}`}>{part}</React.Fragment>
      )
    );
  };

  const weightGroups = [
    {
      key: "attention",
      label: "Attention weights",
      tooltip: "CLS token attention to each modality in CrossModalAttention models.",
      data: explainability?.attention_weights,
    },
    {
      key: "gate",
      label: "Gate weights",
      tooltip: "Learned softmax gate for HybridFusion models.",
      data: explainability?.gate_weights,
    },
    {
      key: "learned",
      label: "Learned weights",
      tooltip: "Trained per-modality scalars for LateFusion models.",
      data: explainability?.learned_weights,
    },
  ].filter((group) => group.data && Object.keys(group.data).length > 0);

  const shapEntries = explainability?.shap_values
    ? Object.entries(explainability.shap_values)
    : [];

  const impactEntries = explainability?.directional_impact
    ? Object.entries(explainability.directional_impact)
    : [];

  const topDimGroups = [
    { key: "text", label: "Text", data: explainability?.top_text_dims },
    { key: "audio", label: "Audio", data: explainability?.top_audio_dims },
    { key: "video", label: "Video", data: explainability?.top_video_dims },
  ].filter((group) => group.data && group.data.length > 0);

  const textEvidence = explanation?.text_evidence?.sentences ?? [];
  const audioPeaks = explanation?.audio_evidence?.energy_peaks ?? [];
  const audioBands = explanation?.audio_evidence?.band_energy ?? null;
  const videoPeaks = explanation?.video_evidence?.motion_peaks ?? [];
  const audioPatches = explanation?.audio_evidence?.spectrogram_patches ?? [];
  const videoFrames = explanation?.video_evidence?.key_frames ?? [];
  const audioCueCards = audioPatches.length > 0 ? audioPatches : audioPeaks;
  const videoCueCards = videoFrames.length > 0 ? videoFrames : videoPeaks;

  const openExplainModal = () => setIsExplainOpen(true);
  const closeExplainModal = () => setIsExplainOpen(false);


  // Animate probability bar on mount
  const barRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = barRef.current;
    if (!el) return;
    el.style.width = "0%";
    const raf = requestAnimationFrame(() => {
      setTimeout(() => {
        el.style.width = `${probabilityPct}%`;
      }, 300);
    });
    return () => cancelAnimationFrame(raf);
  }, [probabilityPct]);

  const stageEntries = Object.entries(result.stage_latency ?? {}).filter(([k]) => k !== "TOTAL");
  const totalLatency = result.stage_latency?.["TOTAL"] ?? stageEntries.reduce((acc, [, v]) => acc + v, 0);

  return (
    <div className="bg-mesh min-h-screen py-12 px-4">
      <div style={{ maxWidth: 860, margin: "0 auto" }}>
        {isExplainOpen && (
          <div className="modal-overlay" onClick={closeExplainModal}>
            <div className="modal-card" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <div>
                  <div className="section-label" style={{ marginBottom: 6 }}>Explainability Narrative</div>
                  <h3 style={{ fontSize: "1.1rem", fontWeight: 700, margin: 0 }}>Why this prediction was made</h3>
                </div>
                <button className="icon-btn" onClick={closeExplainModal} aria-label="Close">
                  <IconX size={16} />
                </button>
              </div>

              <div style={{ display: "grid", gap: 14 }}>
                <p style={{ color: "var(--text-secondary)", lineHeight: 1.6, margin: 0 }}>
                  {narrativeSummary || "No narrative summary is available for this prediction."}
                </p>

                {textModelExpl?.available &&
                  textModelExpl.token_attributions &&
                  textModelExpl.token_attributions.length > 0 && (
                    <div>
                      <div className="section-label">Linguistic explainer (fine-tuned RoBERTa)</div>
                      <p style={{ color: "var(--text-muted)", fontSize: "0.82rem", margin: "0 0 8px" }}>
                        Colors show gradient salience per class (see disclaimer); stopwords are muted. Independent of
                        multimodal fusion. Model label:{" "}
                        <strong style={{ color: "var(--text-primary)" }}>{textModelExpl.label}</strong>
                        {" · "}
                        P(higher-risk wording):{" "}
                        {((textModelExpl.ptsd_probability ?? 0) * 100).toFixed(1)}%
                      </p>
                      <TextModelTokenStrip model={textModelExpl} />
                      {textModelExpl.disclaimer && (
                        <div className="evidence-note" style={{ marginTop: 10 }}>
                          {textModelExpl.disclaimer}
                        </div>
                      )}
                    </div>
                  )}
                {textModelExpl && textModelExpl.available === false && textModelExpl.message && (
                  <div className="evidence-note">{textModelExpl.message}</div>
                )}

                {textEvidence.length > 0 && (
                  <div>
                    <div className="section-label">Text cues</div>
                    <div className="evidence-grid">
                      {textEvidence.map((item, idx) => (
                        <div key={`${item.text}-${idx}`} className="evidence-card">
                          <div className="evidence-kicker">Cue {idx + 1}</div>
                          <div className="evidence-text">
                            {highlightText(item.text, item.keywords)}
                          </div>
                          {item.keywords.length > 0 ? (
                            <div className="evidence-meta">
                              Matched cues: {item.keywords.join(", ")}
                            </div>
                          ) : (
                            <div className="evidence-meta">
                              No keyword cues found for this risk tier. Showing the top-ranked sentence.
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {(audioCueCards.length > 0 || audioBands) && (
                  <div>
                    <div className="section-label">Audio cues</div>
                    <div className="evidence-grid">
                      {audioCueCards.slice(0, 3).map((item, idx) => {
                        const image = "image_b64" in item ? (item as { image_b64?: string | null }).image_b64 : null;
                        return (
                          <div key={`${item.start_s}-${idx}`} className="evidence-card evidence-card-media">
                            <div className="evidence-kicker">Patch {idx + 1}</div>
                            {image ? (
                              <img
                                className="evidence-media"
                                src={`data:image/png;base64,${image}`}
                                alt={`Spectrogram patch ${idx + 1}`}
                              />
                            ) : (
                              <div className="evidence-media evidence-placeholder">Spectrogram preview unavailable</div>
                            )}
                            <div className="evidence-meta">
                              Time {formatSeconds(item.start_s)} - {formatSeconds(item.end_s)} | RMS {item.rms.toFixed(4)}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    {audioBands && (
                      <div className="evidence-note">
                        Band energy: low {Math.round(audioBands.low * 100)}%, mid {Math.round(audioBands.mid * 100)}%, high {Math.round(audioBands.high * 100)}%
                      </div>
                    )}
                  </div>
                )}

                {videoCueCards.length > 0 && (
                  <div>
                    <div className="section-label">Video cues</div>
                    <div className="evidence-grid">
                      {videoCueCards.slice(0, 3).map((item, idx) => {
                        const image = "image_b64" in item ? (item as { image_b64?: string | null }).image_b64 : null;
                        return (
                          <div key={`${item.timestamp_s}-${idx}`} className="evidence-card evidence-card-media">
                            <div className="evidence-kicker">Frame {idx + 1}</div>
                            {image ? (
                              <img
                                className="evidence-media"
                                src={`data:image/jpeg;base64,${image}`}
                                alt={`Evidence frame ${idx + 1}`}
                              />
                            ) : (
                              <div className="evidence-media evidence-placeholder">Frame preview unavailable</div>
                            )}
                            <div className="evidence-meta">
                              Time {formatSeconds(item.timestamp_s)} | Score {item.score.toFixed(4)}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {explanation?.disclaimer && (
                  <div className="evidence-note">{explanation.disclaimer}</div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ── Header ── */}
        <div className="animate-fade-up" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 32, flexWrap: "wrap", gap: 12 }}>
          <div>
            <h2 style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: 4 }}>
              Analysis Results
            </h2>
            <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
              Multimodal risk assessment complete
            </p>
          </div>
          <button
            id="new-analysis-btn"
            onClick={onReset}
            className="btn-secondary"
            style={{ display: "flex", alignItems: "center", gap: 8, fontSize: "0.88rem", padding: "10px 20px" }}
          >
            <IconRefreshCw size={15} />
            New Analysis
          </button>
        </div>

        {/* ── Prediction Hero Card ── */}
        <div
          className={`glass-card animate-fade-up delay-100 ${isPtsd ? "result-ptsd" : "result-safe"}`}
          style={{ padding: "36px 40px", marginBottom: 20, display: "grid", gridTemplateColumns: "1fr auto", gap: 32, alignItems: "center", flexWrap: "wrap" }}
        >
          {/* Left side */}
          <div>
            <div className="section-label" style={{ marginBottom: 14 }}>Primary Signal</div>
            <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 20, flexWrap: "wrap" }}>
              <div
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 12,
                  background: isPtsd ? "rgba(239,68,68,0.1)" : "rgba(16,185,129,0.1)",
                  border: `1px solid ${isPtsd ? "rgba(239,68,68,0.3)" : "rgba(16,185,129,0.3)"}`,
                  borderRadius: 14,
                  padding: "12px 24px",
                }}
              >
                <div
                  style={{
                    width: 10,
                    height: 10,
                    borderRadius: "50%",
                    background: isPtsd ? "var(--ptsd-red)" : "var(--safe-green)",
                    animation: "pulse-glow 2s ease-in-out infinite",
                    boxShadow: `0 0 12px ${isPtsd ? "rgba(239,68,68,0.5)" : "rgba(16,185,129,0.5)"}`,
                  }}
                />
                <span
                  style={{
                    fontSize: "1.7rem",
                    fontWeight: 800,
                    color: isPtsd ? "#f87171" : "#34d399",
                    letterSpacing: "-0.01em",
                  }}
                >
                  {riskLabel}
                </span>
              </div>
              <ConfidenceBadge level={result.confidence} />
            </div>

            {/* Probability bar */}
            <div style={{ marginBottom: 6 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                <span className="section-label" style={{ marginBottom: 0 }}>Risk Probability</span>
                <span style={{ fontWeight: 700, fontSize: "0.95rem", color: isPtsd ? "#f87171" : "#34d399" }}>
                  {probabilityPct}%
                </span>
              </div>
              <div className="progress-track" style={{ height: 10 }}>
                <div
                  ref={barRef}
                  className="progress-fill"
                  style={{
                    background: isPtsd
                      ? "linear-gradient(90deg, #b91c1c, #ef4444)"
                      : "linear-gradient(90deg, #059669, #10b981)",
                    transition: "width 1.1s cubic-bezier(0.4,0,0.2,1)",
                  }}
                />
              </div>
            </div>
          </div>

          {/* Right: gauge */}
          <div style={{ minWidth: 180 }}>
            <ProbabilityGauge value={result.ptsd_probability} isElevated={isElevated} />
          </div>
        </div>

        {/* ── Metrics Row ── */}
        <div
          className="animate-fade-up delay-200"
          style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 16, marginBottom: 20 }}
        >
          {/* Uncertainty */}
          <div className="metric-card">
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
              <span className="section-label" style={{ marginBottom: 0 }}>Uncertainty</span>
              <Tooltip text="Model uncertainty reflects how confident the AI is. Lower is better. Values above 0.3 suggest limited confidence.">
                <IconInfo size={14} style={{ color: "var(--text-muted)", cursor: "help" }} />
              </Tooltip>
            </div>
            <div style={{ fontSize: "2rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: 4 }}>
              {result.uncertainty.toFixed(3)}
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              {result.uncertainty < 0.1 ? "Very low" : result.uncertainty < 0.2 ? "Low" : result.uncertainty < 0.35 ? "Moderate" : "High"} uncertainty
            </div>
          </div>

          {/* 95% CI */}
          <div className="metric-card">
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
              <span className="section-label" style={{ marginBottom: 0 }}>95% Confidence Interval</span>
              <Tooltip text="There is a 95% probability that the true risk falls within this range. A narrower range means a more precise estimate.">
                <IconInfo size={14} style={{ color: "var(--text-muted)", cursor: "help" }} />
              </Tooltip>
            </div>
            <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "var(--accent-blue)", fontVariantNumeric: "tabular-nums", marginBottom: 4 }}>
              [{(result.ci_lower * 100).toFixed(1)}%, {(result.ci_upper * 100).toFixed(1)}%]
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              Width: {((result.ci_upper - result.ci_lower) * 100).toFixed(1)}pp
            </div>
          </div>

          {/* Confidence level */}
          <div className="metric-card">
            <div style={{ marginBottom: 10 }}>
              <span className="section-label" style={{ marginBottom: 0 }}>Model Confidence</span>
            </div>
            <div style={{ marginBottom: 8 }}>
              <ConfidenceBadge level={result.confidence} />
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              Reflects overall prediction certainty
            </div>
          </div>

          {/* Total time */}
          <div className="metric-card">
            <div style={{ marginBottom: 10 }}>
              <span className="section-label" style={{ marginBottom: 0 }}>Total Processing Time</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
              <IconClock size={18} style={{ color: "var(--accent-blue)" }} />
              <span style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--text-primary)" }}>
                {formatLatency(totalLatency)}
              </span>
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              {stageEntries.length} processing stages
            </div>
          </div>
        </div>

        {auditSpectrograms.length > 0 || auditFrames.length > 0 ? (
          <div
            className="glass-card animate-fade-up delay-225"
            style={{ padding: "24px 28px", marginBottom: 20 }}
          >
            <div className="section-label" style={{ marginBottom: 8 }}>
              Full inference visuals
            </div>
            <p style={{ color: "var(--text-muted)", fontSize: "0.82rem", margin: "0 0 16px", lineHeight: 1.5 }}>
              Mel thumbnails use shorter windows than the audio embedding model (default 5s per tile via{" "}
              <code style={{ fontSize: "0.78rem" }}>VISUAL_AUDIT_CHUNK_S</code>) so you see one spectrogram per
              slice instead of only one per 30s chunk. Video tiles list every sampled frame from ingest.
            </p>

            {auditSpectrograms.length > 0 && (
              <div style={{ marginBottom: 22 }}>
                <div style={{ fontWeight: 600, fontSize: "0.88rem", marginBottom: 10, color: "var(--text-secondary)" }}>
                  Audio spectrograms ({auditSpectrograms.length})
                </div>
                <div className="visual-audit-grid">
                  {auditSpectrograms.map((ch) => (
                    <div key={ch.chunk_index} className="visual-audit-cell">
                      {ch.image_b64 ? (
                        <img
                          src={`data:image/png;base64,${ch.image_b64}`}
                          alt={`Spectrogram chunk ${ch.chunk_index}`}
                        />
                      ) : (
                        <div
                          className="visual-audit-meta"
                          style={{ height: 72, display: "flex", alignItems: "center", justifyContent: "center" }}
                        >
                          N/A
                        </div>
                      )}
                      <div className="visual-audit-meta">
                        #{ch.chunk_index} · {formatSeconds(ch.start_s)}–{formatSeconds(ch.end_s)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {auditFrames.length > 0 && (
              <div>
                <div style={{ fontWeight: 600, fontSize: "0.88rem", marginBottom: 10, color: "var(--text-secondary)" }}>
                  Video frames ({auditFrames.length})
                </div>
                <div className="visual-audit-grid">
                  {auditFrames.map((fr) => (
                    <div key={fr.frame_index} className="visual-audit-cell">
                      {fr.image_b64 ? (
                        <img
                          src={`data:image/jpeg;base64,${fr.image_b64}`}
                          alt={`Frame ${fr.frame_index}`}
                        />
                      ) : (
                        <div
                          className="visual-audit-meta"
                          style={{ height: 72, display: "flex", alignItems: "center", justifyContent: "center" }}
                        >
                          N/A
                        </div>
                      )}
                      <div className="visual-audit-meta">
                        #{fr.frame_index} · t={formatSeconds(fr.timestamp_s)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : null}

        {textModelExpl?.available && textModelExpl.token_attributions && textModelExpl.token_attributions.length > 0 && (
          <div className="glass-card animate-fade-up delay-228" style={{ padding: "24px 28px", marginBottom: 20 }}>
            <div className="section-label" style={{ marginBottom: 8 }}>
              Transcript linguistic cues
            </div>
            <p style={{ color: "var(--text-muted)", fontSize: "0.82rem", margin: "0 0 12px", lineHeight: 1.5 }}>
              Red / green show gradient salience toward each class logit (not a keyword list). Common stopwords are
              de-emphasized. Open Explainability Narrative for the full strip and disclaimer.
            </p>
            <TextModelTokenStrip model={textModelExpl} />
          </div>
        )}

        {/* ── Explainability ── */}
        <div
          className="animate-fade-up delay-250"
          style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 16, marginBottom: 20 }}
        >
          {/* Modality contribution */}
          <div
            className="glass-card"
            role="button"
            tabIndex={0}
            onClick={openExplainModal}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                openExplainModal();
              }
            }}
            style={{ padding: "24px 28px", cursor: "pointer" }}
          >
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <IconTrendingUp size={18} style={{ color: "var(--accent-blue)" }} />
                <span style={{ fontWeight: 600, color: "var(--text-primary)", fontSize: "1rem" }}>
                  Modality Contribution
                </span>
              </div>
              <Tooltip text="Normalized contribution of each modality to the final decision. Higher means stronger influence.">
                <IconInfo size={14} style={{ color: "var(--text-muted)", cursor: "help" }} />
              </Tooltip>
            </div>
            {contributionEntries.length > 0 ? (
              <div style={{ display: "grid", gap: 14 }}>
                {contributionEntries.map(([key, value]) => {
                  const pct = Math.max(0, Math.min(100, (value ?? 0) * 100));
                  const label = formatStageName(key);
                  const color =
                    key === "text"
                      ? "#4f8ef7"
                      : key === "audio"
                      ? "#f59e0b"
                      : "#10b981";
                  return (
                    <div key={key}>
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
                        <span style={{ color: "var(--text-secondary)", fontSize: "0.85rem", fontWeight: 600 }}>
                          {label}
                        </span>
                        <span style={{ color: color, fontWeight: 700, fontSize: "0.85rem" }}>
                          {formatPercent(value)}
                        </span>
                      </div>
                      <div style={{ height: 8, background: "rgba(99,134,194,0.12)", borderRadius: 99, overflow: "hidden" }}>
                        <div
                          style={{
                            width: `${pct}%`,
                            height: "100%",
                            background: `linear-gradient(90deg, ${color}, rgba(255,255,255,0.35))`,
                            borderRadius: 99,
                            transition: "width 0.6s ease",
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
                Explainability was not returned for this prediction.
              </div>
            )}
          </div>

          {/* Explainability details */}
          <div className="glass-card" style={{ padding: "24px 28px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
              <IconBrain size={18} style={{ color: "var(--accent-indigo)" }} />
              <span style={{ fontWeight: 600, color: "var(--text-primary)", fontSize: "1rem" }}>
                Explainability Details
              </span>
            </div>
            {explainability ? (
              <div style={{ display: "grid", gap: 14 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <span className="section-label" style={{ marginBottom: 0 }}>Primary Method</span>
                  <span
                    style={{
                      background: "rgba(99,102,241,0.12)",
                      border: "1px solid rgba(99,102,241,0.3)",
                      color: "#a5b4fc",
                      padding: "4px 10px",
                      borderRadius: 8,
                      fontSize: "0.72rem",
                      fontWeight: 700,
                      textTransform: "uppercase",
                      letterSpacing: "0.08em",
                    }}
                  >
                    {explainability.method}
                  </span>
                </div>

                {weightGroups.map((group) => (
                  <div key={group.key} style={{ display: "grid", gap: 8 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ color: "var(--text-secondary)", fontSize: "0.82rem", fontWeight: 600 }}>
                        {group.label}
                      </span>
                      <Tooltip text={group.tooltip}>
                        <IconInfo size={13} style={{ color: "var(--text-muted)", cursor: "help" }} />
                      </Tooltip>
                    </div>
                    <div style={{ display: "grid", gap: 6 }}>
                      {Object.entries(group.data ?? {}).map(([key, value]) => (
                        <div key={key} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: "0.82rem" }}>
                          <span style={{ color: "var(--text-secondary)" }}>{formatStageName(key)}</span>
                          <span style={{ color: "var(--accent-blue)", fontWeight: 600 }}>{formatPercent(value)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}

                {shapEntries.length > 0 && (
                  <div style={{ display: "grid", gap: 8 }}>
                    <span style={{ color: "var(--text-secondary)", fontSize: "0.82rem", fontWeight: 600 }}>
                      SHAP signal (toward higher risk)
                    </span>
                    {shapEntries.map(([key, value]) => (
                      <div key={key} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: "0.82rem" }}>
                        <span style={{ color: "var(--text-secondary)" }}>{formatStageName(key)}</span>
                        <span style={{ color: value >= 0 ? "#34d399" : "#f87171", fontWeight: 600 }}>
                          {formatSigned(value)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {impactEntries.length > 0 && (
                  <div style={{ display: "grid", gap: 8 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ color: "var(--text-secondary)", fontSize: "0.82rem", fontWeight: 600 }}>
                        Directional impact (ablation)
                      </span>
                      <Tooltip text="Signed change in risk probability when each modality is removed. Positive means it pushed the score toward higher risk.">
                        <IconInfo size={13} style={{ color: "var(--text-muted)", cursor: "help" }} />
                      </Tooltip>
                    </div>
                    {impactEntries.map(([key, value]) => (
                      <div key={key} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: "0.82rem" }}>
                        <span style={{ color: "var(--text-secondary)" }}>{formatStageName(key)}</span>
                        <span style={{ color: value >= 0 ? "#34d399" : "#f87171", fontWeight: 600 }}>
                          {formatPp(value)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {topDimGroups.length > 0 && (
                  <div style={{ display: "grid", gap: 10 }}>
                    <span style={{ color: "var(--text-secondary)", fontSize: "0.82rem", fontWeight: 600 }}>
                      Top feature dimensions
                    </span>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 10 }}>
                      {topDimGroups.map((group) => (
                        <div
                          key={group.key}
                          style={{
                            background: "rgba(7, 13, 26, 0.6)",
                            border: "1px solid var(--border-subtle)",
                            borderRadius: 10,
                            padding: "10px 12px",
                            display: "grid",
                            gap: 6,
                            fontSize: "0.75rem",
                          }}
                        >
                          <span style={{ color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em" }}>
                            {group.label}
                          </span>
                          {(group.data ?? []).slice(0, 5).map(([dim, val]) => (
                            <div key={dim} style={{ display: "flex", justifyContent: "space-between", color: "var(--text-secondary)" }}>
                              <span>Dim {dim}</span>
                              <span style={{ color: val >= 0 ? "#34d399" : "#f87171", fontWeight: 600 }}>
                                {formatSigned(val)}
                              </span>
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
                Explainability details are not available for this prediction.
              </div>
            )}
          </div>
        </div>

        {/* ── Narrative ── */}
        <div
          className="animate-fade-up delay-300"
          style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 16, marginBottom: 20 }}
        >
          <div className="glass-card" style={{ padding: "24px 28px" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
              <span style={{ fontWeight: 600, color: "var(--text-primary)", fontSize: "1rem" }}>
                Narrative Summary
              </span>
              <span className="status-pill status-good">Insight</span>
            </div>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: 16 }}>
              {narrativeSummary || "Open explainability to review the narrative summary."}
            </p>
            <button className="btn-secondary" onClick={openExplainModal}>
              View full narrative
            </button>
          </div>
        </div>

        {/* ── Model + Metadata ── */}
        <div
          className="animate-fade-up delay-300"
          style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 16, marginBottom: 20 }}
        >
          {/* Model breakdown */}
          <div className="glass-card" style={{ padding: "24px 28px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
              <IconCpu size={18} style={{ color: "var(--accent-blue)" }} />
              <span style={{ fontWeight: 600, color: "var(--text-primary)", fontSize: "1rem" }}>
                Model Breakdown
              </span>
            </div>
            {modelEntries.length > 0 ? (
              <div style={{ display: "grid", gap: 12 }}>
                {modelEntries.map(([name, data], idx) => {
                  const isPrimary = name === primaryModel;
                  return (
                    <div
                      key={name}
                      style={{
                        display: "grid",
                        gridTemplateColumns: "1fr auto",
                        gap: 12,
                        paddingBottom: idx === modelEntries.length - 1 ? 0 : 12,
                        borderBottom: idx === modelEntries.length - 1 ? "none" : "1px solid rgba(99,134,194,0.08)",
                      }}
                    >
                      <div>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                          <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>
                            {formatStageName(name)}
                          </span>
                          {isPrimary && (
                            <span
                              style={{
                                background: "rgba(16,185,129,0.12)",
                                border: "1px solid rgba(16,185,129,0.3)",
                                color: "#34d399",
                                padding: "2px 8px",
                                borderRadius: 8,
                                fontSize: "0.65rem",
                                fontWeight: 700,
                                textTransform: "uppercase",
                                letterSpacing: "0.08em",
                              }}
                            >
                              Primary
                            </span>
                          )}
                        </div>
                        <div style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>
                          Risk tier: {data.prob >= 0.5 ? "Elevated" : "Lower"}
                        </div>
                        {data.mc_prob !== undefined && data.mc_std !== undefined && (
                          <div style={{ color: "var(--text-secondary)", fontSize: "0.75rem", marginTop: 6 }}>
                            MC mean {formatPercent(data.mc_prob)} | std {data.mc_std.toFixed(3)}
                          </div>
                        )}
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <div style={{ color: "var(--accent-blue)", fontWeight: 700, fontSize: "1.05rem" }}>
                          {formatPercent(data.prob)}
                        </div>
                        <div style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>Risk prob</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
                Model breakdown is not available.
              </div>
            )}
          </div>

          {/* Metadata */}
          <div className="glass-card" style={{ padding: "24px 28px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
              <IconActivity size={18} style={{ color: "var(--accent-indigo)" }} />
              <span style={{ fontWeight: 600, color: "var(--text-primary)", fontSize: "1rem" }}>
                Inference Metadata
              </span>
            </div>
            {result.metadata ? (
              <div style={{ display: "grid", gap: 10, fontSize: "0.82rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span style={{ color: "var(--text-secondary)" }}>Duration</span>
                  <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>{result.metadata.duration_s.toFixed(1)} s</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span style={{ color: "var(--text-secondary)" }}>Frames analyzed</span>
                  <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>{result.metadata.n_frames}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span style={{ color: "var(--text-secondary)" }}>Audio chunks</span>
                  <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>{result.metadata.n_audio_chunks}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span style={{ color: "var(--text-secondary)" }}>Primary model</span>
                  <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>{formatStageName(result.metadata.primary_model)}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span style={{ color: "var(--text-secondary)" }}>Device</span>
                  <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>{result.metadata.device}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span style={{ color: "var(--text-secondary)" }}>Cache hit</span>
                  <span style={{ color: result.metadata.cache_hit ? "#34d399" : "#fbbf24", fontWeight: 700 }}>
                    {result.metadata.cache_hit ? "Yes" : "No"}
                  </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span style={{ color: "var(--text-secondary)" }}>Video path</span>
                  <span style={{ color: "var(--text-muted)", fontWeight: 500, maxWidth: 220, textAlign: "right", overflowWrap: "anywhere" }}>
                    {result.metadata.video}
                  </span>
                </div>
              </div>
            ) : (
              <div style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
                Metadata not available.
              </div>
            )}
          </div>
        </div>

        {/* ── Stage Latencies ── */}
        {stageEntries.length > 0 && (
          <div className="glass-card animate-fade-up delay-400" style={{ padding: "28px 32px", marginBottom: 20 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20 }}>
              <IconActivity size={18} style={{ color: "var(--accent-indigo)" }} />
              <span style={{ fontWeight: 600, color: "var(--text-primary)", fontSize: "1rem" }}>
                Processing Stage Timings
              </span>
            </div>
            <div>
              {stageEntries.map(([key, val]) => {
                const pct = totalLatency > 0 ? (val / totalLatency) * 100 : 0;
                return (
                  <div key={key} className="latency-row">
                    <span style={{ color: "var(--text-secondary)", fontWeight: 500 }}>
                      {formatStageName(key)}
                    </span>
                    <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                      {/* Mini bar */}
                      <div
                        style={{
                          width: 80,
                          height: 4,
                          background: "rgba(99,134,194,0.1)",
                          borderRadius: 99,
                          overflow: "hidden",
                        }}
                      >
                        <div
                          style={{
                            height: "100%",
                            width: `${pct}%`,
                            background: "linear-gradient(90deg, #4f8ef7, #6366f1)",
                            borderRadius: 99,
                          }}
                        />
                      </div>
                      <span style={{ color: "var(--accent-blue)", fontWeight: 600, fontVariantNumeric: "tabular-nums", minWidth: 60, textAlign: "right" }}>
                        {formatLatency(val)}
                      </span>
                    </div>
                  </div>
                );
              })}
              <div
                className="latency-row"
                style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: 12, marginTop: 4, fontWeight: 700 }}
              >
                <span style={{ color: "var(--text-primary)" }}>Total</span>
                <span style={{ color: "var(--text-primary)", fontVariantNumeric: "tabular-nums" }}>
                  {formatLatency(totalLatency)}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* ── Disclaimer ── */}
        <div className="animate-fade-up delay-500">
          <div className="info-banner">
            <span style={{ color: "var(--accent-indigo)", flexShrink: 0 }}>
              <IconShield size={16} />
            </span>
            <span>
              <strong>Research Use Only.</strong> This system is not intended to replace clinical diagnosis or professional mental health assessment.
              Results should be interpreted by a qualified clinician. Consult a licensed mental health professional for any medical decisions.
            </span>
          </div>
        </div>

        {/* ── Reset button ── */}
        <div className="animate-fade-up delay-500" style={{ textAlign: "center", marginTop: 32 }}>
          <button
            id="analyze-another-btn"
            onClick={onReset}
            className="btn-glow"
            style={{ display: "inline-flex", alignItems: "center", gap: 10 }}
          >
            <IconRefreshCw size={17} />
            Analyze Another Video
          </button>
        </div>
      </div>
    </div>
  );
}
