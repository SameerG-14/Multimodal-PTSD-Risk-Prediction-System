"use client";

import React, { useEffect, useState } from "react";
import { IconMic, IconFilm, IconCpu, IconActivity } from "./Icons";

const STAGES = [
  {
    id: "extract",
    icon: <IconMic size={18} />,
    label: "Extracting audio track",
    sublabel: "Separating audio from video stream",
    duration: 3000,
  },
  {
    id: "frames",
    icon: <IconFilm size={18} />,
    label: "Processing video frames",
    sublabel: "Sampling key frames for visual analysis",
    duration: 4000,
  },
  {
    id: "model",
    icon: <IconCpu size={18} />,
    label: "Running AI model",
    sublabel: "Multimodal early-fusion inference",
    duration: 5000,
  },
  {
    id: "post",
    icon: <IconActivity size={18} />,
    label: "Post-processing results",
    sublabel: "Computing confidence intervals & statistics",
    duration: 2000,
  },
];

export default function ProcessingLoader() {
  const [activeStage, setActiveStage] = useState(0);
  const [completedStages, setCompletedStages] = useState<Set<number>>(new Set());
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  // Cycle through stages
  useEffect(() => {
    let stageIdx = 0;
    const cycleStages = () => {
      if (stageIdx < STAGES.length) {
        setActiveStage(stageIdx);
        const timer = setTimeout(() => {
          setCompletedStages((prev) => new Set([...prev, stageIdx]));
          stageIdx++;
          if (stageIdx < STAGES.length) {
            setActiveStage(stageIdx);
            cycleStages();
          }
        }, STAGES[stageIdx].duration);
        return () => clearTimeout(timer);
      }
    };
    cycleStages();
  }, []);

  // Elapsed timer
  useEffect(() => {
    const interval = setInterval(() => {
      setElapsedSeconds((s) => s + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const formatElapsed = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
  };

  return (
    <div
      className="min-h-screen bg-mesh flex flex-col items-center justify-center px-4 py-16"
    >
      <div style={{ width: "100%", maxWidth: 520, textAlign: "center" }}>
        {/* Animated orb */}
        <div
          className="animate-fade-in"
          style={{ display: "flex", justifyContent: "center", marginBottom: 40 }}
        >
          <div
            style={{
              position: "relative",
              width: 100,
              height: 100,
            }}
          >
            {/* Outer ring */}
            <div
              style={{
                position: "absolute",
                inset: 0,
                borderRadius: "50%",
                border: "2px solid rgba(79,142,247,0.2)",
                animation: "spin 3s linear infinite",
              }}
            />
            {/* Middle ring */}
            <div
              style={{
                position: "absolute",
                inset: 12,
                borderRadius: "50%",
                border: "2px solid rgba(99,102,241,0.3)",
                animation: "spin 2s linear infinite reverse",
              }}
            />
            {/* Core */}
            <div
              style={{
                position: "absolute",
                inset: 24,
                borderRadius: "50%",
                background: "radial-gradient(circle, rgba(79,142,247,0.3) 0%, rgba(99,102,241,0.15) 100%)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                animation: "pulse-glow 2s ease-in-out infinite",
              }}
            >
              <IconCpu size={22} style={{ color: "var(--accent-blue)" }} />
            </div>
          </div>
        </div>

        <h2
          className="animate-fade-up"
          style={{
            fontSize: "1.6rem",
            fontWeight: 700,
            marginBottom: 10,
            color: "var(--text-primary)",
          }}
        >
          Analyzing Video
        </h2>
        <p
          className="animate-fade-up delay-100"
          style={{ color: "var(--text-secondary)", fontSize: "0.95rem", marginBottom: 8 }}
        >
          The AI model is processing your interview footage
        </p>
        <div
          className="animate-fade-up delay-200"
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            color: "var(--text-muted)",
            fontSize: "0.82rem",
            marginBottom: 48,
          }}
        >
          <span className="spinner" style={{ width: 12, height: 12 }} />
          Elapsed: {formatElapsed(elapsedSeconds)}
        </div>

        {/* Stage list */}
        <div className="glass-card animate-fade-up delay-300" style={{ padding: "8px 0", textAlign: "left" }}>
          {STAGES.map((stage, idx) => {
            const isDone = completedStages.has(idx);
            const isActive = activeStage === idx && !isDone;
            const isPending = idx > activeStage;

            return (
              <div
                key={stage.id}
                className="stage-item"
                style={{
                  padding: "14px 24px",
                  opacity: isPending ? 0.4 : 1,
                  transition: "opacity 0.3s ease",
                }}
              >
                {/* Status indicator */}
                <div
                  style={{
                    width: 36,
                    height: 36,
                    borderRadius: "50%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                    background: isDone
                      ? "rgba(16,185,129,0.15)"
                      : isActive
                      ? "rgba(79,142,247,0.15)"
                      : "rgba(99,134,194,0.08)",
                    border: `1px solid ${
                      isDone
                        ? "rgba(16,185,129,0.3)"
                        : isActive
                        ? "rgba(79,142,247,0.3)"
                        : "var(--border-subtle)"
                    }`,
                    color: isDone
                      ? "#34d399"
                      : isActive
                      ? "var(--accent-blue)"
                      : "var(--text-muted)",
                    transition: "all 0.3s ease",
                  }}
                >
                  {isDone ? (
                    <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="20 6 9 17 4 12"/>
                    </svg>
                  ) : isActive ? (
                    <span className="spinner" style={{ width: 16, height: 16, borderTopColor: "var(--accent-blue)" }} />
                  ) : (
                    stage.icon
                  )}
                </div>

                <div style={{ flex: 1 }}>
                  <div
                    style={{
                      fontWeight: 600,
                      fontSize: "0.92rem",
                      color: isDone
                        ? "#34d399"
                        : isActive
                        ? "var(--text-primary)"
                        : "var(--text-muted)",
                      marginBottom: 3,
                    }}
                  >
                    {stage.label}
                  </div>
                  <div style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
                    {stage.sublabel}
                  </div>
                </div>

                {isDone && (
                  <span
                    style={{
                      fontSize: "0.75rem",
                      color: "#34d399",
                      fontWeight: 600,
                      background: "rgba(16,185,129,0.1)",
                      border: "1px solid rgba(16,185,129,0.2)",
                      borderRadius: 6,
                      padding: "2px 8px",
                    }}
                  >
                    Done
                  </span>
                )}
                {isActive && (
                  <span
                    style={{
                      fontSize: "0.75rem",
                      color: "var(--accent-blue)",
                      fontWeight: 600,
                      background: "rgba(79,142,247,0.1)",
                      border: "1px solid rgba(79,142,247,0.2)",
                      borderRadius: 6,
                      padding: "2px 8px",
                    }}
                  >
                    Running
                  </span>
                )}
              </div>
            );
          })}
        </div>

        <p
          className="animate-fade-up delay-400"
          style={{
            marginTop: 24,
            color: "var(--text-muted)",
            fontSize: "0.78rem",
            textAlign: "center",
          }}
        >
          Large videos may take several minutes to process
        </p>
      </div>
    </div>
  );
}
