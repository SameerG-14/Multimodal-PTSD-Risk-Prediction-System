"use client";

import React from "react";
import { IconAlertTriangle, IconRefreshCw, IconX } from "./Icons";

interface ErrorStateProps {
  message: string;
  onRetry: () => void;
  onDismiss?: () => void;
}

export default function ErrorState({ message, onRetry, onDismiss }: ErrorStateProps) {
  return (
    <div className="min-h-screen bg-mesh flex flex-col items-center justify-center px-4 py-16">
      <div
        className="animate-scale-in"
        style={{ width: "100%", maxWidth: 500, textAlign: "center" }}
      >
        {/* Error Icon */}
        <div
          style={{
            width: 88,
            height: 88,
            borderRadius: "50%",
            background: "rgba(239,68,68,0.1)",
            border: "1px solid rgba(239,68,68,0.25)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            margin: "0 auto 28px",
            animation: "pulse-glow 2.5s ease-in-out infinite",
          }}
        >
          <IconAlertTriangle size={36} style={{ color: "#f87171" }} />
        </div>

        <h2
          style={{
            fontSize: "1.5rem",
            fontWeight: 700,
            color: "var(--text-primary)",
            marginBottom: 12,
          }}
        >
          Analysis failed
        </h2>

        {/* Error box */}
        <div
          style={{
            background: "rgba(239,68,68,0.07)",
            border: "1px solid rgba(239,68,68,0.2)",
            borderRadius: 14,
            padding: "18px 24px",
            marginBottom: 32,
            textAlign: "left",
            position: "relative",
          }}
          data-error={message}
        >
          <div style={{ display: "grid", gap: 10 }}>
            <p
              style={{
                color: "#fca5a5",
                fontSize: "0.95rem",
                fontWeight: 600,
                margin: 0,
              }}
            >
              Invalid video.
            </p>
            <div style={{ color: "#fca5a5", fontSize: "0.88rem", lineHeight: 1.6 }}>
              Please ensure:
            </div>
            <div style={{ display: "grid", gap: 6, color: "#fca5a5", fontSize: "0.88rem", lineHeight: 1.6 }}>
              <div>• Face is clearly visible and centered.</div>
              <div>• Audio contains clear speech with minimal silence.</div>
              <div>• Video is well-lit with minimal blur.</div>
              <div>• File format is MP4, MOV, or AVI.</div>
              <div>• File is not corrupted and plays locally.</div>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div style={{ display: "flex", gap: 12, justifyContent: "center" }}>
          {onDismiss && (
            <button
              id="error-back-btn"
              onClick={onDismiss}
              className="btn-secondary"
            >
              Go Back
            </button>
          )}
          <button
            id="retry-btn"
            onClick={onRetry}
            className="btn-glow"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 10,
            }}
          >
            <IconRefreshCw size={16} />
            Try Again
          </button>
        </div>
      </div>
    </div>
  );
}
