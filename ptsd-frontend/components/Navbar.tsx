"use client";

import React from "react";
import { IconBrain, IconShield } from "./Icons";

type AppStep = "landing" | "upload" | "processing" | "results" | "error";

const STEPS: { id: AppStep; label: string }[] = [
  { id: "upload", label: "Upload" },
  { id: "processing", label: "Analysis" },
  { id: "results", label: "Results" },
];

interface NavbarProps {
  currentStep: AppStep;
  onLogoClick: () => void;
}

export default function Navbar({ currentStep, onLogoClick }: NavbarProps) {
  const stepIndex = STEPS.findIndex((s) => s.id === currentStep);

  return (
    <header
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        zIndex: 50,
        background: "rgba(7, 13, 26, 0.85)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        borderBottom: "1px solid var(--border-subtle)",
      }}
    >
      <div
        style={{
          maxWidth: 1100,
          margin: "0 auto",
          padding: "0 24px",
          height: 60,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 16,
        }}
      >
        {/* Logo */}
        <button
          id="nav-logo-btn"
          onClick={onLogoClick}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            background: "none",
            border: "none",
            cursor: "pointer",
            padding: 0,
            flexShrink: 0,
          }}
        >
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 9,
              background: "linear-gradient(135deg, #4f8ef7 0%, #6366f1 100%)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <IconBrain size={17} style={{ color: "#fff" }} />
          </div>
          <span
            style={{
              fontWeight: 700,
              fontSize: "0.95rem",
              color: "var(--text-primary)",
              letterSpacing: "-0.01em",
            }}
          >
            PTSD<span style={{ color: "var(--accent-blue)" }}>Detect</span>
          </span>
        </button>

        {/* Step breadcrumb — only show on active flow */}
        {currentStep !== "landing" && currentStep !== "error" && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              fontSize: "0.78rem",
              fontWeight: 600,
            }}
          >
            {STEPS.map((step, idx) => {
              const isDone = idx < stepIndex;
              const isActive = idx === stepIndex;
              return (
                <React.Fragment key={step.id}>
                  {idx > 0 && (
                    <svg
                      width="12"
                      height="12"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.5"
                      style={{ color: "var(--text-muted)" }}
                    >
                      <polyline points="9 18 15 12 9 6" />
                    </svg>
                  )}
                  <span
                    style={{
                      color: isActive
                        ? "var(--accent-blue)"
                        : isDone
                        ? "var(--safe-green)"
                        : "var(--text-muted)",
                      display: "flex",
                      alignItems: "center",
                      gap: 4,
                    }}
                  >
                    {isDone && (
                      <svg
                        width="10"
                        height="10"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="3"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                    )}
                    {step.label}
                  </span>
                </React.Fragment>
              );
            })}
          </div>
        )}

        {/* Right: research badge */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            background: "rgba(99,102,241,0.08)",
            border: "1px solid rgba(99,102,241,0.18)",
            borderRadius: 8,
            padding: "4px 12px",
            fontSize: "0.72rem",
            fontWeight: 600,
            color: "var(--text-muted)",
            letterSpacing: "0.04em",
            textTransform: "uppercase",
            flexShrink: 0,
          }}
        >
          <IconShield size={12} />
          Research Only
        </div>
      </div>
    </header>
  );
}
