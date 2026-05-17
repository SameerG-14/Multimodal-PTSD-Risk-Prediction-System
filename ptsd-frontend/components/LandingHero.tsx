"use client";

import React from "react";
import { IconBrain, IconShield, IconZap } from "./Icons";

interface LandingHeroProps {
  onGetStarted: () => void;
}

export default function LandingHero({ onGetStarted }: LandingHeroProps) {
  return (
    <div className="min-h-screen bg-mesh flex flex-col items-center justify-center px-4 py-16 relative overflow-hidden">
      {/* Decorative grid lines */}
      <div
        aria-hidden
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage:
            "linear-gradient(rgba(79,142,247,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(79,142,247,0.04) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
          pointerEvents: "none",
        }}
      />

      {/* Top badge */}
      <div
        className="animate-fade-up"
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 10,
          background: "rgba(79,142,247,0.1)",
          border: "1px solid rgba(79,142,247,0.25)",
          borderRadius: 99,
          padding: "8px 20px",
          marginBottom: 40,
          fontSize: "0.82rem",
          fontWeight: 600,
          color: "var(--accent-blue)",
          letterSpacing: "0.04em",
          textTransform: "uppercase",
        }}
      >
        <span
          style={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            background: "var(--accent-blue)",
            animation: "pulse-glow 1.5s ease-in-out infinite",
          }}
        />
        AI-Powered Mental Health Assessment
      </div>

      {/* Main title */}
      <h1
        className="animate-fade-up delay-100"
        style={{
          fontSize: "clamp(2.2rem, 6vw, 4rem)",
          fontWeight: 800,
          lineHeight: 1.1,
          textAlign: "center",
          maxWidth: 700,
          marginBottom: 24,
          letterSpacing: "-0.02em",
        }}
      >
        <span className="gradient-text">PTSD Risk</span>
        <br />
        <span style={{ color: "var(--text-primary)" }}>Detection System</span>
      </h1>

      {/* Subtitle */}
      <p
        className="animate-fade-up delay-200"
        style={{
          color: "var(--text-secondary)",
          fontSize: "1.1rem",
          lineHeight: 1.7,
          textAlign: "center",
          maxWidth: 540,
          marginBottom: 48,
        }}
      >
        Upload a clinical interview video and our multimodal AI model analyzes
        audio, visual, and linguistic cues to assess PTSD risk indicators in seconds.
      </p>

      {/* Feature chips */}
      <div
        className="animate-fade-up delay-300"
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 12,
          justifyContent: "center",
          marginBottom: 52,
        }}
      >
        {[
          { icon: <IconBrain size={15} />, label: "Multimodal AI" },
          { icon: <IconShield size={15} />, label: "95% Confidence Interval" },
          { icon: <IconZap size={15} />, label: "Real-Time Analysis" },
        ].map((chip) => (
          <div
            key={chip.label}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              background: "rgba(15,24,41,0.8)",
              border: "1px solid var(--border-subtle)",
              borderRadius: 10,
              padding: "8px 16px",
              fontSize: "0.85rem",
              color: "var(--text-secondary)",
              fontWeight: 500,
            }}
          >
            <span style={{ color: "var(--accent-blue)" }}>{chip.icon}</span>
            {chip.label}
          </div>
        ))}
      </div>

      {/* CTA Button */}
      <div className="animate-fade-up delay-400" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
        <button
          id="get-started-btn"
          onClick={onGetStarted}
          className="btn-glow"
          style={{ fontSize: "1.05rem", padding: "16px 44px" }}
        >
          Upload Interview Video
        </button>
        <span style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>
          Supports MP4, MOV, AVI · Max file size 500 MB
        </span>
      </div>

      {/* Floating cards */}
      <div
        className="animate-fade-up delay-500"
        style={{
          marginTop: 72,
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          gap: 16,
          width: "100%",
          maxWidth: 720,
        }}
      >
        {[
          { value: "Multimodal", label: "Audio + Video + Text fusion" },
          { value: "PyTorch", label: "Early fusion deep learning" },
          { value: "Research", label: "Not a clinical diagnosis" },
        ].map((stat, i) => (
          <div
            key={stat.label}
            className="glass-card"
            style={{
              padding: "20px 24px",
              textAlign: "center",
              animationDelay: `${0.5 + i * 0.1}s`,
            }}
          >
            <div
              style={{
                fontSize: "1.05rem",
                fontWeight: 700,
                color: "var(--accent-blue)",
                marginBottom: 6,
              }}
            >
              {stat.value}
            </div>
            <div style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
              {stat.label}
            </div>
          </div>
        ))}
      </div>

      {/* Research disclaimer */}
      <div
        className="animate-fade-up delay-500"
        style={{ marginTop: 40, maxWidth: 560 }}
      >
        <div className="info-banner">
          <span style={{ color: "var(--accent-indigo)", flexShrink: 0 }}>
            <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>
            </svg>
          </span>
          This system is intended for <strong>&nbsp;research purposes only&nbsp;</strong> and
          does not constitute a clinical diagnosis. Always consult a licensed mental
          health professional.
        </div>
      </div>
    </div>
  );
}
