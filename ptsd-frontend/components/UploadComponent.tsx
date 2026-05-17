"use client";

import React, { useCallback, useRef, useState } from "react";
import { UploadedFile } from "@/types/prediction";
import { formatFileSize, isValidVideoType } from "@/lib/api";
import { IconUpload, IconVideo, IconX, IconAlertTriangle, IconCheckCircle, IconMic } from "./Icons";

interface UploadComponentProps {
  onUpload: (file: UploadedFile) => void;
  uploadProgress: number;
  isUploading: boolean;
}

export default function UploadComponent({
  onUpload,
  uploadProgress,
  isUploading,
}: UploadComponentProps) {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<UploadedFile | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    setFileError(null);
    if (!isValidVideoType(file)) {
      setFileError("Invalid file type. Please upload an MP4, MOV, or AVI video.");
      setSelectedFile(null);
      return;
    }
    if (file.size > 500 * 1024 * 1024) {
      setFileError("File is too large. Maximum size is 500 MB.");
      setSelectedFile(null);
      return;
    }
    setSelectedFile({ file, name: file.name, sizeBytes: file.size });
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => setDragOver(false), []);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleSubmit = () => {
    if (selectedFile && !isUploading) {
      onUpload(selectedFile);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setFileError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div
      className="min-h-screen bg-mesh flex flex-col items-center justify-center px-4 py-16"
      style={{ position: "relative" }}
    >
      <div style={{ width: "100%", maxWidth: 600 }}>
        {/* Header */}
        <div className="animate-fade-up" style={{ marginBottom: 36, textAlign: "center" }}>
          <h2
            style={{
              fontSize: "1.8rem",
              fontWeight: 700,
              marginBottom: 10,
              color: "var(--text-primary)",
            }}
          >
            Upload Interview Video
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem" }}>
            Select a video file to begin risk analysis. Make sure the face is clearly visible and audio is clear.
          </p>
        </div>

        {/* Drop Zone */}
        <div
          className={`animate-fade-up delay-100 drop-zone ${dragOver ? "dragover" : ""} ${selectedFile ? "has-file" : ""}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => !selectedFile && !isUploading && inputRef.current?.click()}
          style={{
            cursor: selectedFile || isUploading ? "default" : "pointer",
            padding: "60px 40px",
            textAlign: "center",
            transition: "all 0.25s ease",
          }}
        >
          <input
            ref={inputRef}
            id="video-input"
            type="file"
            accept="video/mp4,video/quicktime,video/avi,video/x-msvideo,.mp4,.mov,.avi,.wmv,.webm"
            style={{ display: "none" }}
            onChange={handleInputChange}
            disabled={isUploading}
          />

          {!selectedFile ? (
            <div className="animate-float">
              <div
                style={{
                  width: 80,
                  height: 80,
                  borderRadius: "50%",
                  background: "rgba(79,142,247,0.1)",
                  border: "1px solid rgba(79,142,247,0.2)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  margin: "0 auto 24px",
                }}
              >
                <IconUpload size={32} style={{ color: "var(--accent-blue)" }} />
              </div>
              <p style={{ color: "var(--text-primary)", fontWeight: 600, fontSize: "1.05rem", marginBottom: 8 }}>
                Drop your video here
              </p>
              <p style={{ color: "var(--text-muted)", fontSize: "0.88rem", marginBottom: 12 }}>
                or click to browse
              </p>
              <p style={{ color: "var(--text-muted)", fontSize: "0.82rem", marginBottom: 16 }}>
                Tip: Use a front-facing, well-lit interview with clear speech.
              </p>
              <div style={{ display: "flex", gap: 8, justifyContent: "center", flexWrap: "wrap" }}>
                {["MP4", "MOV", "AVI"].map((fmt) => (
                  <span
                    key={fmt}
                    style={{
                      background: "rgba(99,134,194,0.08)",
                      border: "1px solid var(--border-subtle)",
                      borderRadius: 6,
                      padding: "3px 10px",
                      fontSize: "0.75rem",
                      fontWeight: 600,
                      color: "var(--text-muted)",
                      letterSpacing: "0.05em",
                    }}
                  >
                    {fmt}
                  </span>
                ))}
              </div>
            </div>
          ) : (
            <div className="animate-scale-in">
              {/* File info card */}
              <div
                style={{
                  background: "rgba(99,102,241,0.08)",
                  border: "1px solid rgba(99,102,241,0.2)",
                  borderRadius: 14,
                  padding: "20px 24px",
                  display: "flex",
                  alignItems: "center",
                  gap: 16,
                  textAlign: "left",
                  position: "relative",
                }}
              >
                <div
                  style={{
                    width: 48,
                    height: 48,
                    borderRadius: 12,
                    background: "rgba(99,102,241,0.15)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                  }}
                >
                  <IconVideo size={22} />
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      fontWeight: 600,
                      color: "var(--text-primary)",
                      fontSize: "0.95rem",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {selectedFile.name}
                  </div>
                  <div style={{ color: "var(--text-muted)", fontSize: "0.82rem", marginTop: 4 }}>
                    {formatFileSize(selectedFile.sizeBytes)}
                  </div>
                </div>
                {!isUploading && (
                  <button
                    id="clear-file-btn"
                    onClick={(e) => { e.stopPropagation(); clearFile(); }}
                    style={{
                      background: "rgba(239,68,68,0.1)",
                      border: "1px solid rgba(239,68,68,0.2)",
                      borderRadius: 8,
                      color: "#f87171",
                      cursor: "pointer",
                      padding: 6,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      flexShrink: 0,
                      transition: "background 0.2s ease",
                    }}
                  >
                    <IconX size={16} />
                  </button>
                )}
              </div>

              {/* Upload progress */}
              {isUploading && (
                <div className="animate-fade-in" style={{ marginTop: 20 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8, fontSize: "0.82rem" }}>
                    <span style={{ color: "var(--text-secondary)" }}>Uploading…</span>
                    <span style={{ color: "var(--accent-blue)", fontWeight: 600 }}>{uploadProgress}%</span>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill"
                      style={{
                        width: `${uploadProgress}%`,
                        background: "linear-gradient(90deg, #4f8ef7, #6366f1)",
                      }}
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div
          className="glass-card animate-fade-up delay-150"
          style={{ marginTop: 18, padding: "16px 18px", textAlign: "left" }}
        >
          <div
            style={{
              fontSize: "0.78rem",
              fontWeight: 700,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "var(--text-muted)",
              marginBottom: 10,
            }}
          >
            Quality Checklist
          </div>
          <div style={{ display: "grid", gap: 10 }}>
            {[
              {
                icon: <IconVideo size={16} style={{ color: "var(--accent-blue)" }} />,
                text: "Face clearly visible and centered for most of the video.",
              },
              {
                icon: <IconMic size={16} style={{ color: "#f59e0b" }} />,
                text: "Clear speech with minimal background noise.",
              },
              {
                icon: <IconCheckCircle size={16} style={{ color: "#34d399" }} />,
                text: "Well-lit, steady camera with minimal blur.",
              },
            ].map((item, idx) => (
              <div key={idx} style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                <span
                  style={{
                    width: 26,
                    height: 26,
                    borderRadius: 8,
                    background: "rgba(99,134,194,0.08)",
                    border: "1px solid var(--border-subtle)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                    marginTop: 1,
                  }}
                >
                  {item.icon}
                </span>
                <span style={{ color: "var(--text-secondary)", fontSize: "0.86rem", lineHeight: 1.4 }}>
                  {item.text}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* File error */}
        {fileError && (
          <div
            className="animate-fade-in"
            style={{
              marginTop: 16,
              display: "flex",
              alignItems: "center",
              gap: 10,
              background: "rgba(239,68,68,0.08)",
              border: "1px solid rgba(239,68,68,0.2)",
              borderRadius: 10,
              padding: "12px 16px",
              color: "#f87171",
              fontSize: "0.88rem",
            }}
          >
            <IconAlertTriangle size={16} />
            {fileError}
          </div>
        )}

        {/* Actions */}
        <div className="animate-fade-up delay-200" style={{ marginTop: 28, display: "flex", gap: 12 }}>
          {selectedFile && !isUploading && (
            <button
              id="change-file-btn"
              onClick={() => inputRef.current?.click()}
              className="btn-secondary"
              style={{ flex: 1 }}
            >
              Change File
            </button>
          )}
          <button
            id="analyze-btn"
            onClick={handleSubmit}
            disabled={!selectedFile || isUploading}
            className="btn-glow"
            style={{
              flex: selectedFile && !isUploading ? 2 : 1,
              width: !selectedFile || isUploading ? "100%" : undefined,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 10,
              fontSize: "1rem",
            }}
          >
            {isUploading ? (
              <>
                <span className="spinner" style={{ width: 18, height: 18 }} />
                Uploading…
              </>
            ) : (
              <>
                <IconUpload size={18} />
                Analyze Video
              </>
            )}
          </button>
        </div>

        {/* Privacy note */}
        <div
          className="animate-fade-up delay-300"
          style={{ marginTop: 24, textAlign: "center", color: "var(--text-muted)", fontSize: "0.78rem" }}
        >
          Videos are processed locally and not stored on external servers
        </div>
      </div>
    </div>
  );
}
