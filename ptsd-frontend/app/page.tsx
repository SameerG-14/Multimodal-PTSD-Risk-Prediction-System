"use client";

import React, { useState, useCallback } from "react";
import Navbar from "@/components/Navbar";
import LandingHero from "@/components/LandingHero";
import UploadComponent from "@/components/UploadComponent";
import ProcessingLoader from "@/components/ProcessingLoader";
import ResultDashboard from "@/components/ResultDashboard";
import ErrorState from "@/components/ErrorState";
import { predictVideo, ApiError } from "@/lib/api";
import { PredictionResponse, UploadedFile } from "@/types/prediction";

type AppStep = "landing" | "upload" | "processing" | "results" | "error";

export default function Home() {
  const [step, setStep] = useState<AppStep>("landing");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");

  const handleGetStarted = useCallback(() => {
    setStep("upload");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const handleUpload = useCallback(async (uploaded: UploadedFile) => {
    setIsUploading(true);
    setUploadProgress(0);

    try {
      // Phase 1: upload (0-100%)
      const data = await predictVideo(uploaded.file, (pct) => {
        setUploadProgress(pct);
      });

      // Phase 2: transition to processing state
      setIsUploading(false);
      setStep("processing");

      // Give the server time to respond after upload completes
      // (The XHR already awaited the full response, so data is ready)
      // Small UX delay so the processing screen is visible
      await new Promise((res) => setTimeout(res, 2000));

      setResult(data);
      setStep("results");
    } catch (err) {
      setIsUploading(false);
      let msg = "An unexpected error occurred. Please try again.";
      if (err instanceof ApiError) {
        msg = err.message;
      } else if (err instanceof Error) {
        msg = err.message;
      }
      setErrorMessage(msg);
      setStep("error");
    }
  }, []);

  const handleReset = useCallback(() => {
    setStep("upload");
    setResult(null);
    setErrorMessage("");
    setUploadProgress(0);
    setIsUploading(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const handleRetry = useCallback(() => {
    setStep("upload");
    setErrorMessage("");
    setUploadProgress(0);
    setIsUploading(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const handleLogoClick = useCallback(() => {
    setStep("landing");
    setResult(null);
    setErrorMessage("");
    setUploadProgress(0);
    setIsUploading(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  return (
    <>
      <Navbar
        currentStep={step}
        onLogoClick={handleLogoClick}
      />

      {/* Offset for fixed navbar */}
      <main style={{ paddingTop: step === "landing" ? 0 : 60 }}>
        {step === "landing" && (
          <div
            key="landing"
            style={{ animation: "fadeIn 0.4s ease" }}
          >
            <LandingHero onGetStarted={handleGetStarted} />
          </div>
        )}

        {step === "upload" && (
          <div
            key="upload"
            style={{ animation: "fadeUp 0.4s ease" }}
          >
            <UploadComponent
              onUpload={handleUpload}
              uploadProgress={uploadProgress}
              isUploading={isUploading}
            />
          </div>
        )}

        {step === "processing" && (
          <div
            key="processing"
            style={{ animation: "fadeIn 0.4s ease" }}
          >
            <ProcessingLoader />
          </div>
        )}

        {step === "results" && result && (
          <div
            key="results"
            style={{ animation: "fadeUp 0.5s ease" }}
          >
            <ResultDashboard result={result} onReset={handleReset} />
          </div>
        )}

        {step === "error" && (
          <div
            key="error"
            style={{ animation: "scaleIn 0.4s ease" }}
          >
            <ErrorState
              message={errorMessage}
              onRetry={handleRetry}
              onDismiss={handleRetry}
            />
          </div>
        )}
      </main>
    </>
  );
}
