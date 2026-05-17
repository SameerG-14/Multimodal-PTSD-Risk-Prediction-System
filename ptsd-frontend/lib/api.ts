import { PredictionResponse } from "@/types/prediction";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public code?: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export interface PredictVideoOptions {
  includeTranscript?: boolean;
  includeExplanation?: boolean;
  /** Every mel chunk spectrogram + every sampled frame (large JSON). Default true. */
  includeFullVisualAudit?: boolean;
}

/**
 * Submit a video file for PTSD prediction.
 * Calls POST /predict with multipart/form-data.
 */
export async function predictVideo(
  file: File,
  onUploadProgress?: (percent: number) => void,
  options: PredictVideoOptions = {}
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("video", file);
  formData.append(
    "include_transcript",
    String(options.includeTranscript ?? true)
  );
  formData.append(
    "include_explanation",
    String(options.includeExplanation ?? true)
  );
  formData.append(
    "include_full_visual_audit",
    String(options.includeFullVisualAudit ?? true)
  );

  // Use XMLHttpRequest for upload progress tracking
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onUploadProgress) {
        const pct = Math.round((e.loaded / e.total) * 100);
        onUploadProgress(pct);
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status === 200) {
        try {
          const data: PredictionResponse = JSON.parse(xhr.responseText);
          resolve(data);
        } catch {
          reject(
            new ApiError(
              "Invalid JSON response from server",
              xhr.status,
              "PARSE_ERROR"
            )
          );
        }
      } else if (xhr.status === 0) {
        reject(
          new ApiError(
            "Cannot reach the server. Is the backend running on " +
              BASE_URL +
              "?",
            0,
            "NETWORK_ERROR"
          )
        );
      } else if (xhr.status === 422) {
        reject(
          new ApiError(
            "Invalid video format. Please upload an mp4, mov, or avi file.",
            422,
            "INVALID_FORMAT"
          )
        );
      } else if (xhr.status >= 500) {
        reject(
          new ApiError(
            "Server error (" + xhr.status + "). Please try again later.",
            xhr.status,
            "SERVER_ERROR"
          )
        );
      } else {
        reject(
          new ApiError(
            "Unexpected error: HTTP " + xhr.status,
            xhr.status,
            "UNKNOWN"
          )
        );
      }
    });

    xhr.addEventListener("error", () => {
      reject(
        new ApiError(
          "Network error. Please check your connection and ensure the backend is running.",
          0,
          "NETWORK_ERROR"
        )
      );
    });

    xhr.addEventListener("timeout", () => {
      reject(
        new ApiError(
          "Request timed out. The video may be too large or the server is busy.",
          0,
          "TIMEOUT"
        )
      );
    });

    xhr.open("POST", `${BASE_URL}/predict`);
    xhr.timeout = 15 * 60 * 1000; // long runs + large explanation payloads
    xhr.send(formData);
  });
}

/** Format bytes to human-readable size */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

/** Validate video file type */
export function isValidVideoType(file: File): boolean {
  const allowed = ["video/mp4", "video/quicktime", "video/avi", "video/x-msvideo", "video/x-ms-wmv", "video/webm"];
  const allowedExt = [".mp4", ".mov", ".avi", ".wmv", ".webm"];
  const ext = "." + file.name.split(".").pop()?.toLowerCase();
  return allowed.includes(file.type) || allowedExt.includes(ext);
}

/** Format latency in ms/s */
export function formatLatency(seconds: number): string {
  if (seconds < 1) return (seconds * 1000).toFixed(0) + " ms";
  return seconds.toFixed(2) + " s";
}

/** Human-readable stage names */
export function formatStageName(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
