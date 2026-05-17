import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "PTSD Risk Detection System | AI-Powered Mental Health Assessment",
  description:
    "Upload a clinical interview video and our multimodal AI model analyzes audio, visual, and linguistic cues to assess PTSD risk indicators in real time.",
  keywords: ["PTSD", "AI", "mental health", "detection", "multimodal", "deep learning"],
  authors: [{ name: "PTSD Detect Research" }],
  robots: "noindex, nofollow", // research tool — not for public indexing
  openGraph: {
    title: "PTSD Risk Detection System",
    description:
      "AI-powered multimodal PTSD risk assessment from clinical interview videos.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body>{children}</body>
    </html>
  );
}
