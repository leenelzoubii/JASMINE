import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { ThemeProvider } from "@/components/providers/theme-provider";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "JASMINE — Autism Screening Platform",
  description:
    "A privacy-preserving AI platform for autism spectrum disorder screening using pose estimation. Trusted by healthcare professionals.",
  keywords: ["autism screening", "pose estimation", "ASD", "AI healthcare", "JASMINE"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} min-h-screen antialiased`}
      >
        <ThemeProvider>
          <div className="animate-fade-slide-up" style={{ animationDuration: "0.4s" }}>
            {children}
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
