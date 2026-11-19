import type { Metadata, Viewport } from "next";
import { Public_Sans } from "next/font/google";
import "./globals.css";

// Public Sans is the U.S. Web Design System typeface: the voice of federal forms online.
const publicSans = Public_Sans({ subsets: ["latin"], variable: "--font-sans", display: "swap" });

export const metadata: Metadata = {
  title: "TaxLens",
  description: "Answers from IRS forms for the tax year you mean, with the exact line cited.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={publicSans.variable}>
      <body>{children}</body>
    </html>
  );
}
