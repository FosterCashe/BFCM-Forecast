import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Forecast results",
  description: "Daily revenue forecast ranges and backtested expected error",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
