import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import "./globals.css";
import { ThemeProvider } from "@/components/ThemeProvider";
import { AppShell } from "@/components/AppShell";
import { themeInitScript } from "@/lib/theme";

export const metadata: Metadata = {
  title: "Converge · ASTraM Traffic Intelligence",
  description:
    "Read-only operations dashboard over the Converge / ASTraM Bengaluru traffic disruption ML pipeline.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      data-theme="light"
      className={`${GeistSans.variable} ${GeistMono.variable}`}
    >
      <head>
        {/* Blocking (no async/defer) so the right theme lands on <html>
            before first paint - prevents a flash of the wrong theme. */}
        <script dangerouslySetInnerHTML={{ __html: themeInitScript() }} />
      </head>
      <body>
        <ThemeProvider>
          <AppShell>{children}</AppShell>
        </ThemeProvider>
      </body>
    </html>
  );
}
