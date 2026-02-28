import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Cairo", "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        title: ["Barlow Condensed", "Cairo", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      colors: {
        bg: "var(--bg)",
        surface: "var(--surface)",
        line: "var(--line)",
        text: "var(--text)",
        muted: "var(--muted)",
        teal: "var(--teal)",
        violet: "var(--violet)",
      },
      boxShadow: {
        glow: "0 0 0 1px color-mix(in oklab, var(--teal), transparent 72%), 0 12px 34px rgba(74,35,19,.12)",
      },
      keyframes: {
        floatIn: {
          "0%": { opacity: "0", transform: "translateY(14px) scale(0.99)" },
          "100%": { opacity: "1", transform: "translateY(0) scale(1)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      animation: {
        floatIn: "floatIn 500ms cubic-bezier(0.2, 0.7, 0.2, 1) both",
        shimmer: "shimmer 1.6s linear infinite",
      },
    },
  },
  plugins: [],
} satisfies Config;
