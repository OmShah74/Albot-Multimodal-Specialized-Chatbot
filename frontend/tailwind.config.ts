import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        primary: {
          DEFAULT: "#9333ea", // purple-600
          hover: "#7e22ce",   // purple-700
          glow: "#a855f7",    // purple-500
        },
        dark: {
          bg: "#000000",
          card: "#0a0a0a",
          border: "#1f1f1f",
        }
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic":
          "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
        "purple-glow": "radial-gradient(circle at center, rgba(168, 85, 247, 0.15) 0%, transparent 70%)",
      },
    },
  },
  plugins: [],
};
export default config;
