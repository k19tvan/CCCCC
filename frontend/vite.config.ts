import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` forwards /api to VITE_API_BASE_URL (see .env.example).
// `npm run build` writes to dist/, which the backend serves at "/" when present.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "VITE_");
  const target = env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
  return {
    plugins: [react()],
    server: { proxy: { "/api": target } },
  };
});
