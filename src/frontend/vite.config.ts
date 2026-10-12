import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api and /gallery to the FastAPI backend so the
// browser never hits CORS during local development. In production, point
// VITE_API_BASE_URL at the deployed backend (see .env.example).
const API_TARGET = process.env.VITE_DEV_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: API_TARGET, changeOrigin: true },
      "/gallery": { target: API_TARGET, changeOrigin: true },
      "/health": { target: API_TARGET, changeOrigin: true },
    },
  },
});
