/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The browser talks only to the Vite dev server; these paths are forwarded to the FastAPI backend,
// so no CORS configuration is needed. Evidence URLs returned by the API ("/inspections/<id>/annotated")
// work unchanged for the same reason.
const API = process.env.VAV_API ?? "http://127.0.0.1:8000";
const proxied = ["/inspect", "/inspections", "/products", "/health"];

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: Object.fromEntries(proxied.map((p) => [p, { target: API, changeOrigin: true }])) },
  test: { environment: "jsdom", globals: true, setupFiles: ["./src/test-setup.ts"] },
});