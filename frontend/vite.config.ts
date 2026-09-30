import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

/**
 * The dev server proxies `/api` to the backend so the browser makes same-origin
 * requests. That keeps CORS out of the development loop entirely -- the backend
 * still declares explicit CORS origins for direct access, but a misconfigured
 * origin cannot silently break local development.
 */
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
