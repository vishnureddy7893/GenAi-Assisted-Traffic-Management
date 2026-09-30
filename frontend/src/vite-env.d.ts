/// <reference types="vite/client" />

/**
 * Typed environment variables.
 *
 * Declared explicitly rather than relying on Vite's permissive index signature,
 * so a typo in an env key is a compile error instead of `undefined` at runtime.
 */
interface ImportMetaEnv {
  /** Backend origin. Empty in development: the dev server proxies `/api`. */
  readonly VITE_API_BASE_URL?: string;
  /** Milliseconds before an API request is abandoned. */
  readonly VITE_API_TIMEOUT_MS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
