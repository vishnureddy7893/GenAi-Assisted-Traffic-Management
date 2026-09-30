/**
 * Vitest setup.
 *
 * `fetch` is stubbed per-test rather than globally faked, so a component that
 * makes an unexpected request fails the test instead of silently receiving a
 * default response.
 */

import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});
