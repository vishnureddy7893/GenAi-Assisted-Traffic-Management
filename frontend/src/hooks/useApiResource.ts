/**
 * Four-state resource hook.
 *
 * The important design point is that **"not implemented yet" is a distinct state
 * from "error", and both are distinct from "empty success"**.
 *
 * A conventional `{ data, loading, error }` triple collapses them: a 501 becomes
 * an error (alarming the operator about a phase boundary that is working as
 * designed), or worse, a failed fetch falls through to `data = []` and the panel
 * renders empty. In a traffic dashboard an empty panel reads as *"there is no
 * congestion and there are no incidents"* — a specific, false claim about the
 * world. Every state below is therefore rendered differently and none of them
 * can silently degrade into another.
 */

import { useCallback, useEffect, useState } from "react";
import { ApiError, apiGet } from "../api/client";

/** Discriminated union of every outcome an API-backed panel can be in. */
export type Resource<T> =
  | { readonly state: "loading" }
  | { readonly state: "success"; readonly data: T }
  | {
      /** The capability belongs to a later development phase. Not a fault. */
      readonly state: "unavailable";
      readonly message: string;
      readonly phase: number | null;
    }
  | {
      readonly state: "error";
      readonly message: string;
      /** True when the backend could not be reached at all. */
      readonly unreachable: boolean;
    };

export interface UseApiResourceResult<T> {
  readonly resource: Resource<T>;
  /** Re-run the request. Bound to the retry control on error states. */
  readonly refresh: () => void;
}

/**
 * Fetch `path` and expose its outcome as a {@link Resource}.
 *
 * @param path      API path to GET.
 * @param pollMs    Optional polling interval. Used only for liveness; traffic
 *                  data is never polled speculatively, because a stale repaint
 *                  is indistinguishable from a fresh one to a reader.
 */
export function useApiResource<T>(path: string, pollMs?: number): UseApiResourceResult<T> {
  const [resource, setResource] = useState<Resource<T>>({ state: "loading" });
  const [nonce, setNonce] = useState(0);

  const refresh = useCallback(() => {
    setNonce((value) => value + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;

    const run = async (): Promise<void> => {
      try {
        const data = await apiGet<T>(path, controller.signal);
        if (!cancelled) setResource({ state: "success", data });
      } catch (error) {
        if (cancelled) return;
        if (error instanceof ApiError && error.isNotAvailableYet) {
          setResource({
            state: "unavailable",
            message: error.message,
            phase: error.phase,
          });
          return;
        }
        const apiError = error instanceof ApiError ? error : null;
        setResource({
          state: "error",
          message: apiError?.message ?? "Unexpected failure while contacting the backend.",
          unreachable: apiError?.isUnreachable ?? true,
        });
      }
    };

    void run();

    const timer =
      pollMs !== undefined ? window.setInterval(() => void run(), pollMs) : undefined;

    return () => {
      cancelled = true;
      controller.abort();
      if (timer !== undefined) window.clearInterval(timer);
    };
  }, [path, pollMs, nonce]);

  return { resource, refresh };
}
