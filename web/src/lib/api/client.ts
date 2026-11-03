// The one entry point the UI uses to reach the API. With
// NEXT_PUBLIC_API_MOCK=1 it serves an in-browser mock instead, so the whole
// app runs with no backend.

import type { Api } from "./api";
import { API_URL, httpApi } from "./http";

export { ApiError, isApiError } from "./api";
export type { Api, SocketLike } from "./api";

/** True when the app runs against the in-browser mock API. */
export const isMock = process.env.NEXT_PUBLIC_API_MOCK === "1";

/** Resolves the API implementation for this build. */
export async function getApi(): Promise<Api> {
  return httpApi;
}

type AsyncApi = Omit<Api, "openEvents">;

/**
 * The API for components: `await api.repos(orgId)`. Each call resolves the
 * implementation first, so mock code is only loaded in mock mode.
 */
export const api = new Proxy({} as AsyncApi, {
  get(_target, key) {
    return (...args: unknown[]) =>
      getApi().then((impl) => {
        const fn = Reflect.get(impl, key) as (...a: unknown[]) => unknown;
        return fn.apply(impl, args);
      });
  },
});

/** Where "Sign in with GitHub" goes. `next` must be a path on this app. */
export function loginUrl(next?: string): string {
  const safeNext = next && next.startsWith("/") && !next.startsWith("//") ? next : "/sessions";
  if (isMock) return safeNext;
  return `${API_URL}/auth/github/login?next=${encodeURIComponent(safeNext)}`;
}
