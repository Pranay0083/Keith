"use client";

// Helpers the org pages share: a keyed loader, error text, and the
// "couldn't load" banner.

import { usePathname } from "next/navigation";
import { useCallback, useEffect, useEffectEvent, useState } from "react";
import { isApiError } from "@/lib/api/client";
import { Banner, Button, LinkButton } from "@/components/ui";

type Settled<T> = { key: string; ok: true; data: T } | { key: string; ok: false; error: unknown };

export interface Loaded<T> {
  /** The loaded value; undefined while loading or after a failure. */
  data: T | undefined;
  /** The failure, if the last load failed. */
  error: unknown;
  loading: boolean;
  /** Loads again from scratch, showing the loading state. */
  retry: () => void;
  /**
   * Changes the loaded value in place (after a mutation or a quiet refresh).
   * A no-op once `key` has changed, so a late response can't clobber another
   * org's data.
   */
  update: (fn: (data: T) => T) => void;
}

/**
 * Runs `load` whenever `key` changes and keeps the result for that key.
 * Loading is derived (no result for the current key yet), so switching org
 * shows the loading state without setting state in an effect.
 */
export function useLoad<T>(key: string, load: () => Promise<T>): Loaded<T> {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<Settled<T> | null>(null);
  const current = `${attempt}\u0000${key}`;
  const run = useEffectEvent(() => load());

  useEffect(() => {
    let live = true;
    run().then(
      (data) => {
        if (live) setState({ key: current, ok: true, data });
      },
      (error: unknown) => {
        if (live) setState({ key: current, ok: false, error });
      },
    );
    return () => {
      live = false;
    };
  }, [current]);

  const retry = useCallback(() => setAttempt((n) => n + 1), []);
  const update = useCallback(
    (fn: (data: T) => T) => {
      setState((s) => (s !== null && s.ok && s.key === current ? { ...s, data: fn(s.data) } : s));
    },
    [current],
  );

  const settled = state !== null && state.key === current ? state : null;
  if (settled === null) return { data: undefined, error: undefined, loading: true, retry, update };
  if (settled.ok) return { data: settled.data, error: undefined, loading: false, retry, update };
  return { data: undefined, error: settled.error, loading: false, retry, update };
}

/** A sentence to show for a failed call. */
export function errorMessage(e: unknown): string {
  if (isApiError(e)) {
    if (e.status === 401) return "Your session has expired. Sign in again.";
    if (e.status === 0) return e.message || "Couldn't reach Keith. Check your connection and try again.";
    return e.message || "Something went wrong.";
  }
  if (e instanceof Error && e.message) return e.message;
  return "Something went wrong.";
}

/** The server's message for one request field, if the error carried one. */
export function fieldError(error: unknown, field: string): string | undefined {
  return isApiError(error) ? error.fields[field] : undefined;
}

/**
 * The message to show above a form, or null when every field error is
 * already shown next to one of `fields`.
 */
export function formError(error: unknown, fields: readonly string[]): string | null {
  if (!error) return null;
  if (isApiError(error)) {
    const keys = Object.keys(error.fields);
    if (keys.length > 0 && keys.every((k) => fields.includes(k))) return null;
  }
  return errorMessage(error);
}

/** The banner for a failed page load, with Retry (or Sign in after a 401). */
export function LoadError({ title, error, onRetry }: { title: string; error: unknown; onRetry: () => void }) {
  const pathname = usePathname();
  const expired = isApiError(error) && error.status === 401;
  return (
    <Banner
      tone="bad"
      title={title}
      actions={
        expired ? (
          <LinkButton href={`/login?next=${encodeURIComponent(pathname)}`} size="sm">
            Sign in
          </LinkButton>
        ) : (
          <Button size="sm" onClick={onRetry}>
            Retry
          </Button>
        )
      }
    >
      {errorMessage(error)}
    </Banner>
  );
}

/** Case-insensitive substring match for list filters. */
export function matchesFilter(text: string, filter: string): boolean {
  const q = filter.trim().toLowerCase();
  return q === "" || text.toLowerCase().includes(q);
}
