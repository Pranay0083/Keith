"use client";

// useSessionEvents connects a session view to its event stream: it owns the
// reducer state, an EventStream with backoff and after_seq replay, and the
// gap watchdog. Events are batched per animation frame so a replay of
// thousands of events renders once, not thousands of times.

import { useCallback, useEffect, useReducer, useRef, type Dispatch } from "react";
import { getApi, isApiError } from "@/lib/api/client";
import type { SessionEvent } from "@/lib/api/events";
import { hasGap, initialSessionState, sessionReducer, type SessionAction, type SessionState } from "./reducer";
import { EventStream, type ProbeResult } from "./stream";

/** How long a gap in seq may persist before the stream reconnects to refill it. */
export const GAP_RESYNC_MS = 3_000;

export interface SessionEvents {
  state: SessionState;
  dispatch: Dispatch<SessionAction>;
  /** Reconnects now: the Retry button. */
  retry(): void;
}

export function useSessionEvents(sessionId: string): SessionEvents {
  const [state, dispatch] = useReducer(sessionReducer, sessionId, initialSessionState);
  const lastSeqRef = useRef(0);
  const streamRef = useRef<EventStream | null>(null);

  useEffect(() => {
    lastSeqRef.current = state.lastSeq;
  }, [state.lastSeq]);

  useEffect(() => {
    let cancelled = false;
    let frame = 0;
    const queue: SessionEvent[] = [];

    const flush = () => {
      frame = 0;
      if (queue.length > 0) dispatch({ type: "events", events: queue.splice(0) });
    };
    const schedule = () => {
      if (frame) return;
      frame =
        typeof requestAnimationFrame === "function"
          ? requestAnimationFrame(flush)
          : (setTimeout(flush, 16) as unknown as number);
    };

    void getApi().then((api) => {
      if (cancelled) return;
      const stream = new EventStream({
        connect: (afterSeq) => api.openEvents(sessionId, afterSeq),
        // Queued-but-unflushed events are not counted; replaying them again
        // is harmless because the reducer dedupes by seq.
        lastSeq: () => lastSeqRef.current,
        onEvent: (e) => {
          queue.push(e);
          schedule();
        },
        onConnection: (connection) => {
          if (!cancelled) dispatch({ type: "connection", connection });
        },
        probe: () =>
          api.getSession(sessionId).then(
            (): ProbeResult => "ok",
            (err: unknown): ProbeResult => {
              if (!isApiError(err)) return "ok";
              if (err.status === 401) return "unauthenticated";
              if (err.status === 403) return "forbidden";
              if (err.status === 404) return "not_found";
              return "ok";
            },
          ),
      });
      streamRef.current = stream;
      stream.start();
    });

    return () => {
      cancelled = true;
      if (frame) {
        if (typeof cancelAnimationFrame === "function") cancelAnimationFrame(frame);
        else clearTimeout(frame);
      }
      streamRef.current?.stop();
      streamRef.current = null;
    };
  }, [sessionId]);

  // A gap that does not fill on its own means the live channel dropped
  // events; reconnecting replays them from the store.
  const gap = hasGap(state);
  useEffect(() => {
    if (!gap) return;
    const t = setTimeout(() => streamRef.current?.resync(), GAP_RESYNC_MS);
    return () => clearTimeout(t);
  }, [gap, state.lastSeq]);

  const retry = useCallback(() => streamRef.current?.reconnectNow(), []);

  return { state, dispatch, retry };
}
