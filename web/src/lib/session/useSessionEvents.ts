"use client";

// useSessionEvents connects a session view to its event stream: it owns the
// reducer state, an EventStream with backoff and after_seq replay, and the
// gap watchdog. Events are batched on a short timer so a replay of
// thousands of events renders once, not thousands of times. (A timer, not
// requestAnimationFrame: rAF stops in background tabs and headless browsers,
// and the state should keep up there too.)

import { useCallback, useEffect, useReducer, useRef, type Dispatch } from "react";
import { getApi, isApiError } from "@/lib/api/client";
import type { StreamEvent } from "@/lib/api/events";
import { hasGap, initialSessionState, sessionReducer, type SessionAction, type SessionState } from "./reducer";
import { EventStream, type ProbeResult } from "./stream";

/** How long a gap in seq may persist before the stream reconnects to refill it. */
export const GAP_RESYNC_MS = 3_000;

/** How long events collect before they are applied in one render. */
const BATCH_MS = 16;

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
    let timer: ReturnType<typeof setTimeout> | null = null;
    const queue: StreamEvent[] = [];

    const flush = () => {
      timer = null;
      if (queue.length > 0) dispatch({ type: "events", events: queue.splice(0) });
    };
    const schedule = () => {
      if (timer === null) timer = setTimeout(flush, BATCH_MS);
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
      if (timer !== null) clearTimeout(timer);
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
