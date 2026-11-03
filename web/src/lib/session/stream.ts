// EventStream keeps one session's event WebSocket alive: it connects with
// the last applied seq, reconnects with backoff, and stops for good on the
// close codes that mean "don't come back" (docs/api.md, Close codes). It
// knows nothing about React; useSessionEvents wires it to the reducer.

import type { SocketLike } from "@/lib/api/api";
import { parseEvent, type StreamEvent } from "@/lib/api/events";
import { backoffDelay, DEFAULT_BACKOFF, type BackoffOptions } from "./backoff";
import type { ConnectionState } from "./reducer";

/** Why a failed connection failed, found by asking the REST API. */
export type ProbeResult = "ok" | "unauthenticated" | "forbidden" | "not_found";

export interface Timers {
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
}

export interface EventStreamOptions {
  /** Opens a socket that replays stored events after afterSeq, then goes live. */
  connect(afterSeq: number): SocketLike;
  /** The highest stored seq the UI has applied. */
  lastSeq(): number;
  onEvent(event: StreamEvent): void;
  onConnection(state: ConnectionState): void;
  /**
   * Called when a socket closes without ever opening, which is how a browser
   * reports a rejected handshake (401/403/404 are invisible to it).
   */
  probe?(): Promise<ProbeResult>;
  timers?: Timers;
  random?: () => number;
  backoff?: BackoffOptions;
  /** How long a connection must stay open before the attempt count resets. */
  stableAfterMs?: number;
}

/** Close codes after which reconnecting cannot help. */
const FATAL_CLOSE: Record<number, "unauthenticated" | "forbidden" | "not_found"> = {
  4401: "unauthenticated",
  4403: "forbidden",
  4404: "not_found",
};

const defaultTimers: Timers = {
  setTimeout: (fn, ms) => globalThis.setTimeout(fn, ms),
  clearTimeout: (h) => globalThis.clearTimeout(h as ReturnType<typeof globalThis.setTimeout>),
};

export class EventStream {
  private readonly opts: EventStreamOptions;
  private readonly timers: Timers;
  private socket: SocketLike | null = null;
  private opened = false;
  private attempt = 0;
  private retryTimer: unknown = null;
  private stableTimer: unknown = null;
  private stopped = true;
  /** Bumped on every connect; callbacks from older sockets are ignored. */
  private generation = 0;

  constructor(opts: EventStreamOptions) {
    this.opts = opts;
    this.timers = opts.timers ?? defaultTimers;
  }

  /** Connects. Calling it again while running does nothing. */
  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    this.attempt = 0;
    this.open();
  }

  /** Closes the socket and cancels any pending reconnect. */
  stop(): void {
    this.stopped = true;
    this.clearTimers();
    this.dropSocket();
    this.opts.onConnection({ state: "closed", reason: "stopped" });
  }

  /** Reconnects right away (the Retry button), even after a fatal close. */
  reconnectNow(): void {
    this.stopped = false;
    this.attempt = 0;
    this.clearTimers();
    this.dropSocket();
    this.open();
  }

  /** Reopens the socket from the last applied seq, to refill a gap. */
  resync(): void {
    if (this.stopped) return;
    this.clearTimers();
    this.dropSocket();
    this.open();
  }

  /** The current retry attempt (0 while healthy). */
  get attempts(): number {
    return this.attempt;
  }

  private open(): void {
    const gen = ++this.generation;
    this.opened = false;
    this.opts.onConnection({ state: "connecting", attempt: this.attempt });
    let sock: SocketLike;
    try {
      sock = this.opts.connect(this.opts.lastSeq());
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.socket = sock;
    sock.onopen = () => {
      if (gen !== this.generation) return;
      this.opened = true;
      this.opts.onConnection({ state: "open" });
      this.stableTimer = this.timers.setTimeout(() => {
        this.stableTimer = null;
        this.attempt = 0;
      }, this.opts.stableAfterMs ?? 5_000);
    };
    sock.onmessage = (data) => {
      if (gen !== this.generation) return;
      const ev = parseEvent(data);
      if (ev) this.opts.onEvent(ev);
    };
    sock.onclose = (code) => {
      if (gen !== this.generation) return;
      this.socket = null;
      if (this.stableTimer !== null) {
        this.timers.clearTimeout(this.stableTimer);
        this.stableTimer = null;
      }
      if (this.stopped) return;
      const fatal = FATAL_CLOSE[code];
      if (fatal) {
        this.stopped = true;
        this.opts.onConnection({ state: "closed", reason: fatal });
        return;
      }
      const neverOpened = !this.opened;
      this.scheduleReconnect();
      if (neverOpened && this.opts.probe) void this.runProbe();
    };
  }

  private scheduleReconnect(): void {
    const delayMs = backoffDelay(this.attempt, this.opts.random, this.opts.backoff ?? DEFAULT_BACKOFF);
    this.attempt += 1;
    this.opts.onConnection({ state: "reconnecting", attempt: this.attempt, delayMs });
    this.retryTimer = this.timers.setTimeout(() => {
      this.retryTimer = null;
      if (!this.stopped) this.open();
    }, delayMs);
  }

  private async runProbe(): Promise<void> {
    let result: ProbeResult;
    try {
      result = await this.opts.probe!();
    } catch {
      return; // can't tell; keep retrying
    }
    // A socket that opened since then proves access; otherwise trust the probe.
    if (result === "ok" || this.stopped || this.opened) return;
    this.stopped = true;
    this.clearTimers();
    this.dropSocket();
    this.opts.onConnection({ state: "closed", reason: result });
  }

  private clearTimers(): void {
    if (this.retryTimer !== null) this.timers.clearTimeout(this.retryTimer);
    if (this.stableTimer !== null) this.timers.clearTimeout(this.stableTimer);
    this.retryTimer = null;
    this.stableTimer = null;
  }

  private dropSocket(): void {
    const s = this.socket;
    this.socket = null;
    this.generation += 1; // silence the old socket's callbacks
    if (s) {
      s.onopen = s.onmessage = s.onclose = null;
      s.close(1000, "client closing");
    }
  }
}
