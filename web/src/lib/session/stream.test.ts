import { describe, expect, it } from "vitest";
import type { SocketLike } from "@/lib/api/api";
import type { SessionEvent } from "@/lib/api/events";
import { backoffDelay, DEFAULT_BACKOFF } from "./backoff";
import type { ConnectionState } from "./reducer";
import { EventStream, type ProbeResult, type Timers } from "./stream";

class FakeSocket implements SocketLike {
  onopen: (() => void) | null = null;
  onmessage: ((data: string) => void) | null = null;
  onclose: ((code: number, reason: string) => void) | null = null;
  closed: { code?: number } | null = null;

  constructor(readonly afterSeq: number) {}

  open() {
    this.onopen?.();
  }
  send(data: unknown) {
    this.onmessage?.(typeof data === "string" ? data : JSON.stringify(data));
  }
  serverClose(code: number) {
    this.onclose?.(code, "");
  }
  close(code?: number) {
    this.closed = { code };
  }
}

class FakeTimers implements Timers {
  now = 0;
  private seq = 0;
  private pending = new Map<number, { at: number; fn: () => void }>();

  setTimeout(fn: () => void, ms: number): unknown {
    const id = ++this.seq;
    this.pending.set(id, { at: this.now + ms, fn });
    return id;
  }
  clearTimeout(handle: unknown): void {
    this.pending.delete(handle as number);
  }
  /** Runs every timer due within `ms`, in time order. */
  advance(ms: number) {
    const end = this.now + ms;
    for (;;) {
      let next: [number, { at: number; fn: () => void }] | undefined;
      for (const entry of this.pending) if (entry[1].at <= end && (!next || entry[1].at < next[1].at)) next = entry;
      if (!next) break;
      this.pending.delete(next[0]);
      this.now = next[1].at;
      next[1].fn();
    }
    this.now = end;
  }
  get size() {
    return this.pending.size;
  }
}

function event(seq: number): SessionEvent {
  return { session_id: "s", seq, ts: "2026-11-02T10:00:00Z", type: "session.status", payload: { status: "running" } };
}

function setup(opts: { probe?: () => Promise<ProbeResult>; random?: () => number } = {}) {
  const timers = new FakeTimers();
  const sockets: FakeSocket[] = [];
  const events: SessionEvent[] = [];
  const states: ConnectionState[] = [];
  let lastSeq = 0;
  const stream = new EventStream({
    connect: (afterSeq) => {
      const s = new FakeSocket(afterSeq);
      sockets.push(s);
      return s;
    },
    lastSeq: () => lastSeq,
    onEvent: (e) => {
      events.push(e);
      if (e.seq > lastSeq) lastSeq = e.seq;
    },
    onConnection: (c) => states.push(c),
    probe: opts.probe,
    timers,
    random: opts.random ?? (() => 0),
  });
  const last = () => sockets[sockets.length - 1]!;
  return { stream, timers, sockets, events, states, last, setLastSeq: (n: number) => (lastSeq = n) };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

describe("backoffDelay", () => {
  it("doubles from 0.5 s with jitter in [0.5, 1) of the ceiling", () => {
    expect(backoffDelay(0, () => 0)).toBe(250);
    expect(backoffDelay(0, () => 0.999999)).toBe(500);
    expect(backoffDelay(1, () => 0)).toBe(500);
    expect(backoffDelay(2, () => 0)).toBe(1000);
    expect(backoffDelay(3, () => 0.5)).toBe(3000);
  });

  it("stays within [ceiling/2, ceiling] for any random value", () => {
    for (let attempt = 0; attempt < 12; attempt++) {
      const ceiling = Math.min(DEFAULT_BACKOFF.maxMs, DEFAULT_BACKOFF.baseMs * 2 ** attempt);
      for (const r of [0, 0.1, 0.5, 0.9, 0.999999]) {
        const d = backoffDelay(attempt, () => r);
        expect(d).toBeGreaterThanOrEqual(ceiling / 2);
        expect(d).toBeLessThanOrEqual(ceiling);
      }
    }
  });

  it("caps at 15 s and survives absurd inputs", () => {
    expect(backoffDelay(10, () => 0.999999)).toBe(15_000);
    expect(backoffDelay(10_000, () => 0)).toBe(7_500);
    expect(backoffDelay(-3, () => 0)).toBe(250);
    expect(backoffDelay(0, () => 7)).toBe(500);
  });
});

describe("EventStream", () => {
  it("connects with the last applied seq and forwards parsed events", () => {
    const t = setup();
    t.setLastSeq(41);
    t.stream.start();
    expect(t.sockets).toHaveLength(1);
    expect(t.last().afterSeq).toBe(41);
    expect(t.states.at(-1)).toEqual({ state: "connecting", attempt: 0 });

    t.last().open();
    expect(t.states.at(-1)).toEqual({ state: "open" });
    t.last().send(event(42));
    t.last().send("not json");
    t.last().send({ session_id: "s", seq: 43, ts: "x", type: "unknown.type", payload: {} });
    t.last().send({ session_id: "s", seq: -1, ts: "x", type: "error", payload: {} });
    expect(t.events.map((e) => e.seq)).toEqual([42]);
  });

  it("reconnects with backoff and resumes from the newest seq", () => {
    const t = setup();
    t.stream.start();
    t.last().open();
    t.last().send(event(1));
    t.last().send(event(2));
    t.last().serverClose(1006);
    expect(t.states.at(-1)).toEqual({ state: "reconnecting", attempt: 1, delayMs: 250 });

    t.timers.advance(249);
    expect(t.sockets).toHaveLength(1);
    t.timers.advance(1);
    expect(t.sockets).toHaveLength(2);
    expect(t.last().afterSeq).toBe(2);
  });

  it("grows the delay on consecutive failures and resets once a connection holds", () => {
    const t = setup();
    t.stream.start();
    const delays: number[] = [];
    for (let i = 0; i < 7; i++) {
      t.last().serverClose(1006);
      const s = t.states.at(-1);
      if (s?.state === "reconnecting") delays.push(s.delayMs);
      t.timers.advance(20_000);
    }
    expect(delays).toEqual([250, 500, 1000, 2000, 4000, 7500, 7500]);

    // Open but drop before 5 s: still backing off.
    t.last().open();
    t.timers.advance(4_999);
    t.last().serverClose(1006);
    expect(t.states.at(-1)).toMatchObject({ state: "reconnecting", delayMs: 7500 });
    t.timers.advance(7_500);

    // Open and hold for 5 s: the next drop starts over.
    t.last().open();
    t.timers.advance(5_000);
    t.last().serverClose(1001);
    expect(t.states.at(-1)).toEqual({ state: "reconnecting", attempt: 1, delayMs: 250 });
  });

  it("stops for good on 4401, 4403 and 4404", () => {
    for (const [code, reason] of [
      [4401, "unauthenticated"],
      [4403, "forbidden"],
      [4404, "not_found"],
    ] as const) {
      const t = setup();
      t.stream.start();
      t.last().open();
      t.last().serverClose(code);
      expect(t.states.at(-1)).toEqual({ state: "closed", reason });
      t.timers.advance(60_000);
      expect(t.sockets).toHaveLength(1);
    }
  });

  it("asks the API why a handshake failed and stops if access is gone", async () => {
    const t = setup({ probe: () => Promise.resolve("not_found") });
    t.stream.start();
    t.last().serverClose(1006); // never opened
    await flush();
    expect(t.states.at(-1)).toEqual({ state: "closed", reason: "not_found" });
    t.timers.advance(60_000);
    expect(t.sockets).toHaveLength(1);
  });

  it("keeps retrying when the probe finds nothing wrong", async () => {
    let probes = 0;
    const t = setup({
      probe: () => {
        probes++;
        return Promise.resolve("ok");
      },
    });
    t.stream.start();
    t.last().serverClose(1006);
    await flush();
    expect(probes).toBe(1);
    t.timers.advance(250);
    expect(t.sockets).toHaveLength(2);
  });

  it("does not probe when a connection that had opened drops", async () => {
    let probes = 0;
    const t = setup({
      probe: () => {
        probes++;
        return Promise.resolve("unauthenticated");
      },
    });
    t.stream.start();
    t.last().open();
    t.last().serverClose(1011);
    await flush();
    expect(probes).toBe(0);
    expect(t.states.at(-1)).toMatchObject({ state: "reconnecting" });
  });

  it("stop() closes the socket and cancels reconnects", () => {
    const t = setup();
    t.stream.start();
    t.last().open();
    const first = t.last();
    t.stream.stop();
    expect(first.closed).toEqual({ code: 1000 });
    expect(t.states.at(-1)).toEqual({ state: "closed", reason: "stopped" });
    expect(t.timers.size).toBe(0);

    // A pending reconnect is cancelled too.
    const u = setup();
    u.stream.start();
    u.last().serverClose(1006);
    u.stream.stop();
    u.timers.advance(60_000);
    expect(u.sockets).toHaveLength(1);
  });

  it("reconnectNow retries immediately, even after a fatal close", () => {
    const t = setup();
    t.stream.start();
    t.last().serverClose(4403);
    expect(t.states.at(-1)).toEqual({ state: "closed", reason: "forbidden" });
    t.stream.reconnectNow();
    expect(t.sockets).toHaveLength(2);
    expect(t.states.at(-1)).toEqual({ state: "connecting", attempt: 0 });
  });

  it("resync reopens from the last seq and ignores the old socket", () => {
    const t = setup();
    t.stream.start();
    const old = t.last();
    old.open();
    old.send(event(5));
    t.stream.resync();
    expect(old.closed).not.toBeNull();
    expect(t.sockets).toHaveLength(2);
    expect(t.last().afterSeq).toBe(5);

    // Late callbacks from the replaced socket change nothing.
    old.send(event(6));
    old.serverClose(1006);
    expect(t.events.map((e) => e.seq)).toEqual([5]);
    expect(t.states.at(-1)).toEqual({ state: "connecting", attempt: 0 });
  });

  it("start() is idempotent", () => {
    const t = setup();
    t.stream.start();
    t.stream.start();
    expect(t.sockets).toHaveLength(1);
  });
});
