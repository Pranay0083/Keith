// The in-browser stand-in for `keith api` plus a worker: per-session event
// logs with seq, live subscribers, and a runner that plays scripted beats on
// timers. Sockets replay stored events after after_seq, then go live, just
// like the real WebSocket (docs/api.md, Events).

import type { SocketLike } from "../api";
import type { RawEvent } from "../events";
import { isEphemeral, isTerminal } from "../events";
import type { RepoDiff, Session } from "../types";
import type { Beat, Emit, Script } from "./scripts";
import { fileDiff } from "./unidiff";

type Listener = (e: RawEvent) => void;

interface Scheduler {
  now(): number;
  after(ms: number, fn: () => void): void;
}

export interface MockSessionRecord {
  session: Session;
  log: RawEvent[];
  subs: Set<Listener>;
  script: Script | null;
  pending: Beat[];
  busy: boolean;
  started: boolean;
  waitingFor: string | null;
  timers: Set<ReturnType<typeof setTimeout>>;
  /** repo -> path -> current content (null: deleted). */
  files: Record<string, Record<string, string | null>>;
  diffAt: Record<string, string>;
  turns: number;
}

/** Playback speed: ?mockSpeed=4 or localStorage keith.mockSpeed. */
function readSpeed(): number {
  if (typeof window === "undefined") return 1;
  try {
    const q = new URLSearchParams(window.location.search).get("mockSpeed");
    const v = Number(q ?? window.localStorage.getItem("keith.mockSpeed") ?? "1");
    return Number.isFinite(v) && v > 0 ? v : 1;
  } catch {
    return 1;
  }
}

/** Splits text into word-ish chunks for streaming. */
function chunksOf(text: string): string[] {
  const words = text.match(/\S+\s*/g) ?? [text];
  const out: string[] = [];
  for (let i = 0; i < words.length; i += 2) out.push(words.slice(i, i + 2).join(""));
  return out;
}

export class MockServer {
  readonly sessions = new Map<string, MockSessionRecord>();
  readonly sockets = new Set<MockSocket>();
  speed = readSpeed();

  create(session: Session, script: Script | null): MockSessionRecord {
    const rec: MockSessionRecord = {
      session,
      log: [],
      subs: new Set(),
      script,
      pending: script ? [...script.beats] : [],
      busy: false,
      started: false,
      waitingFor: null,
      timers: new Set(),
      files: {},
      diffAt: {},
      turns: 0,
    };
    this.sessions.set(session.id, rec);
    return rec;
  }

  /** Starts a live session's script the first time someone looks at it. */
  ensureStarted(rec: MockSessionRecord) {
    if (rec.started) return;
    rec.started = true;
    this.next(rec, this.live(rec));
  }

  /** Plays a script to its end (or its first question) with no delays. */
  playInstant(rec: MockSessionRecord, startMs: number) {
    rec.started = true;
    let t = startMs;
    const queue: Array<{ at: number; fn: () => void }> = [];
    const sched: Scheduler = {
      now: () => t,
      after: (ms, fn) => {
        // Spread instant beats out in virtual time so timestamps look real.
        queue.push({ at: t + Math.max(ms, 300), fn });
        queue.sort((a, b) => a.at - b.at);
      },
    };
    this.next(rec, sched);
    while (queue.length > 0) {
      const job = queue.shift()!;
      t = job.at;
      job.fn();
    }
  }

  private live(rec: MockSessionRecord): Scheduler {
    return {
      now: () => Date.now(),
      after: (ms, fn) => {
        const h = setTimeout(() => {
          rec.timers.delete(h);
          fn();
        }, ms / this.speed);
        rec.timers.add(h);
      },
    };
  }

  emit(rec: MockSessionRecord, ev: Emit, ts: number = Date.now()): RawEvent {
    const stored = !isEphemeral(ev.type);
    const event: RawEvent = {
      session_id: rec.session.id,
      seq: stored ? rec.log.length + 1 : 0,
      ts: new Date(ts).toISOString(),
      type: ev.type,
      payload: ev.payload,
    };
    if (stored) rec.log.push(event);
    this.track(rec, ev, event.ts);
    for (const l of rec.subs) l(event);
    return event;
  }

  /** Keeps the REST session object in step with its events. */
  private track(rec: MockSessionRecord, ev: Emit, ts: string) {
    const s = rec.session;
    if (ev.type === "session.status") {
      s.status = ev.payload.status as Session["status"];
      if (isTerminal(s.status)) {
        s.finished_at = ts;
        s.cancel_requested = false;
      } else {
        s.finished_at = null;
      }
    } else if (ev.type === "pr.opened") {
      const r = s.repos.find((x) => x.full_name === ev.payload.repo);
      if (r) {
        r.pr_number = ev.payload.number;
        r.pr_url = ev.payload.url;
      }
    }
  }

  private next(rec: MockSessionRecord, sched: Scheduler) {
    if (rec.waitingFor) {
      rec.busy = false;
      return;
    }
    const beat = rec.pending.shift();
    if (!beat) {
      rec.busy = false;
      return;
    }
    rec.busy = true;
    sched.after(beat.wait, () => this.run(rec, beat, sched, () => this.next(rec, sched)));
  }

  private run(rec: MockSessionRecord, beat: Beat, sched: Scheduler, done: () => void) {
    if ("emit" in beat) {
      this.emit(rec, beat.emit, sched.now());
      done();
    } else if ("say" in beat) {
      const { id, text } = beat.say;
      const parts = chunksOf(text);
      let i = 0;
      const step = () => {
        if (i < parts.length) {
          this.emit(rec, { type: "message.delta", payload: { message_id: id, text: parts[i++]! } }, sched.now());
          sched.after(45, step);
        } else {
          this.emit(rec, { type: "message.assistant", payload: { message_id: id, text } }, sched.now());
          done();
        }
      };
      step();
    } else if ("output" in beat) {
      const { callId, shellId, chunks, every } = beat.output;
      let i = 0;
      const step = () => {
        if (i < chunks.length) {
          this.emit(rec, { type: "tool_call.output", payload: { call_id: callId, shell_id: shellId, chunk: chunks[i++]! } }, sched.now());
          sched.after(every, step);
        } else {
          done();
        }
      };
      step();
    } else if ("file" in beat) {
      const { repo, path, content } = beat.file;
      (rec.files[repo] ??= {})[path] = content;
      rec.diffAt[repo] = new Date(sched.now()).toISOString();
      done();
    } else {
      this.emit(rec, { type: "question.asked", payload: beat.ask }, sched.now());
      this.emit(rec, { type: "session.status", payload: { status: "waiting_for_user" } }, sched.now());
      rec.waitingFor = beat.ask.question_id;
      rec.busy = false;
    }
  }

  /** Queues beats ahead of the rest of the script and makes sure they play. */
  inject(rec: MockSessionRecord, beats: Beat[]) {
    rec.pending.unshift(...beats);
    rec.started = true;
    if (!rec.busy) this.next(rec, this.live(rec));
  }

  /** Resumes after an answer: the answer, then the script's continuation. */
  answer(rec: MockSessionRecord, messageId: string, userId: string, questionId: string, text: string) {
    rec.waitingFor = null;
    const after = rec.script?.afterAnswer?.(text) ?? [];
    this.inject(rec, [
      { wait: 500, emit: { type: "message.user", payload: { message_id: messageId, user_id: userId, text, question_id: questionId } } },
      { wait: 150, emit: { type: "session.status", payload: { status: "running" } } },
      ...after,
    ]);
  }

  stop(rec: MockSessionRecord) {
    for (const h of rec.timers) clearTimeout(h);
    rec.timers.clear();
    rec.pending = [];
    rec.waitingFor = null;
    rec.busy = false;
    rec.session.cancel_requested = true;
    this.inject(rec, [{ wait: 700, emit: { type: "session.status", payload: { status: "cancelled", reason: "Stopped by you." } } }]);
  }

  subscribe(rec: MockSessionRecord, l: Listener): () => void {
    rec.subs.add(l);
    return () => rec.subs.delete(l);
  }

  diff(rec: MockSessionRecord): RepoDiff[] {
    return rec.session.repos.map((r) => {
      const current = rec.files[r.full_name] ?? {};
      const base = rec.script?.baseFiles?.[r.full_name] ?? {};
      const text = Object.keys(current)
        .sort()
        .map((p) => fileDiff(p, base[p] ?? null, current[p] ?? null))
        .join("");
      return { full_name: r.full_name, diff: text, truncated: false, updated_at: rec.diffAt[r.full_name] ?? null };
    });
  }

  /** Drops every open socket as a network failure would (window.__keithMock.drop()). */
  dropAll() {
    for (const s of [...this.sockets]) s.drop();
  }
}

/** A SocketLike backed by the mock server, with a little latency. */
export class MockSocket implements SocketLike {
  onopen: (() => void) | null = null;
  onmessage: ((data: string) => void) | null = null;
  onclose: ((code: number, reason: string) => void) | null = null;
  private closed = false;
  private unsub: (() => void) | null = null;

  constructor(
    private readonly server: MockServer,
    sessionId: string,
    afterSeq: number,
  ) {
    setTimeout(() => {
      if (this.closed) return;
      const rec = server.sessions.get(sessionId);
      if (!rec) {
        this.closed = true;
        this.onclose?.(4404, "session not found");
        return;
      }
      server.sockets.add(this);
      this.onopen?.();
      for (const e of rec.log) if (e.seq > afterSeq) this.deliver(e);
      this.unsub = server.subscribe(rec, (e) => this.deliver(e));
      server.ensureStarted(rec);
    }, 150);
  }

  private deliver(e: RawEvent) {
    if (!this.closed) this.onmessage?.(JSON.stringify(e));
  }

  close() {
    this.closed = true;
    this.unsub?.();
    this.unsub = null;
    this.server.sockets.delete(this);
  }

  /** Simulates the network dropping the connection. */
  drop() {
    const onclose = this.onclose;
    this.close();
    onclose?.(1006, "");
  }
}
