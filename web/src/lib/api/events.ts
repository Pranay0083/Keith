// The typed layer over events.gen.ts (generated from internal/events).
//
// tygo emits each payload struct and the event type constants, but not a
// mapping between them, and Go's string enums come out as plain `string`.
// This file adds both, and the type assertions at the bottom stop
// type-checking if the generated file and this one drift apart.

import * as gen from "./events.gen";

export type {
  ErrorPayload,
  FileChangedPayload,
  MessageAssistantPayload,
  MessageDeltaPayload,
  MessageUserPayload,
  PROpenedPayload,
  QuestionAskedPayload,
  SessionStatusPayload,
  ShellCreatedPayload,
  ToolCallFinishedPayload,
  ToolCallOutputPayload,
  ToolCallStartedPayload,
} from "./events.gen";

/** Every session status, in lifecycle order. */
export const SESSION_STATUSES = [
  "queued",
  "provisioning",
  "setting_up",
  "running",
  "waiting_for_user",
  "completed",
  "failed",
  "cancelled",
  "no_changes",
] as const;

export type SessionStatusName = (typeof SESSION_STATUSES)[number];

/** Statuses in which no more work happens without a new user message. */
export const TERMINAL_STATUSES: readonly SessionStatusName[] = [
  "completed",
  "failed",
  "cancelled",
  "no_changes",
];

export function isTerminal(status: SessionStatusName | null): boolean {
  return status !== null && TERMINAL_STATUSES.includes(status);
}

export function isSessionStatus(s: string): s is SessionStatusName {
  return (SESSION_STATUSES as readonly string[]).includes(s);
}

export type FileActionName = "created" | "modified" | "deleted";

/** Maps each event type to its payload. */
export interface EventPayloadMap {
  [gen.TypeSessionStatus]: gen.SessionStatusPayload;
  [gen.TypeMessageUser]: gen.MessageUserPayload;
  [gen.TypeMessageAssistant]: gen.MessageAssistantPayload;
  [gen.TypeMessageDelta]: gen.MessageDeltaPayload;
  [gen.TypeToolCallStarted]: gen.ToolCallStartedPayload;
  [gen.TypeToolCallOutput]: gen.ToolCallOutputPayload;
  [gen.TypeToolCallFinished]: gen.ToolCallFinishedPayload;
  [gen.TypeShellCreated]: gen.ShellCreatedPayload;
  [gen.TypeFileChanged]: gen.FileChangedPayload;
  [gen.TypeQuestionAsked]: gen.QuestionAskedPayload;
  [gen.TypePROpened]: gen.PROpenedPayload;
  [gen.TypeError]: gen.ErrorPayload;
}

export type EventType = keyof EventPayloadMap;

/** An event whose payload is typed by its `type`. */
export type SessionEvent = {
  [K in EventType]: Omit<gen.Event, "type" | "payload"> & {
    type: K;
    payload: EventPayloadMap[K];
  };
}[EventType];

/** The raw wire shape, before the type is checked. */
export type RawEvent = gen.Event;

/** Ephemeral events are published but never stored, and carry seq 0. */
export function isEphemeral(type: string): boolean {
  return type === gen.TypeMessageDelta;
}

const KNOWN_TYPES: ReadonlySet<string> = new Set<EventType>([
  gen.TypeSessionStatus,
  gen.TypeMessageUser,
  gen.TypeMessageAssistant,
  gen.TypeMessageDelta,
  gen.TypeToolCallStarted,
  gen.TypeToolCallOutput,
  gen.TypeToolCallFinished,
  gen.TypeShellCreated,
  gen.TypeFileChanged,
  gen.TypeQuestionAsked,
  gen.TypePROpened,
  gen.TypeError,
]);

/**
 * Parses one WebSocket frame into an event. It returns null for anything
 * that is not a well-formed event of a known type, so an older client keeps
 * working when the server adds event types.
 */
export function parseEvent(data: string): SessionEvent | null {
  let v: unknown;
  try {
    v = JSON.parse(data);
  } catch {
    return null;
  }
  if (typeof v !== "object" || v === null) return null;
  const e = v as Partial<gen.Event>;
  if (
    typeof e.session_id !== "string" ||
    typeof e.seq !== "number" ||
    !Number.isInteger(e.seq) ||
    e.seq < 0 ||
    typeof e.ts !== "string" ||
    typeof e.type !== "string" ||
    !KNOWN_TYPES.has(e.type) ||
    typeof e.payload !== "object" ||
    e.payload === null
  ) {
    return null;
  }
  return e as SessionEvent;
}

// --- Drift checks: these fail type-checking, not at runtime. ---

type Equals<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;
type Assert<T extends true> = T;

/* eslint-disable @typescript-eslint/no-unused-vars */
// Every generated event type constant has a payload mapping, and no more.
type _EventTypesMatch = Assert<Equals<gen.Type, EventType>>;
/* eslint-enable @typescript-eslint/no-unused-vars */

/**
 * The generated status and file action constants, for the runtime drift
 * test in events.test.ts (tygo types them as plain strings).
 */
export const GENERATED_STATUSES: readonly string[] = [
  gen.StatusQueued,
  gen.StatusProvisioning,
  gen.StatusSettingUp,
  gen.StatusRunning,
  gen.StatusWaitingForUser,
  gen.StatusCompleted,
  gen.StatusFailed,
  gen.StatusCancelled,
  gen.StatusNoChanges,
];

export const GENERATED_FILE_ACTIONS: readonly string[] = [
  gen.FileCreated,
  gen.FileModified,
  gen.FileDeleted,
];
