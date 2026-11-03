"use client";

// The live session page. Everything shown here comes from the reducer state
// that useSessionEvents folds from the event stream; the REST session detail
// only adds what events don't carry (prompt, repos, model).

import Link from "next/link";
import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";
import { api, isApiError, loginUrl } from "@/lib/api/client";
import { isTerminal } from "@/lib/api/events";
import type { Session } from "@/lib/api/types";
import { cx } from "@/lib/cx";
import {
  currentStep,
  openQuestion,
  pendingAnswer,
  visibleError,
  type ConnectionState,
  type SessionState,
} from "@/lib/session/reducer";
import { useSessionEvents } from "@/lib/session/useSessionEvents";
import { useMe } from "../app-shell";
import { ChevronIcon, ExternalIcon, PullRequestIcon, StopIcon } from "../icons";
import { RelativeTime } from "../time";
import { Banner, Button, EmptyState, LinkButton, Spinner, StatusBadge } from "../ui";
import { Chat } from "./chat";
import { Composer } from "./composer";
import { describeStep, firstLine } from "./format";
import { useMediaQuery } from "./hooks";
import { PRCard } from "./pr-card";
import { Timeline } from "./timeline";
import { followTab, Workspace, type TabId } from "./workspace";

export function SessionView({ id }: { id: string }) {
  const { state, dispatch, retry } = useSessionEvents(id);
  const me = useMe();
  const [session, setSession] = useState<Session | null>(null);
  const [loadError, setLoadError] = useState<{ status: number; message: string } | null>(null);
  const [reload, setReload] = useState(0);
  const [stopRequested, setStopRequested] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const wide = useMediaQuery("(min-width: 1280px)");
  const lg = useMediaQuery("(min-width: 1024px)");
  const [pane, setPane] = useState<"chat" | "work">("chat");
  const [selected, setSelected] = useState<TabId | null>(null);
  const [follow, setFollow] = useState(true);

  // The REST detail: on load, when the session ends, and when a PR opens
  // (both change fields the header shows).
  const terminal = isTerminal(state.status);

  // A stop applies to the run it was asked for. Once that run ends, forget
  // it, so a follow-up that resumes the session can be stopped again.
  if (stopRequested && terminal) setStopRequested(false);

  useEffect(() => {
    let live = true;
    api.getSession(id).then(
      (s) => {
        if (!live) return;
        setSession(s);
        setLoadError(null);
        dispatch({ type: "session_loaded", session: s });
      },
      (e: unknown) => {
        if (!live) return;
        setLoadError({
          status: isApiError(e) ? e.status : 0,
          message: e instanceof Error ? e.message : "Couldn't load the session.",
        });
      },
    );
    return () => {
      live = false;
    };
  }, [id, dispatch, terminal, state.prs.length, reload]);

  const retryAll = useCallback(() => {
    dispatch({ type: "dismiss_errors" });
    retry();
    setReload((n) => n + 1);
  }, [dispatch, retry]);

  async function sendMessage(text: string) {
    setActionError(null);
    const { message_id } = await api.sendMessage(id, text);
    dispatch({ type: "message_sent", messageId: message_id, text, ts: new Date().toISOString() });
  }

  async function answer(questionId: string, text: string) {
    setActionError(null);
    const { message_id } = await api.answer(id, questionId, text);
    dispatch({ type: "message_sent", messageId: message_id, text, ts: new Date().toISOString(), questionId });
  }

  async function stop() {
    setActionError(null);
    try {
      await api.stop(id);
      setStopRequested(true);
    } catch (e) {
      setActionError(e instanceof Error ? e.message : "Couldn't stop the session.");
    }
  }

  if (loadError && !session && (loadError.status === 404 || loadError.status === 403)) {
    return (
      <div className="mx-auto max-w-xl p-6">
        <EmptyState
          title="Session not found"
          description="It doesn't exist, or it belongs to an organization you're not a member of."
          action={<LinkButton href="/sessions">Back to sessions</LinkButton>}
        />
      </div>
    );
  }

  const active = state.status !== null && !terminal;
  const question = openQuestion(state);
  const step = currentStep(state);
  const error = visibleError(state);
  const conn = state.connection;
  const streamGone = conn.state === "closed" && conn.reason !== "stopped";

  // Workspace tab: follow the agent unless the user picked one.
  const valid = (t: TabId | null): t is TabId =>
    t !== null &&
    (t === "diff" || (t === "steps" && !wide) || state.shells.some((s) => `shell:${s.id}` === t));
  const fallback: TabId = !wide ? "steps" : state.shells[0] ? `shell:${state.shells[0].id}` : "diff";
  const wanted = follow ? (followTab(state) ?? selected) : selected;
  const activeTab: TabId = valid(wanted) ? wanted : fallback;
  const selectTab = (t: TabId) => {
    setSelected(t);
    setFollow(false);
    if (!lg) setPane("work");
  };
  const toggleFollow = () => {
    if (follow) {
      setSelected(activeTab);
      setFollow(false);
    } else {
      setFollow(true);
    }
  };

  const title = session ? firstLine(session.prompt) || session.issue_url || "Untitled task" : "";

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="shrink-0 border-b border-line px-4 py-2.5">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <nav aria-label="Breadcrumb" className="flex min-w-[min(16rem,100%)] flex-1 items-center gap-1 text-[13px]">
            <Link href="/sessions" className="shrink-0 rounded text-muted hover:text-fg">
              Sessions
            </Link>
            <ChevronIcon className="size-3 text-subtle" />
            <h1 className="min-w-0 truncate font-medium text-fg" title={session?.prompt}>
              {title || <span className="text-subtle">Loading…</span>}
            </h1>
          </nav>
          <div className="flex items-center gap-2">
            <ConnectionDot connection={conn} />
            <StatusBadge status={state.status} />
            <StopControl
              active={active}
              stopping={stopRequested || (session?.cancel_requested ?? false)}
              onStop={stop}
            />
          </div>
        </div>
        {session ? <SessionMeta session={session} state={state} /> : null}
      </header>

      <SessionBanners
        id={id}
        connection={conn}
        error={error}
        statusReason={terminal ? state.statusReason : ""}
        status={state.status}
        loadError={session ? null : (loadError?.message ?? null)}
        actionError={actionError}
        onRetry={retryAll}
        onDismiss={() => dispatch({ type: "dismiss_errors" })}
      />

      {!lg ? (
        <div className="flex shrink-0 gap-1 border-b border-line bg-panel p-1" role="group" aria-label="Show">
          {(["chat", "work"] as const).map((p) => (
            <button
              key={p}
              type="button"
              aria-pressed={pane === p}
              onClick={() => setPane(p)}
              className={cx(
                "h-7 flex-1 rounded text-[13px]",
                pane === p ? "bg-raised font-medium text-fg shadow-[0_0_0_1px_var(--line)]" : "text-muted",
              )}
            >
              {p === "chat" ? "Conversation" : "Workspace"}
            </button>
          ))}
        </div>
      ) : null}

      <div
        className={cx(
          "grid min-h-0 flex-1",
          wide
            ? "grid-cols-[minmax(340px,420px)_minmax(300px,380px)_minmax(0,1fr)]"
            : lg
              ? "grid-cols-[minmax(320px,400px)_minmax(0,1fr)]"
              : "grid-cols-1",
        )}
      >
        <section
          aria-label="Conversation"
          className={cx("min-h-0 flex-col border-line lg:border-r", !lg && pane !== "chat" ? "hidden" : "flex")}
        >
          <Chat
            items={state.chat}
            emptyText={
              state.status === "queued" || state.status === null
                ? "Waiting for a worker to pick up this task…"
                : "The agent is getting started…"
            }
            viewer={{ meId: me?.user.id ?? null, creator: session?.created_by ?? null }}
          />
          {step ? (
            <div className="flex shrink-0 items-center gap-2 border-t border-line px-3 py-1.5 text-xs" role="status">
              <Spinner className="size-3.5 text-accent-text" />
              <span className="font-mono text-fg">{step.tool}</span>
              <span className="min-w-0 truncate font-mono text-muted">{describeStep(step)}</span>
            </div>
          ) : null}
          {state.prs.length > 0 ? (
            <div className="flex shrink-0 flex-col gap-2 border-t border-line p-3">
              {state.prs.map((pr) => (
                <PRCard key={pr.repo} pr={pr} />
              ))}
            </div>
          ) : null}
          <Composer
            status={state.status}
            question={question}
            pendingAnswer={pendingAnswer(state)}
            disabled={streamGone || state.status === null}
            onSend={sendMessage}
            onAnswer={answer}
          />
        </section>

        {wide ? (
          <section aria-label="Steps" className="min-h-0 border-r border-line">
            <Timeline steps={state.steps} onShowShell={(s) => selectTab(`shell:${s}`)} className="h-full" />
          </section>
        ) : null}

        <section aria-label="Workspace" className={cx("min-h-0 min-w-0", !lg && pane !== "work" && "hidden")}>
          <Workspace
            state={state}
            withSteps={!wide}
            active={activeTab}
            follow={follow}
            onSelect={selectTab}
            onFollow={toggleFollow}
            live={active}
          />
        </section>
      </div>
    </div>
  );
}

function SessionMeta({ session, state }: { session: Session; state: SessionState }) {
  return (
    <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-subtle">
      <ul className="flex flex-wrap items-center gap-1.5" aria-label="Repos">
        {session.repos.map((r) => (
          <li key={r.repo_id} className="rounded border border-line bg-panel px-1.5 py-0.5 font-mono text-muted" title={`Branch ${r.branch}`}>
            {r.full_name}
          </li>
        ))}
      </ul>
      {session.issue_url ? (
        <a href={session.issue_url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 hover:text-fg">
          {issueLabel(session.issue_url)}
          <ExternalIcon className="size-3" />
        </a>
      ) : null}
      <span className="font-mono">
        {session.provider} · {session.model}
      </span>
      <span>
        started <RelativeTime iso={session.created_at} />
        {session.created_by ? ` by ${session.created_by.login}` : ""}
      </span>
      {state.prs.map((pr) => (
        <a
          key={pr.repo}
          href={pr.url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1 font-medium text-ok hover:underline"
        >
          <PullRequestIcon className="size-3.5" />
          {pr.repo}#{pr.number}
        </a>
      ))}
    </div>
  );
}

function issueLabel(url: string): string {
  const m = /github\.com\/([^/]+\/[^/]+)\/issues\/(\d+)/.exec(url);
  return m ? `${m[1]}#${m[2]}` : url;
}

function ConnectionDot({ connection }: { connection: ConnectionState }) {
  const { label, cls } =
    connection.state === "open"
      ? { label: "Live", cls: "bg-ok" }
      : connection.state === "reconnecting"
        ? { label: "Reconnecting", cls: "bg-warn pulse-dot" }
        : connection.state === "closed"
          ? { label: "Offline", cls: "bg-bad" }
          : { label: "Connecting", cls: "bg-subtle pulse-dot" };
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-subtle" role="status" aria-label={`Event stream: ${label}`}>
      <span className={cx("size-1.5 rounded-full", cls)} aria-hidden />
      <span aria-hidden>{label}</span>
    </span>
  );
}

function StopControl({ active, stopping, onStop }: { active: boolean; stopping: boolean; onStop(): Promise<void> }) {
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const stopRun = useRef<HTMLButtonElement>(null);
  // Focus "Stop run" when the confirm opens, once: the header re-renders on
  // every event batch, and refocusing then would pull the user out of
  // whatever they moved on to (the composer, say).
  useEffect(() => {
    if (confirm) stopRun.current?.focus();
  }, [confirm]);
  if (!active) return null;
  if (stopping) {
    return (
      <Button size="sm" disabled loading>
        Stopping…
      </Button>
    );
  }
  if (!confirm) {
    return (
      <Button size="sm" onClick={() => setConfirm(true)}>
        <StopIcon className="size-3.5" />
        Stop
      </Button>
    );
  }
  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Escape") setConfirm(false);
  };
  return (
    <div role="group" aria-label="Confirm stop" className="flex items-center gap-1.5" onKeyDown={onKeyDown}>
      <span className="text-xs text-muted">Stop this run?</span>
      <Button
        size="sm"
        variant="danger"
        loading={busy}
        ref={stopRun}
        onClick={async () => {
          setBusy(true);
          await onStop();
          setBusy(false);
          setConfirm(false);
        }}
      >
        Stop run
      </Button>
      <Button size="sm" variant="ghost" onClick={() => setConfirm(false)}>
        Keep running
      </Button>
    </div>
  );
}

interface BannersProps {
  id: string;
  connection: ConnectionState;
  error: ReturnType<typeof visibleError>;
  status: SessionState["status"];
  statusReason: string;
  loadError: string | null;
  actionError: string | null;
  onRetry(): void;
  onDismiss(): void;
}

function SessionBanners({ id, connection, error, status, statusReason, loadError, actionError, onRetry, onDismiss }: BannersProps) {
  const items: React.ReactNode[] = [];

  if (connection.state === "closed" && connection.reason === "unauthenticated") {
    items.push(
      <Banner
        key="auth"
        tone="bad"
        title="You've been signed out"
        actions={<LinkButton href={loginUrl(`/sessions/${id}`)} size="sm">Sign in</LinkButton>}
      >
        Sign in again to keep watching this session.
      </Banner>,
    );
  } else if (connection.state === "closed" && connection.reason !== "stopped") {
    items.push(
      <Banner
        key="gone"
        tone="bad"
        title="This session is no longer available"
        actions={<LinkButton href="/sessions" size="sm">Back to sessions</LinkButton>}
      >
        It was deleted, or you no longer have access to its organization.
      </Banner>,
    );
  } else if (connection.state === "reconnecting" && connection.attempt >= 2) {
    items.push(
      <Banner
        key="conn"
        tone="warn"
        title="Lost the live connection"
        actions={
          <Button size="sm" onClick={onRetry}>
            Retry now
          </Button>
        }
      >
        Reconnecting (attempt {connection.attempt}). Nothing is lost: missed events replay when it reconnects.
      </Banner>,
    );
  }

  if (error) {
    items.push(
      <Banner
        key={`err-${error.seq}`}
        tone="bad"
        title={
          <>
            The agent hit an error
            {error.code ? <span className="ml-1.5 font-mono text-xs font-normal text-muted">{error.code}</span> : null}
          </>
        }
        actions={
          <>
            <Button size="sm" onClick={onRetry} title="Reconnect to the event stream and reload the session">
              Retry
            </Button>
            <Button size="sm" variant="ghost" onClick={onDismiss}>
              Dismiss
            </Button>
          </>
        }
      >
        {error.message}
      </Banner>,
    );
  }

  if (loadError) {
    items.push(
      <Banner key="load" tone="bad" title="Couldn't load the session" actions={<Button size="sm" onClick={onRetry}>Retry</Button>}>
        {loadError}
      </Banner>,
    );
  }

  if (actionError) {
    items.push(
      <Banner key="action" tone="bad">
        {actionError}
      </Banner>,
    );
  }

  // The reason repeats the error banner when there is one, so show it alone.
  if (statusReason && !error && (status === "failed" || status === "cancelled" || status === "no_changes")) {
    items.push(
      <Banner key="reason" tone={status === "failed" ? "bad" : "info"}>
        {statusReason}
      </Banner>,
    );
  }

  if (items.length === 0) return null;
  return <div className="flex shrink-0 flex-col gap-2 border-b border-line px-4 py-2">{items}</div>;
}
