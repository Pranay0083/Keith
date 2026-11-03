"use client";

// The right-hand workspace: one terminal tab per shell, the diff, and (on
// narrow screens, where the timeline has no column of its own) the steps.
// It follows the agent by default: the active tab is the shell the agent
// last used, until the user picks a tab themselves.

import { useRef, type KeyboardEvent } from "react";
import type { SessionState } from "@/lib/session/reducer";
import { cx } from "@/lib/cx";
import { DiffIcon, StepsIcon, TerminalIcon } from "../icons";
import { DiffView } from "./diff-view";
import { TerminalView } from "./terminal";
import { Timeline } from "./timeline";

export type TabId = "steps" | "diff" | `shell:${string}`;

export interface WorkspaceProps {
  state: SessionState;
  /** Whether the steps live in this workspace (no timeline column). */
  withSteps: boolean;
  active: TabId;
  follow: boolean;
  onSelect(tab: TabId): void;
  onFollow(): void;
  live: boolean;
}

/** The tab the workspace shows when following the agent. */
export function followTab(state: SessionState): TabId | null {
  for (let i = state.steps.length - 1; i >= 0; i--) {
    const id = state.steps[i]!.shellId;
    if (id && state.shells.some((s) => s.id === id)) return `shell:${id}`;
  }
  const last = state.shells[state.shells.length - 1];
  return last ? `shell:${last.id}` : null;
}

export function Workspace({ state, withSteps, active, follow, onSelect, onFollow, live }: WorkspaceProps) {
  const tabs: { id: TabId; label: string; icon: typeof TerminalIcon; badge?: number }[] = [
    ...(withSteps ? [{ id: "steps" as const, label: "Steps", icon: StepsIcon, badge: state.steps.length }] : []),
    ...state.shells.map((s) => ({ id: `shell:${s.id}` as const, label: s.id, icon: TerminalIcon })),
    { id: "diff", label: "Diff", icon: DiffIcon, badge: state.files.length },
  ];
  const tabRefs = useRef(new Map<TabId, HTMLButtonElement>());

  function onKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    const i = tabs.findIndex((t) => t.id === active);
    let next = -1;
    if (e.key === "ArrowRight") next = (i + 1) % tabs.length;
    else if (e.key === "ArrowLeft") next = (i - 1 + tabs.length) % tabs.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = tabs.length - 1;
    if (next < 0) return;
    e.preventDefault();
    const tab = tabs[next]!;
    onSelect(tab.id);
    tabRefs.current.get(tab.id)?.focus();
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-9 shrink-0 items-stretch border-b border-line bg-panel">
        <div
          role="tablist"
          aria-label="Workspace"
          onKeyDown={onKeyDown}
          className="flex min-w-0 flex-1 items-stretch overflow-x-auto"
        >
          {tabs.map(({ id, label, icon: Icon, badge }) => {
            const selected = id === active;
            return (
              <button
                key={id}
                ref={(el) => {
                  if (el) tabRefs.current.set(id, el);
                  else tabRefs.current.delete(id);
                }}
                type="button"
                role="tab"
                id={`tab-${id}`}
                aria-selected={selected}
                aria-controls={`panel-${id}`}
                tabIndex={selected ? 0 : -1}
                onClick={() => onSelect(id)}
                className={cx(
                  "relative flex shrink-0 items-center gap-1.5 border-r border-line px-3 text-xs whitespace-nowrap",
                  selected ? "bg-canvas font-medium text-fg" : "text-muted hover:bg-hover hover:text-fg",
                )}
              >
                <Icon className="size-3.5 text-subtle" />
                <span className={id.startsWith("shell:") ? "font-mono" : undefined}>{label}</span>
                {badge ? <span className="rounded bg-hover px-1 text-[10px] text-subtle tabular-nums">{badge}</span> : null}
                {selected ? <span className="absolute inset-x-0 -bottom-px h-0.5 bg-accent" aria-hidden /> : null}
              </button>
            );
          })}
        </div>
        <button
          type="button"
          aria-pressed={follow}
          onClick={onFollow}
          title="Switch tabs to follow what the agent is doing"
          className={cx(
            "shrink-0 border-l border-line px-3 text-xs",
            follow ? "text-accent-text" : "text-subtle hover:bg-hover hover:text-fg",
          )}
        >
          {follow ? "Following" : "Follow agent"}
        </button>
      </div>

      <div className="relative min-h-0 flex-1">
        {withSteps ? (
          <div
            role="tabpanel"
            id="panel-steps"
            aria-labelledby="tab-steps"
            hidden={active !== "steps"}
            className="absolute inset-0 flex flex-col"
          >
            <Timeline steps={state.steps} onShowShell={(id) => onSelect(`shell:${id}`)} className="h-full" />
          </div>
        ) : null}
        {state.shells.map((s) => {
          const id = `shell:${s.id}` as const;
          return (
            // Every terminal stays mounted so it keeps its scrollback.
            <div
              key={id}
              role="tabpanel"
              id={`panel-${id}`}
              aria-labelledby={`tab-${id}`}
              hidden={active !== id}
              className="absolute inset-0 bg-term"
            >
              <TerminalView output={s.output} dropped={s.dropped} visible={active === id} label={`Shell ${s.id} output`} />
            </div>
          );
        })}
        <div
          role="tabpanel"
          id="panel-diff"
          aria-labelledby="tab-diff"
          hidden={active !== "diff"}
          className="absolute inset-0"
        >
          <DiffView sessionId={state.sessionId} version={state.diffVersion} files={state.files} live={live} />
        </div>
        {state.shells.length === 0 && active.startsWith("shell:") ? (
          <p className="p-6 text-center text-[13px] text-subtle">No shells yet.</p>
        ) : null}
      </div>
    </div>
  );
}
