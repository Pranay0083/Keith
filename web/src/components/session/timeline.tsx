"use client";

import { useState } from "react";
import type { ToolStep } from "@/lib/session/reducer";
import { cx } from "@/lib/cx";
import { CheckIcon, ChevronIcon, TerminalIcon, XIcon } from "../icons";
import { formatDuration } from "../time";
import { Spinner } from "../ui";
import { describeStep, formatInput, stripAnsi } from "./format";
import { useStickToBottom } from "./hooks";

export interface TimelineProps {
  steps: ToolStep[];
  /** Selects a shell's terminal tab. */
  onShowShell(shellId: string): void;
  className?: string;
}

export function Timeline({ steps, onShowShell, className }: TimelineProps) {
  const last = steps[steps.length - 1];
  const { ref, onScroll } = useStickToBottom<HTMLDivElement>(`${steps.length}:${last?.status}`);
  const running = steps.filter((s) => s.status === "running").length;
  const failed = steps.filter((s) => s.status === "error").length;

  return (
    <div className={cx("flex min-h-0 flex-col", className)}>
      <div className="flex h-9 shrink-0 items-center justify-between border-b border-line px-3 text-xs text-subtle">
        <h2 className="font-medium text-fg">Steps</h2>
        <span>
          {steps.length} {steps.length === 1 ? "call" : "calls"}
          {running ? ` · ${running} running` : ""}
          {failed ? ` · ${failed} failed` : ""}
        </span>
      </div>
      <div ref={ref} onScroll={onScroll} className="min-h-0 flex-1 overflow-y-auto py-1">
        {steps.length === 0 ? (
          <p className="px-3 py-6 text-center text-[13px] text-subtle">Tool calls appear here as the agent works.</p>
        ) : (
          <ol aria-label="Tool calls">
            {steps.map((s) => (
              <StepRow key={s.callId} step={s} onShowShell={onShowShell} />
            ))}
          </ol>
        )}
      </div>
    </div>
  );
}

function StatusIcon({ step }: { step: ToolStep }) {
  switch (step.status) {
    case "running":
      return <Spinner className="size-3.5 text-accent-text" label="Running" />;
    case "error":
      return <XIcon className="size-3.5 text-bad" title="Failed" />;
    case "interrupted":
      return (
        <span className="grid size-3.5 place-items-center text-subtle" role="img" aria-label="Interrupted">
          <span className="h-px w-2.5 bg-current" />
        </span>
      );
    default:
      return step.exitCode ? (
        <XIcon className="size-3.5 text-warn" title={`Exited ${step.exitCode}`} />
      ) : (
        <CheckIcon className="size-3.5 text-ok" title="Done" />
      );
  }
}

function StepRow({ step, onShowShell }: { step: ToolStep; onShowShell(id: string): void }) {
  const [open, setOpen] = useState(false);
  const desc = describeStep(step);
  const detailId = `step-${step.callId}`;
  const input = formatInput(step.input);

  return (
    <li className={cx("border-b border-line/60 last:border-b-0", step.status === "running" && "bg-accent-soft/60")}>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={detailId}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-1.5 text-left hover:bg-hover"
      >
        <StatusIcon step={step} />
        <span className="shrink-0 font-mono text-xs text-fg">{step.tool}</span>
        <span className="min-w-0 flex-1 truncate font-mono text-xs text-muted" title={desc}>
          {desc}
        </span>
        {step.exitCode ? (
          <span className="shrink-0 rounded bg-warn-soft px-1 font-mono text-[11px] text-warn">exit {step.exitCode}</span>
        ) : null}
        {step.timedOut ? <span className="shrink-0 rounded bg-bad-soft px-1 text-[11px] text-bad">timed out</span> : null}
        <span className="w-12 shrink-0 text-right text-[11px] text-subtle tabular-nums">
          {step.durationMs !== null ? formatDuration(step.durationMs) : step.status === "interrupted" ? "stopped" : ""}
        </span>
        <ChevronIcon className={cx("size-3 text-subtle transition-transform", open && "rotate-90")} />
      </button>
      {open ? (
        <div id={detailId} className="space-y-1.5 px-3 pb-2 pl-8 text-xs">
          {step.summary ? (
            <p className={cx(step.status === "error" ? "text-bad" : "text-muted")}>{step.summary}</p>
          ) : null}
          {input && input !== "{}" ? (
            <pre className="max-h-48 overflow-auto rounded border border-line bg-panel p-2 font-mono text-[11px] leading-relaxed text-muted">
              {input}
            </pre>
          ) : null}
          {step.output ? (
            <pre className="max-h-48 overflow-auto rounded border border-line bg-term p-2 font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
              {stripAnsi(step.output)}
            </pre>
          ) : null}
          {step.shellId ? (
            <button
              type="button"
              onClick={() => onShowShell(step.shellId!)}
              className="inline-flex items-center gap-1 rounded text-accent-text hover:underline"
            >
              <TerminalIcon className="size-3.5" />
              Show in terminal “{step.shellId}”
            </button>
          ) : null}
        </div>
      ) : (
        <div id={detailId} hidden />
      )}
    </li>
  );
}
