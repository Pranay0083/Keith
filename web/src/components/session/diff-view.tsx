"use client";

// The session's diff, per repo, from GET /api/sessions/{id}/diff. It
// refetches when `version` moves (every file.changed), debounced so a burst
// of edits costs one request, but never waiting more than 2 s.

import "react-diff-view/style/index.css";
import { useEffect, useMemo, useRef, useState } from "react";
import { Decoration, Diff, Hunk, isDelete, isInsert, parseDiff, type FileData } from "react-diff-view";
import { api } from "@/lib/api/client";
import type { RepoDiff, SessionDiff } from "@/lib/api/types";
import type { FileChange } from "@/lib/session/reducer";
import { cx } from "@/lib/cx";
import { ChevronIcon, LockIcon } from "../icons";
import { RelativeTime } from "../time";
import { Badge, Banner, Button, Loading, type Tone } from "../ui";
import { fileAnchor, repoRelativePath } from "./format";

const DEBOUNCE_MS = 500;
const MAX_WAIT_MS = 2_000;
/** Files beyond this many start collapsed, to keep huge diffs responsive. */
const OPEN_FILES = 25;

export interface DiffViewProps {
  sessionId: string;
  version: number;
  files: FileChange[];
  /** The agent is running, so the view is read-only and may change. */
  live: boolean;
}

export function DiffView({ sessionId, version, files, live }: DiffViewProps) {
  const [data, setData] = useState<SessionDiff | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const firstPending = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    const now = Date.now();
    if (firstPending.current === null) firstPending.current = now;
    const waited = now - firstPending.current;
    const delay = version === 0 && reload === 0 ? 0 : Math.max(0, Math.min(DEBOUNCE_MS, MAX_WAIT_MS - waited));
    const t = setTimeout(() => {
      firstPending.current = null;
      api.diff(sessionId).then(
        (d) => {
          if (cancelled) return;
          setData(d);
          setError(null);
        },
        (e: unknown) => {
          if (!cancelled) setError(e instanceof Error ? e.message : "Couldn't load the diff.");
        },
      );
    }, delay);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [sessionId, version, reload]);

  const changed = files.length;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-9 shrink-0 items-center gap-2 border-b border-line px-3 text-xs text-subtle">
        <span className="font-medium text-fg">
          {changed} {changed === 1 ? "file" : "files"} changed
        </span>
        {live ? (
          <span className="inline-flex items-center gap-1" title="You can't edit files while the agent is running.">
            <LockIcon className="size-3.5" />
            Read-only while the agent runs
          </span>
        ) : null}
        <Button variant="ghost" size="sm" className="ml-auto" onClick={() => setReload((n) => n + 1)}>
          Refresh
        </Button>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        {error ? (
          <div className="p-3">
            <Banner
              tone="bad"
              title="Couldn't load the diff"
              actions={
                <Button size="sm" onClick={() => setReload((n) => n + 1)}>
                  Retry
                </Button>
              }
            >
              {error}
            </Banner>
          </div>
        ) : null}
        {files.length > 0 ? <ChangedFiles files={files} /> : null}
        {data === null && !error ? (
          <div className="px-3">
            <Loading label="Loading diff" />
          </div>
        ) : null}
        {data?.repos.map((r) => <RepoDiffView key={r.full_name} diff={r} />)}
      </div>
    </div>
  );
}

const actionTone: Record<FileChange["action"], Tone> = { created: "ok", modified: "accent", deleted: "bad" };

function ChangedFiles({ files }: { files: FileChange[] }) {
  return (
    <nav aria-label="Changed files" className="border-b border-line px-3 py-2">
      <ul className="flex flex-col gap-0.5">
        {files.map((f) => {
          const rel = repoRelativePath(f.path, f.repo);
          return (
            <li key={`${f.repo}:${f.path}`} className="flex items-center gap-2 text-xs">
              <Badge tone={actionTone[f.action]} className="w-16 justify-center">
                {f.action}
              </Badge>
              <a
                href={`#${fileAnchor(f.repo ?? "", rel)}`}
                className="min-w-0 truncate font-mono text-muted hover:text-fg hover:underline"
              >
                {f.repo ? <span className="text-subtle">{f.repo}/</span> : null}
                {rel}
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

function RepoDiffView({ diff }: { diff: RepoDiff }) {
  const files = useMemo<FileData[]>(() => {
    if (!diff.diff) return [];
    try {
      return parseDiff(diff.diff, { nearbySequences: "zip" });
    } catch {
      return [];
    }
  }, [diff.diff]);

  return (
    <section aria-label={`Changes in ${diff.full_name}`} className="border-b border-line">
      <header className="sticky top-0 z-10 flex items-center gap-2 border-b border-line bg-panel px-3 py-1.5 text-xs">
        <span className="font-mono font-medium text-fg">{diff.full_name}</span>
        {diff.updated_at ? (
          <span className="text-subtle">
            updated <RelativeTime iso={diff.updated_at} />
          </span>
        ) : null}
      </header>
      {diff.truncated ? (
        <div className="px-3 pt-2">
          <Banner tone="warn">The diff is over 1 MiB, so only the start is shown. The PR has the full change.</Banner>
        </div>
      ) : null}
      {files.length === 0 ? (
        <p className="px-3 py-4 text-[13px] text-subtle">{diff.diff ? "This diff couldn't be displayed." : "No changes yet."}</p>
      ) : (
        files.map((f, i) => <FileDiff key={`${f.oldPath}->${f.newPath}`} repo={diff.full_name} file={f} defaultOpen={i < OPEN_FILES} />)
      )}
    </section>
  );
}

const typeLabel: Record<FileData["type"], { label: string; tone: Tone }> = {
  add: { label: "added", tone: "ok" },
  delete: { label: "deleted", tone: "bad" },
  modify: { label: "modified", tone: "accent" },
  rename: { label: "renamed", tone: "wait" },
  copy: { label: "copied", tone: "wait" },
};

function FileDiff({ repo, file, defaultOpen }: { repo: string; file: FileData; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const path = file.type === "delete" ? file.oldPath : file.newPath;
  const counts = useMemo(() => {
    let add = 0;
    let del = 0;
    for (const h of file.hunks) {
      for (const c of h.changes) {
        if (isInsert(c)) add++;
        else if (isDelete(c)) del++;
      }
    }
    return { add, del };
  }, [file.hunks]);
  const t = typeLabel[file.type];
  const bodyId = `${fileAnchor(repo, path)}-body`;

  return (
    <div id={fileAnchor(repo, path)} className="scroll-mt-9 border-b border-line last:border-b-0">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={bodyId}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs hover:bg-hover"
      >
        <ChevronIcon className={cx("size-3 text-subtle transition-transform", open && "rotate-90")} />
        <span className="min-w-0 flex-1 truncate font-mono text-fg">
          {file.type === "rename" ? `${file.oldPath} → ${file.newPath}` : path}
        </span>
        <Badge tone={t.tone}>{t.label}</Badge>
        <span className="font-mono text-ok tabular-nums">+{counts.add}</span>
        <span className="font-mono text-bad tabular-nums">−{counts.del}</span>
      </button>
      <div id={bodyId} hidden={!open} className="overflow-x-auto">
        {open && file.hunks.length > 0 ? (
          <Diff viewType="unified" diffType={file.type} hunks={file.hunks} className="keith-diff">
            {(hunks) =>
              hunks.flatMap((h) => [
                <Decoration key={`d-${h.content}`}>
                  <span>{h.content}</span>
                </Decoration>,
                <Hunk key={`h-${h.content}`} hunk={h} />,
              ])
            }
          </Diff>
        ) : open ? (
          <p className="px-3 py-2 text-xs text-subtle">No content changes (mode or binary change).</p>
        ) : null}
      </div>
    </div>
  );
}
