"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useApp } from "@/components/app-shell";
import { PlusIcon, PullRequestIcon, RepoIcon } from "@/components/icons";
import { RelativeTime } from "@/components/time";
import { Banner, Button, EmptyState, LinkButton, Loading, Page, PageHeader, Panel, StatusBadge } from "@/components/ui";
import { api } from "@/lib/api/client";
import { isTerminal } from "@/lib/api/events";
import type { Session } from "@/lib/api/types";
import { LoadError, errorMessage, useLoad } from "./shared";

/** How often the first page refreshes while a session on it is live. */
const REFRESH_MS = 15_000;

interface SessionList {
  /** The first page, replaced on each quiet refresh. */
  first: Session[];
  /** Pages loaded with "Load more". */
  rest: Session[];
  /** The cursor for the next page. */
  cursor: string | null;
  /** How many pages are loaded, so a refresh knows whose cursor to keep. */
  pages: number;
}

/** The sessions to show: newest first, each ID once (the first page is freshest). */
function rows(list: SessionList): Session[] {
  const seen = new Set<string>();
  const out: Session[] = [];
  for (const s of [...list.first, ...list.rest]) {
    if (seen.has(s.id)) continue;
    seen.add(s.id);
    out.push(s);
  }
  return out;
}

function sessionTitle(s: Session): string {
  const firstLine = s.prompt.split("\n").find((l) => l.trim() !== "")?.trim();
  return firstLine || s.issue_url || "Untitled task";
}

function prLabel(fullName: string, prNumber: number | null, prUrl: string): string {
  const n = prNumber ?? /\/pull\/(\d+)/.exec(prUrl)?.[1];
  return n ? `${fullName}#${n}` : fullName;
}

export function SessionsPage() {
  const { org } = useApp();
  const orgId = org.id;
  const list = useLoad<SessionList>(`sessions:${orgId}`, async () => {
    const p = await api.listSessions(orgId);
    return { first: p.sessions, rest: [], cursor: p.next_cursor, pages: 1 };
  });
  const [moreLoading, setMoreLoading] = useState(false);
  const [moreError, setMoreError] = useState<unknown>(null);

  const { update } = list;
  const hasLive = list.data?.first.some((s) => !isTerminal(s.status)) ?? false;

  useEffect(() => {
    if (!hasLive) return;
    const timer = window.setInterval(() => {
      if (document.visibilityState === "hidden") return;
      api.listSessions(orgId).then(
        (p) =>
          update((d) => ({
            ...d,
            first: p.sessions,
            // Past the first page, the loaded pages' cursor still applies.
            cursor: d.pages > 1 ? d.cursor : p.next_cursor,
          })),
        // A failed background refresh is not worth a banner; the next tick retries.
        () => {},
      );
    }, REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [hasLive, orgId, update]);

  async function loadMore() {
    const cursor = list.data?.cursor;
    if (!cursor || moreLoading) return;
    setMoreLoading(true);
    setMoreError(null);
    try {
      const p = await api.listSessions(orgId, cursor);
      update((d) => ({ ...d, rest: [...d.rest, ...p.sessions], cursor: p.next_cursor, pages: d.pages + 1 }));
    } catch (e) {
      setMoreError(e);
    } finally {
      setMoreLoading(false);
    }
  }

  const sessions = list.data ? rows(list.data) : [];

  return (
    <Page>
      <PageHeader
        title="Sessions"
        description="Every task in this organization, newest first."
        actions={
          sessions.length > 0 ? (
            <LinkButton href="/new" variant="primary" size="sm">
              <PlusIcon />
              New task
            </LinkButton>
          ) : null
        }
      />

      {list.loading ? (
        <Loading label="Loading sessions" />
      ) : list.error !== undefined ? (
        <LoadError title="Couldn't load sessions" error={list.error} onRetry={list.retry} />
      ) : sessions.length === 0 ? (
        <EmptyState
          title="No sessions yet"
          description="Describe a task and tag a repo. The agent works on it live, here, until it opens a pull request."
          action={
            <LinkButton href="/new" variant="primary">
              <PlusIcon />
              New task
            </LinkButton>
          }
        />
      ) : (
        <>
          <Panel>
            <ul className="divide-y divide-line">
              {sessions.map((s) => (
                <SessionRow key={s.id} session={s} />
              ))}
            </ul>
          </Panel>
          {moreError ? (
            <Banner tone="bad" className="mt-3">
              {errorMessage(moreError)}
            </Banner>
          ) : null}
          {list.data?.cursor ? (
            <div className="mt-3 flex justify-center">
              <Button onClick={loadMore} loading={moreLoading}>
                Load more
              </Button>
            </div>
          ) : null}
        </>
      )}
    </Page>
  );
}

function SessionRow({ session: s }: { session: Session }) {
  const prs = s.repos.flatMap((r) => (r.pr_url ? [{ key: r.repo_id, url: r.pr_url, label: prLabel(r.full_name, r.pr_number, r.pr_url) }] : []));
  const author = s.created_by;
  return (
    // The title link covers the whole row; the PR links sit above it.
    <li className="relative flex flex-col gap-1 px-3 py-2.5 hover:bg-hover sm:px-4">
      <div className="flex min-w-0 items-center gap-2.5">
        <StatusBadge status={s.status} className="shrink-0" />
        <Link
          href={`/sessions/${s.id}`}
          className="min-w-0 flex-1 truncate text-[13px] font-medium after:absolute after:inset-0"
        >
          {sessionTitle(s)}
        </Link>
        <RelativeTime iso={s.created_at} className="shrink-0 text-xs text-subtle" />
      </div>
      <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 text-xs text-subtle">
        {s.repos.length > 0 ? (
          <span className="flex min-w-0 items-center gap-1">
            <RepoIcon className="size-3.5" />
            <span className="truncate">{s.repos.map((r) => r.full_name).join(", ")}</span>
          </span>
        ) : null}
        {prs.map((pr) => (
          <a
            key={pr.key}
            href={pr.url}
            target="_blank"
            rel="noreferrer"
            className="relative z-10 flex items-center gap-1 text-accent-text hover:underline"
          >
            <PullRequestIcon className="size-3.5" />
            {pr.label}
            <span className="sr-only"> (opens in a new tab)</span>
          </a>
        ))}
        {author ? (
          <span className="ml-auto flex items-center gap-1.5">
            {author.avatar_url ? (
              // eslint-disable-next-line @next/next/no-img-element -- remote avatar, no optimisation needed
              <img src={author.avatar_url} alt="" className="size-4 rounded-full bg-hover" />
            ) : null}
            <span className={author.avatar_url ? "sr-only sm:not-sr-only" : undefined}>{author.login}</span>
          </span>
        ) : null}
      </div>
    </li>
  );
}
