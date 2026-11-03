import type { PullRequest } from "@/lib/session/reducer";
import { ExternalIcon, PullRequestIcon } from "../icons";
import { buttonClass } from "../ui";

export function PRCard({ pr }: { pr: PullRequest }) {
  return (
    <article className="flex items-center gap-3 rounded-lg border border-line bg-ok-soft px-3 py-2">
      <PullRequestIcon className="text-ok" />
      <div className="min-w-0 flex-1">
        <p className="text-xs text-muted">
          Pull request opened · <span className="font-mono">{pr.repo}#{pr.number}</span>
        </p>
        <p className="truncate text-[13px] font-medium text-fg" title={pr.title || undefined}>
          {pr.title || `${pr.repo}#${pr.number}`}
        </p>
        {pr.branch ? <p className="truncate font-mono text-[11px] text-subtle">{pr.branch}</p> : null}
      </div>
      <a href={pr.url} target="_blank" rel="noreferrer" className={buttonClass("secondary", "sm", "shrink-0")}>
        Open on GitHub
        <ExternalIcon className="size-3.5" />
        <span className="sr-only">(opens in a new tab)</span>
      </a>
    </article>
  );
}
