"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState, useSyncExternalStore, type FormEvent, type KeyboardEvent } from "react";
import { useApp } from "@/components/app-shell";
import { ChevronIcon, LockIcon } from "@/components/icons";
import {
  Badge,
  Banner,
  Button,
  EmptyState,
  Field,
  Input,
  Kbd,
  LinkButton,
  Loading,
  Page,
  PageHeader,
  Select,
  Textarea,
  fieldDescribedBy,
} from "@/components/ui";
import { api, isApiError } from "@/lib/api/client";
import type { CreateSessionInput, LLMProviderId, Providers, Repo } from "@/lib/api/types";
import { LoadError, fieldError, formError, matchesFilter, useLoad } from "./shared";

const MAX_REPOS = 10;
const ISSUE_RE = /^https:\/\/github\.com\/([A-Za-z0-9-]+)\/([A-Za-z0-9._-]+)\/issues\/([1-9]\d*)\/?(?:[?#].*)?$/;
const ISSUE_EXAMPLE = "https://github.com/owner/repo/issues/42";

interface ParsedIssue {
  fullName: string;
  /** The URL without a trailing slash, query or fragment. */
  url: string;
}

function parseIssueUrl(raw: string): ParsedIssue | null {
  const m = ISSUE_RE.exec(raw.trim());
  if (!m) return null;
  const [, owner, repo, num] = m;
  if (!owner || !repo || !num) return null;
  return { fullName: `${owner}/${repo}`, url: `https://github.com/${owner}/${repo}/issues/${num}` };
}

function findRepo(repos: readonly Repo[], fullName: string): Repo | undefined {
  const want = fullName.toLowerCase();
  return repos.find((r) => r.full_name.toLowerCase() === want);
}

const providerNames: Record<LLMProviderId, string> = { anthropic: "Anthropic", gemini: "Gemini" };

function providerName(id: LLMProviderId): string {
  return providerNames[id] ?? id;
}

// The platform never changes, so there is nothing to subscribe to.
const noSubscribe = () => () => {};

/** True on Apple platforms, false during SSR and hydration. */
function useIsApple(): boolean {
  return useSyncExternalStore(
    noSubscribe,
    () => /Mac|iPhone|iPad|iPod/.test(navigator.userAgent),
    () => false,
  );
}

export function NewTaskPage() {
  return (
    <Suspense
      fallback={
        <Page>
          <Loading />
        </Page>
      }
    >
      <NewTask />
    </Suspense>
  );
}

function NewTask() {
  const { org } = useApp();
  const params = useSearchParams();
  const data = useLoad(`new:${org.id}`, () =>
    // The form works without the provider list (the server picks the default).
    Promise.all([api.repos(org.id), api.providers().catch(() => null)]),
  );

  const [repos, providers] = data.data ?? [undefined, null];
  const accessible = repos?.filter((r) => r.accessible) ?? [];

  return (
    <Page>
      <PageHeader
        title="New task"
        description="Describe what you want done and tag the repos the agent may change. It works on its own branch and opens a pull request."
      />
      {data.loading ? (
        <Loading />
      ) : data.error !== undefined ? (
        <LoadError title="Couldn't load your repos" error={data.error} onRetry={data.retry} />
      ) : !repos || accessible.length === 0 ? (
        <EmptyState
          title="No repos to work on"
          description="Connect GitHub and give Keith access to at least one repo before starting a task."
          action={
            <LinkButton href="/connect" variant="primary">
              Connect GitHub
            </LinkButton>
          }
        />
      ) : (
        <NewTaskForm
          // Remount when the org or the ?repo= preselection changes.
          key={`${org.id}?${params.toString()}`}
          orgId={org.id}
          repos={repos}
          accessible={accessible}
          providers={providers}
          initialRepoIds={params.getAll("repo")}
        />
      )}
    </Page>
  );
}

interface NewTaskFormProps {
  orgId: string;
  /** Every repo of the org, to tell "not connected" from "inaccessible". */
  repos: Repo[];
  accessible: Repo[];
  providers: Providers | null;
  initialRepoIds: string[];
}

function NewTaskForm({ orgId, repos, accessible, providers, initialRepoIds }: NewTaskFormProps) {
  const router = useRouter();
  const isApple = useIsApple();

  const [prompt, setPrompt] = useState("");
  const [issueUrl, setIssueUrl] = useState("");
  const [issueTouched, setIssueTouched] = useState(false);
  const [selected, setSelected] = useState<string[]>(() => {
    const ids = [...new Set(initialRepoIds)].filter((id) => accessible.some((r) => r.id === id));
    if (ids.length === 0 && accessible.length === 1 && accessible[0]) return [accessible[0].id];
    return ids.slice(0, MAX_REPOS);
  });
  const [filter, setFilter] = useState("");
  const [provider, setProvider] = useState<LLMProviderId | "">("");
  const [model, setModel] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [pending, setPending] = useState(false);
  const [serverError, setServerError] = useState<unknown>(null);

  // --- Derived validation ---

  const trimmedIssue = issueUrl.trim();
  const issue = parseIssueUrl(trimmedIssue);
  const issueRepo = issue ? findRepo(repos, issue.fullName) : undefined;

  function issueProblem(): string | undefined {
    if (trimmedIssue === "") return undefined;
    if (!issue) return issueTouched || submitted ? `Use a GitHub issue URL, like ${ISSUE_EXAMPLE}.` : undefined;
    if (!issueRepo) return `${issue.fullName} isn't connected. Add it to the GitHub App installation first.`;
    if (!issueRepo.accessible) return `Keith can no longer access ${issueRepo.full_name}.`;
    if (submitted && !selected.includes(issueRepo.id)) return `Tag ${issueRepo.full_name} to work on this issue.`;
    return undefined;
  }

  const serverIssueError = fieldError(serverError, "issue_url");
  const issueError =
    (serverIssueError === "issue_repo_not_tagged"
      ? "Tag the issue's repo to work on this issue."
      : serverIssueError) ?? issueProblem();
  const issueHint =
    issue && issueRepo?.accessible ? `Tags ${issueRepo.full_name}` : "Optional. The agent starts from the issue.";

  const promptError =
    fieldError(serverError, "prompt") ??
    (submitted && prompt.trim() === "" && trimmedIssue === "" ? "Describe the task, or paste an issue URL." : undefined);
  const repoError =
    fieldError(serverError, "repo_ids") ?? (submitted && selected.length === 0 ? "Tag at least one repo." : undefined);
  const providerError =
    fieldError(serverError, "provider") ??
    (isApiError(serverError) && serverError.code === "provider_unavailable" ? serverError.message : undefined);
  const modelError = fieldError(serverError, "model");
  const topError =
    isApiError(serverError) && serverError.code === "provider_unavailable"
      ? null
      : formError(serverError, ["prompt", "repo_ids", "issue_url", "provider", "model"]);

  const atMax = selected.length >= MAX_REPOS;
  const visibleRepos = accessible.filter((r) => matchesFilter(r.full_name, filter));
  const defaultModel =
    providers?.providers.find((p) => p.id === (provider || providers.default_provider))?.default_model ?? "";

  // --- Handlers ---

  function changed() {
    if (serverError) setServerError(null);
  }

  function onIssueChange(value: string) {
    setIssueUrl(value);
    changed();
    const parsed = parseIssueUrl(value);
    const repo = parsed ? findRepo(accessible, parsed.fullName) : undefined;
    if (repo) {
      setSelected((s) => (s.includes(repo.id) || s.length >= MAX_REPOS ? s : [repo.id, ...s]));
    }
  }

  function toggleRepo(id: string, on: boolean) {
    changed();
    setSelected((s) => (on ? (s.includes(id) || s.length >= MAX_REPOS ? s : [...s, id]) : s.filter((x) => x !== id)));
  }

  function onPromptKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      e.currentTarget.form?.requestSubmit();
    }
  }

  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (pending) return;
    setSubmitted(true);
    setServerError(null);

    // The same rules as the derived errors above, checked now so the first
    // invalid control gets focus.
    const issueBad =
      trimmedIssue !== "" && (!issue || !issueRepo || !issueRepo.accessible || !selected.includes(issueRepo.id));
    const invalid = [
      prompt.trim() === "" && trimmedIssue === "" ? "new-prompt" : null,
      issueBad ? "new-issue" : null,
      selected.length === 0 ? "new-repo-filter" : null,
    ].find((id) => id !== null);
    if (invalid) {
      document.getElementById(invalid)?.focus();
      return;
    }

    const input: CreateSessionInput = { prompt: prompt.trim(), repo_ids: selected };
    if (issue) input.issue_url = issue.url;
    if (provider) input.provider = provider;
    if (model.trim()) input.model = model.trim();

    setPending(true);
    try {
      const { id } = await api.createSession(orgId, input);
      // Stay pending while the router navigates away.
      router.push(`/sessions/${id}`);
    } catch (err) {
      setServerError(err);
      setPending(false);
      if (fieldError(err, "provider") || fieldError(err, "model") || (isApiError(err) && err.code === "provider_unavailable")) {
        setAdvancedOpen(true);
      }
    }
  }

  return (
    <form onSubmit={submit} noValidate className="flex max-w-3xl flex-col gap-5">
      <Field label="Task" htmlFor="new-prompt" error={promptError}>
        <Textarea
          id="new-prompt"
          value={prompt}
          onChange={(e) => {
            setPrompt(e.target.value);
            changed();
          }}
          onKeyDown={onPromptKeyDown}
          rows={7}
          maxLength={32000}
          placeholder="Fix the flaky date test in utils and add a regression test."
          aria-invalid={promptError ? true : undefined}
          aria-describedby={fieldDescribedBy("new-prompt", undefined, promptError)}
          className="text-[13px]"
        />
      </Field>

      <Field label="Issue URL" htmlFor="new-issue" hint={issueHint} error={issueError}>
        <Input
          id="new-issue"
          type="url"
          inputMode="url"
          value={issueUrl}
          onChange={(e) => onIssueChange(e.target.value)}
          onBlur={() => setIssueTouched(true)}
          placeholder={ISSUE_EXAMPLE}
          autoComplete="off"
          spellCheck={false}
          aria-invalid={issueError ? true : undefined}
          aria-describedby={fieldDescribedBy("new-issue", issueHint, issueError)}
          className="font-mono text-[13px]"
        />
      </Field>

      <fieldset
        className="flex min-w-0 flex-col gap-1.5"
        aria-describedby={repoError ? "new-repos-error" : "new-repos-count"}
      >
        {/* The legend must be the fieldset's first child to name the group. */}
        <legend className="flex w-full flex-wrap items-baseline justify-between gap-2">
          <span className="text-[13px] font-medium">Repos</span>
          <span id="new-repos-count" className="text-xs font-normal text-subtle" aria-live="polite">
            {selected.length} of {MAX_REPOS} selected
          </span>
        </legend>
        <label htmlFor="new-repo-filter" className="sr-only">
          Filter repos
        </label>
        <Input
          id="new-repo-filter"
          type="search"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter repos…"
          autoComplete="off"
          spellCheck={false}
        />
        <div className="max-h-64 overflow-y-auto rounded-md border border-line-strong bg-raised">
          {visibleRepos.length > 0 ? (
            <ul className="divide-y divide-line">
              {visibleRepos.map((r) => {
                const checked = selected.includes(r.id);
                const id = `new-repo-${r.id}`;
                return (
                  <li key={r.id}>
                    <label
                      htmlFor={id}
                      className="flex cursor-pointer items-center gap-2.5 px-2.5 py-1.5 text-[13px] hover:bg-hover has-disabled:cursor-not-allowed has-disabled:opacity-60"
                    >
                      <input
                        id={id}
                        type="checkbox"
                        checked={checked}
                        disabled={!checked && atMax}
                        onChange={(e) => toggleRepo(r.id, e.target.checked)}
                        className="size-3.5 shrink-0 accent-accent"
                      />
                      <span className="min-w-0 flex-1 truncate">{r.full_name}</span>
                      {r.private ? <LockIcon className="size-3.5 text-subtle" title="Private" /> : null}
                      {!r.profile ? <Badge tone="warn">Needs setup</Badge> : null}
                    </label>
                  </li>
                );
              })}
            </ul>
          ) : (
            <p className="px-2.5 py-3 text-[13px] text-muted">No repos match “{filter.trim()}”.</p>
          )}
        </div>
        {repoError ? (
          <p id="new-repos-error" className="text-xs text-bad">
            {repoError}
          </p>
        ) : atMax ? (
          <p className="text-xs text-subtle">A task can tag at most {MAX_REPOS} repos.</p>
        ) : null}
      </fieldset>

      <details
        open={advancedOpen}
        onToggle={(e) => setAdvancedOpen(e.currentTarget.open)}
        className="group rounded-md border border-line"
      >
        <summary className="flex cursor-pointer list-none items-center gap-1.5 px-3 py-2 text-[13px] font-medium select-none [&::-webkit-details-marker]:hidden">
          <ChevronIcon className="text-subtle transition-transform group-open:rotate-90" />
          Advanced
          {provider || model.trim() ? <span className="font-normal text-subtle">· customised</span> : null}
        </summary>
        <div className="grid gap-3 border-t border-line px-3 py-3 sm:grid-cols-2">
          {providers ? (
            <Field label="Provider" htmlFor="new-provider" error={providerError}>
              <Select
                id="new-provider"
                value={provider}
                onChange={(e) => {
                  const v = e.target.value;
                  setProvider(v === "" ? "" : (providers.providers.find((p) => p.id === v)?.id ?? ""));
                  changed();
                }}
                aria-invalid={providerError ? true : undefined}
                aria-describedby={fieldDescribedBy("new-provider", undefined, providerError)}
              >
                <option value="">Default ({providerName(providers.default_provider)})</option>
                {providers.providers.map((p) => (
                  <option key={p.id} value={p.id}>
                    {providerName(p.id)}
                  </option>
                ))}
              </Select>
            </Field>
          ) : (
            <p className="text-[13px] text-muted sm:col-span-2">
              Couldn&apos;t load the provider list, so the default provider will be used.
            </p>
          )}
          <Field
            label="Model"
            htmlFor="new-model"
            hint="Leave empty for the provider's default."
            error={modelError}
          >
            <Input
              id="new-model"
              value={model}
              onChange={(e) => {
                setModel(e.target.value);
                changed();
              }}
              placeholder={defaultModel}
              maxLength={100}
              autoComplete="off"
              spellCheck={false}
              aria-invalid={modelError ? true : undefined}
              aria-describedby={fieldDescribedBy("new-model", "Leave empty for the provider's default.", modelError)}
              className="font-mono text-[13px]"
            />
          </Field>
        </div>
      </details>

      {topError ? (
        <Banner tone="bad" title="Couldn't start the task">
          {topError}
        </Banner>
      ) : null}

      <div className="flex flex-wrap items-center gap-3">
        <Button type="submit" variant="primary" loading={pending}>
          Start task
        </Button>
        <span className="flex items-center gap-1 text-xs text-subtle">
          <Kbd>{isApple ? "⌘" : "Ctrl"}</Kbd>
          <Kbd>Enter</Kbd>
          <span>to start</span>
        </span>
      </div>
    </form>
  );
}
