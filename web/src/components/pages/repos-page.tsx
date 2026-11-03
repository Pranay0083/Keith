"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useState, type FormEvent } from "react";
import { useApp } from "@/components/app-shell";
import { CheckIcon, ChevronIcon, LockIcon } from "@/components/icons";
import { RelativeTime } from "@/components/time";
import {
  Badge,
  Banner,
  Button,
  EmptyState,
  Field,
  Input,
  LinkButton,
  Loading,
  Page,
  PageHeader,
  Panel,
  Textarea,
  fieldDescribedBy,
} from "@/components/ui";
import { api } from "@/lib/api/client";
import type { Repo, RepoProfileInput, RepoProfileSummary } from "@/lib/api/types";
import { cx } from "@/lib/cx";
import { LoadError, fieldError, formError, matchesFilter, useLoad } from "./shared";

/** Show the filter once the list is longer than this. */
const FILTER_AFTER = 8;

export function ReposPage() {
  return (
    <Suspense
      fallback={
        <Page>
          <Loading />
        </Page>
      }
    >
      <Repos />
    </Suspense>
  );
}

function Repos() {
  const { org } = useApp();
  const installed = useSearchParams().get("installed") === "1";
  const repos = useLoad(`repos:${org.id}`, () => api.repos(org.id));
  const [filter, setFilter] = useState("");
  const [open, setOpen] = useState<ReadonlySet<string>>(() => new Set());

  function toggle(id: string) {
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function saved(id: string, profile: RepoProfileSummary) {
    repos.update((list) => list.map((r) => (r.id === id ? { ...r, profile } : r)));
  }

  const list = repos.data;
  const visible = list?.filter((r) => matchesFilter(r.full_name, filter));

  return (
    <Page>
      <PageHeader
        title="Repos"
        description="The repos the GitHub App can see. Give each a profile so the agent knows how to install, build and test it."
        actions={
          list && list.length > 0 ? (
            <LinkButton href="/connect" size="sm">
              Add repos
            </LinkButton>
          ) : null
        }
      />

      {installed ? (
        <Banner tone="ok" title="GitHub App installed" className="mb-4">
          Your repos are synced below. Set up a profile for each one before starting a task.
        </Banner>
      ) : null}

      {repos.loading ? (
        <Loading label="Loading repos" />
      ) : repos.error !== undefined ? (
        <LoadError title="Couldn't load repos" error={repos.error} onRetry={repos.retry} />
      ) : !list || list.length === 0 ? (
        <EmptyState
          title="No repos yet"
          description="Install the GitHub App on a GitHub account and choose which repos Keith can work on."
          action={
            <LinkButton href="/connect" variant="primary">
              Connect GitHub
            </LinkButton>
          }
        />
      ) : (
        <>
          {list.length > FILTER_AFTER ? (
            <div className="mb-3 max-w-xs">
              <label htmlFor="repo-filter" className="sr-only">
                Filter repos
              </label>
              <Input
                id="repo-filter"
                type="search"
                placeholder="Filter repos…"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                autoComplete="off"
                spellCheck={false}
              />
            </div>
          ) : null}
          <Panel>
            {visible && visible.length > 0 ? (
              <ul className="divide-y divide-line">
                {visible.map((r) => (
                  <RepoRow
                    key={r.id}
                    repo={r}
                    open={open.has(r.id)}
                    onToggle={() => toggle(r.id)}
                    onSaved={(p) => saved(r.id, p)}
                  />
                ))}
              </ul>
            ) : (
              <p className="px-4 py-6 text-center text-[13px] text-muted" role="status">
                No repos match “{filter.trim()}”.
              </p>
            )}
          </Panel>
        </>
      )}
    </Page>
  );
}

function RepoRow({
  repo,
  open,
  onToggle,
  onSaved,
}: {
  repo: Repo;
  open: boolean;
  onToggle: () => void;
  onSaved: (p: RepoProfileSummary) => void;
}) {
  const formId = `profile-${repo.id}`;
  return (
    <li>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 px-3 py-2.5 sm:px-4">
        <div className="min-w-0 flex-1 basis-56">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="min-w-0 truncate text-[13px] font-medium">{repo.full_name}</span>
            {repo.private ? (
              <Badge>
                <LockIcon className="size-3" />
                Private
              </Badge>
            ) : null}
            {!repo.accessible ? <Badge tone="bad">Inaccessible</Badge> : null}
          </div>
          <p className="mt-0.5 text-xs text-subtle">
            Default branch <span className="font-mono text-muted">{repo.default_branch}</span>
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {repo.profile ? <Badge tone="ok">Profile set</Badge> : <Badge tone="warn">Needs setup</Badge>}
          {repo.accessible ? (
            <LinkButton href={`/new?repo=${encodeURIComponent(repo.id)}`} size="sm">
              Start task<span className="sr-only"> in {repo.full_name}</span>
            </LinkButton>
          ) : null}
          <Button variant="ghost" size="sm" aria-expanded={open} aria-controls={formId} onClick={onToggle}>
            <ChevronIcon className={cx("transition-transform", open && "rotate-90")} />
            Profile<span className="sr-only"> for {repo.full_name}</span>
          </Button>
        </div>
      </div>
      {open ? (
        <div id={formId} className="border-t border-line bg-panel px-3 py-3 sm:px-4">
          <ProfileForm repo={repo} onSaved={onSaved} />
        </div>
      ) : null}
    </li>
  );
}

const commandFields = [
  { key: "install_cmd", label: "Install command", placeholder: "npm ci" },
  { key: "build_cmd", label: "Build command", placeholder: "npm run build" },
  { key: "test_cmd", label: "Test command", placeholder: "npm test" },
  { key: "run_cmd", label: "Run command", placeholder: "npm run dev" },
] as const;

const profileFields = ["install_cmd", "build_cmd", "test_cmd", "run_cmd", "knowledge"] as const;

const KNOWLEDGE_HINT = "Notes the agent gets in its prompt: conventions, gotchas, how to run things.";

function ProfileForm({ repo, onSaved }: { repo: Repo; onSaved: (p: RepoProfileSummary) => void }) {
  const [values, setValues] = useState<RepoProfileInput>(() => ({
    install_cmd: repo.profile?.install_cmd ?? "",
    build_cmd: repo.profile?.build_cmd ?? "",
    test_cmd: repo.profile?.test_cmd ?? "",
    run_cmd: repo.profile?.run_cmd ?? "",
    knowledge: repo.profile?.knowledge ?? "",
  }));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [saved, setSaved] = useState(false);
  const disabled = !repo.accessible;

  function set(key: keyof RepoProfileInput, value: string) {
    setValues((v) => ({ ...v, [key]: value }));
    setSaved(false);
  }

  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (disabled || saving) return;
    setSaving(true);
    setError(null);
    setSaved(false);
    try {
      const p = await api.putProfile(repo.id, values);
      onSaved({
        install_cmd: p.install_cmd,
        build_cmd: p.build_cmd,
        test_cmd: p.test_cmd,
        run_cmd: p.run_cmd,
        knowledge: p.knowledge,
        updated_at: p.updated_at ?? new Date().toISOString(),
      });
      setSaved(true);
    } catch (err) {
      setError(err);
    } finally {
      setSaving(false);
    }
  }

  const topError = formError(error, profileFields);
  const knowledgeError = fieldError(error, "knowledge");

  return (
    <form onSubmit={submit} aria-label={`Profile for ${repo.full_name}`} noValidate>
      {disabled ? (
        <p className="mb-3 text-[13px] text-muted">
          Keith can no longer access this repo, so its profile can&apos;t be edited. Add it back to the GitHub App
          installation on{" "}
          <Link href="/connect" className="text-accent-text hover:underline">
            Connect GitHub
          </Link>
          .
        </p>
      ) : null}
      <fieldset disabled={disabled} className="flex min-w-0 flex-col gap-3">
        <div className="grid gap-3 sm:grid-cols-2">
          {commandFields.map((f) => {
            const id = `${repo.id}-${f.key}`;
            const err = fieldError(error, f.key);
            return (
              <Field key={f.key} label={f.label} htmlFor={id} error={err}>
                <Input
                  id={id}
                  value={values[f.key]}
                  onChange={(e) => set(f.key, e.target.value)}
                  placeholder={f.placeholder}
                  maxLength={2000}
                  autoComplete="off"
                  autoCapitalize="off"
                  spellCheck={false}
                  aria-invalid={err ? true : undefined}
                  aria-describedby={fieldDescribedBy(id, undefined, err)}
                  className="font-mono text-[13px]"
                />
              </Field>
            );
          })}
        </div>
        <Field label="Knowledge" htmlFor={`${repo.id}-knowledge`} hint={KNOWLEDGE_HINT} error={knowledgeError}>
          <Textarea
            id={`${repo.id}-knowledge`}
            value={values.knowledge}
            onChange={(e) => set("knowledge", e.target.value)}
            rows={5}
            maxLength={20000}
            aria-invalid={knowledgeError ? true : undefined}
            aria-describedby={fieldDescribedBy(`${repo.id}-knowledge`, KNOWLEDGE_HINT, knowledgeError)}
            className="text-[13px]"
          />
        </Field>
        {topError ? <Banner tone="bad">{topError}</Banner> : null}
        <div className="flex flex-wrap items-center gap-3">
          <Button type="submit" variant="primary" size="sm" loading={saving}>
            Save profile
          </Button>
          <span role="status" className="flex items-center gap-1 text-[13px] text-ok">
            {saved ? (
              <>
                <CheckIcon />
                Saved
              </>
            ) : null}
          </span>
          {repo.profile && !saved ? (
            <span className="text-xs text-subtle">
              Updated <RelativeTime iso={repo.profile.updated_at} />
            </span>
          ) : null}
        </div>
      </fieldset>
    </form>
  );
}
