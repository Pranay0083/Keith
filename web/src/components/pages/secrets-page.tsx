"use client";

import { useCallback, useState, type FormEvent } from "react";
import { flushSync } from "react-dom";
import { useApp } from "@/components/app-shell";
import { RelativeTime } from "@/components/time";
import {
  Banner,
  Button,
  Field,
  Input,
  Loading,
  Page,
  PageHeader,
  Panel,
  Select,
  fieldDescribedBy,
} from "@/components/ui";
import { api } from "@/lib/api/client";
import type { Repo, Secret } from "@/lib/api/types";
import { LoadError, errorMessage, fieldError, formError, useLoad } from "./shared";

const NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
const NAME_MAX = 128;
const VALUE_MAX_BYTES = 65_536;

interface SecretsData {
  secrets: Secret[];
  /** The org's repos for the scope picker; null if they couldn't be loaded. */
  repos: Repo[] | null;
}

/** Unique per secret, and safe in an element id (names never contain "/"). */
function secretKey(s: { name: string; repo_id: string | null }): string {
  return `${s.name}/${s.repo_id ?? "org"}`;
}

/** Sorted by name, then org secrets before repo secrets (the API's order). */
function sortSecrets(list: Secret[]): Secret[] {
  return [...list].sort((a, b) => {
    if (a.name !== b.name) return a.name < b.name ? -1 : 1;
    if (a.repo_id === null) return b.repo_id === null ? 0 : -1;
    if (b.repo_id === null) return 1;
    return (a.repo_full_name ?? "").localeCompare(b.repo_full_name ?? "");
  });
}

function nameProblem(name: string): string | undefined {
  if (name === "") return undefined;
  if (name.length > NAME_MAX) return `At most ${NAME_MAX} characters.`;
  if (!NAME_RE.test(name)) return "Use letters, digits and underscores, and don't start with a digit.";
  return undefined;
}

export function SecretsPage() {
  const { org, canManage } = useApp();
  const orgId = org.id;
  const data = useLoad<SecretsData>(`secrets:${orgId}:${canManage ? "manage" : "read"}`, async () => {
    const [secrets, repos] = await Promise.all([
      api.secrets(orgId),
      // Only the add form needs repos; members never see it.
      canManage ? api.repos(orgId).catch(() => null) : Promise.resolve(null),
    ]);
    return { secrets, repos };
  });
  const { update } = data;

  const [confirming, setConfirming] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<{ key: string; message: string } | null>(null);
  const [notice, setNotice] = useState("");

  // Stable, so it runs once when the confirm step opens, not on every render.
  const focusOnMount = useCallback((el: HTMLButtonElement | null) => {
    el?.focus();
  }, []);

  function cancelDelete(key: string) {
    flushSync(() => setConfirming(null));
    document.getElementById(`delete-${key}`)?.focus();
  }

  async function confirmDelete(s: Secret) {
    const key = secretKey(s);
    setDeleting(key);
    setDeleteError(null);
    try {
      await api.deleteSecret(orgId, s.name, s.repo_id);
      update((d) => ({ ...d, secrets: d.secrets.filter((x) => secretKey(x) !== key) }));
      setConfirming(null);
      // The row is gone; keep focus in the list rather than dropping it to the page.
      document.getElementById("secrets-heading")?.focus();
      setNotice(`Deleted ${s.name}${s.repo_full_name ? ` for ${s.repo_full_name}` : ""}.`);
    } catch (e) {
      setDeleteError({ key, message: errorMessage(e) });
    } finally {
      setDeleting(null);
    }
  }

  function saved(secret: Secret) {
    // Show the change at once, then refresh quietly for anything else that moved.
    update((d) => ({
      ...d,
      secrets: sortSecrets([...d.secrets.filter((x) => secretKey(x) !== secretKey(secret)), secret]),
    }));
    api.secrets(orgId).then(
      (secrets) => update((d) => ({ ...d, secrets })),
      () => {},
    );
  }

  const secrets = data.data?.secrets;

  return (
    <Page>
      <PageHeader title="Secrets" description="Environment variables for the agent's sandbox, such as test API keys." />

      {data.loading ? (
        <Loading label="Loading secrets" />
      ) : data.error !== undefined || !data.data || !secrets ? (
        <LoadError title="Couldn't load secrets" error={data.error} onRetry={data.retry} />
      ) : (
        <div className="flex flex-col gap-5">
          <p className="max-w-2xl text-[13px] text-muted">
            Values are write-only: once saved, a value can&apos;t be read back, only replaced. Each secret is injected as
            an environment variable into every session&apos;s sandbox and redacted from logs. A repo secret applies only
            when that repo is tagged, and wins over an org-wide secret with the same name.
          </p>

          {!canManage ? (
            <Banner tone="info" title="Read-only">
              Only an owner or admin of {org.name} can add, replace or delete secrets.
            </Banner>
          ) : null}

          <Panel aria-labelledby="secrets-heading">
            <h2 id="secrets-heading" tabIndex={-1} className="border-b border-line px-3 py-2 text-[13px] font-semibold sm:px-4">
              {secrets.length === 1 ? "1 secret" : `${secrets.length} secrets`}
            </h2>
            {secrets.length === 0 ? (
              <p className="px-3 py-6 text-center text-[13px] text-muted sm:px-4">No secrets yet.</p>
            ) : (
              <ul className="divide-y divide-line">
                {secrets.map((s) => {
                  const key = secretKey(s);
                  const isConfirming = confirming === key;
                  const rowError = deleteError?.key === key ? deleteError.message : null;
                  return (
                    <li key={key} className="flex flex-wrap items-center gap-x-3 gap-y-1.5 px-3 py-2 sm:px-4">
                      <div className="min-w-0 flex-1 basis-48">
                        <p className="truncate font-mono text-[13px]">{s.name}</p>
                        <p className="text-xs text-subtle">
                          {s.repo_full_name ?? "Org-wide"} · updated <RelativeTime iso={s.updated_at} />
                        </p>
                        {rowError ? (
                          <p role="alert" className="mt-0.5 text-xs text-bad">
                            {rowError}
                          </p>
                        ) : null}
                      </div>
                      {canManage ? (
                        isConfirming ? (
                          <div className="flex items-center gap-1.5">
                            <span className="text-xs text-muted">Delete for good?</span>
                            <Button
                              variant="danger"
                              size="sm"
                              ref={focusOnMount}
                              loading={deleting === key}
                              onClick={() => confirmDelete(s)}
                            >
                              Delete<span className="sr-only"> {s.name}</span>
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              disabled={deleting === key}
                              onClick={() => cancelDelete(key)}
                            >
                              Cancel
                            </Button>
                          </div>
                        ) : (
                          <Button
                            id={`delete-${key}`}
                            variant="ghost"
                            size="sm"
                            onClick={() => {
                              setDeleteError(null);
                              setConfirming(key);
                            }}
                          >
                            Delete<span className="sr-only"> {s.name}</span>
                          </Button>
                        )
                      ) : null}
                    </li>
                  );
                })}
              </ul>
            )}
          </Panel>
          <p role="status" className="sr-only">
            {notice}
          </p>

          {canManage ? (
            <SecretForm orgId={orgId} repos={data.data.repos} existing={secrets} onSaved={saved} />
          ) : null}
        </div>
      )}
    </Page>
  );
}

function SecretForm({
  orgId,
  repos,
  existing,
  onSaved,
}: {
  orgId: string;
  repos: Repo[] | null;
  existing: Secret[];
  onSaved: (s: Secret) => void;
}) {
  const [name, setName] = useState("");
  const [value, setValue] = useState("");
  const [showValue, setShowValue] = useState(false);
  const [scope, setScope] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [savedMsg, setSavedMsg] = useState("");

  const valueTooLong = new TextEncoder().encode(value).length > VALUE_MAX_BYTES;
  const nameError =
    fieldError(error, "name") ?? nameProblem(name) ?? (submitted && name === "" ? "Name is required." : undefined);
  const valueError =
    fieldError(error, "value") ??
    (valueTooLong
      ? "At most 64 KiB."
      : submitted && value === ""
        ? "Value is required."
        : undefined);
  const scopeError = fieldError(error, "repo_id");
  const topError = formError(error, ["name", "value", "repo_id"]);

  const replaces = name !== "" && existing.some((s) => s.name === name && (s.repo_id ?? "") === scope);
  const nameHint = replaces ? "A secret with this name and scope exists; saving replaces its value." : undefined;

  function edited() {
    setError(null);
    setSavedMsg("");
  }

  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (saving) return;
    setSubmitted(true);
    setError(null);
    setSavedMsg("");
    const bad = name === "" || nameProblem(name) ? "secret-name" : value === "" || valueTooLong ? "secret-value" : null;
    if (bad) {
      document.getElementById(bad)?.focus();
      return;
    }
    setSaving(true);
    try {
      const s = await api.putSecret(orgId, name, { value, repo_id: scope || null });
      onSaved(s);
      setValue("");
      setShowValue(false);
      setSubmitted(false);
      setSavedMsg(`Saved ${s.name}${s.repo_full_name ? ` for ${s.repo_full_name}` : ""}.`);
    } catch (err) {
      setError(err);
    } finally {
      setSaving(false);
    }
  }

  return (
    <Panel aria-labelledby="secret-form-heading" className="px-3 py-3 sm:px-4">
      <h2 id="secret-form-heading" className="text-[13px] font-semibold">
        Add or replace a secret
      </h2>
      <form onSubmit={submit} noValidate className="mt-3 flex flex-col gap-3">
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Name" htmlFor="secret-name" hint={nameHint} error={nameError}>
            <Input
              id="secret-name"
              value={name}
              onChange={(e) => {
                setName(e.target.value);
                edited();
              }}
              placeholder="STRIPE_TEST_KEY"
              maxLength={NAME_MAX}
              autoComplete="off"
              autoCapitalize="off"
              spellCheck={false}
              aria-invalid={nameError ? true : undefined}
              aria-describedby={fieldDescribedBy("secret-name", nameHint, nameError)}
              className="font-mono text-[13px]"
            />
          </Field>
          <Field label="Scope" htmlFor="secret-scope" error={scopeError}>
            <Select
              id="secret-scope"
              value={scope}
              onChange={(e) => {
                setScope(e.target.value);
                edited();
              }}
              aria-invalid={scopeError ? true : undefined}
              aria-describedby={fieldDescribedBy("secret-scope", undefined, scopeError)}
            >
              <option value="">Org-wide</option>
              {(repos ?? []).map((r) => (
                <option key={r.id} value={r.id}>
                  {r.full_name}
                </option>
              ))}
            </Select>
          </Field>
        </div>
        <Field label="Value" htmlFor="secret-value" error={valueError}>
          <div className="flex gap-2">
            <Input
              id="secret-value"
              type={showValue ? "text" : "password"}
              value={value}
              onChange={(e) => {
                setValue(e.target.value);
                edited();
              }}
              autoComplete="off"
              autoCapitalize="off"
              spellCheck={false}
              aria-invalid={valueError ? true : undefined}
              aria-describedby={fieldDescribedBy("secret-value", undefined, valueError)}
              className="min-w-0 flex-1 font-mono text-[13px]"
            />
            <Button
              variant="secondary"
              onClick={() => setShowValue((v) => !v)}
              aria-controls="secret-value"
              className="w-16 shrink-0"
            >
              {showValue ? "Hide" : "Show"}
              <span className="sr-only"> value</span>
            </Button>
          </div>
        </Field>
        {repos === null ? (
          <p className="text-xs text-subtle">Couldn&apos;t load your repos, so only org-wide secrets can be added.</p>
        ) : null}
        {topError ? <Banner tone="bad">{topError}</Banner> : null}
        <div className="flex flex-wrap items-center gap-3">
          <Button type="submit" variant="primary" size="sm" loading={saving}>
            {replaces ? "Replace secret" : "Add secret"}
          </Button>
          <span role="status" className="text-[13px] text-ok">
            {savedMsg}
          </span>
        </div>
      </form>
    </Panel>
  );
}
