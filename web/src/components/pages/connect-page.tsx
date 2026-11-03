"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { useApp } from "@/components/app-shell";
import { GitHubIcon } from "@/components/icons";
import { RelativeTime } from "@/components/time";
import { Badge, Banner, Button, Loading, Page, PageHeader, Panel, type Tone } from "@/components/ui";
import { api } from "@/lib/api/client";
import type { InstallationStatus } from "@/lib/api/types";
import { LoadError, errorMessage, useLoad } from "./shared";

const statusBadge: Record<InstallationStatus, { label: string; tone: Tone }> = {
  active: { label: "Active", tone: "ok" },
  suspended: { label: "Suspended", tone: "warn" },
  deleted: { label: "Uninstalled", tone: "bad" },
};

/** The message for `/connect?error=<code>` (docs/api.md, GET /github/setup). */
function setupError(code: string): { title: string; body: string } {
  switch (code) {
    case "setup_state":
      return {
        title: "The install link expired",
        body: "Install links are valid for 15 minutes and only for the person who started them. Start the install again.",
      };
    case "installation_not_verified":
      return {
        title: "GitHub didn't confirm the installation",
        body: "We couldn't verify that your GitHub account can access that installation. Install again while signed in to GitHub as the user you sign in to Keith with.",
      };
    case "installation_other_org":
      return {
        title: "That installation is linked to another organization",
        body: "This GitHub account is already connected to a different Keith organization. Switch to that organization, or uninstall the app there first.",
      };
    default:
      return {
        title: "The install didn't finish",
        body: "Something went wrong while connecting GitHub. Try the install again.",
      };
  }
}

const permissions = [
  { name: "Contents", access: "Read and write" },
  { name: "Pull requests", access: "Read and write" },
  { name: "Issues", access: "Read" },
  { name: "Metadata", access: "Read" },
] as const;

export function ConnectPage() {
  return (
    <Suspense
      fallback={
        <Page>
          <Loading />
        </Page>
      }
    >
      <Connect />
    </Suspense>
  );
}

function Connect() {
  const { org, canManage } = useApp();
  const params = useSearchParams();
  const errorCode = params.get("error");
  const pending = params.get("pending") === "1";
  const installs = useLoad(`installations:${org.id}`, () => api.installations(org.id));
  const [installing, setInstalling] = useState(false);
  const [installError, setInstallError] = useState<string | null>(null);

  async function install() {
    setInstalling(true);
    setInstallError(null);
    try {
      const { url } = await api.installUrl(org.id);
      // Leave the button busy: the browser is navigating to GitHub.
      window.location.assign(url);
    } catch (e) {
      setInstallError(errorMessage(e));
      setInstalling(false);
    }
  }

  const hasInstalls = (installs.data?.length ?? 0) > 0;
  const setup = errorCode ? setupError(errorCode) : null;

  return (
    <Page>
      <PageHeader
        title="Connect GitHub"
        description="Install the Keith GitHub App on the accounts whose repos the agent should work on."
        actions={
          !installs.loading ? (
            <Button
              variant={hasInstalls ? "secondary" : "primary"}
              onClick={install}
              loading={installing}
              disabled={!canManage}
              aria-describedby={canManage ? undefined : "install-role-note"}
            >
              <GitHubIcon />
              {hasInstalls ? "Add another account" : "Install GitHub App"}
            </Button>
          ) : null
        }
      />

      <div className="flex flex-col gap-3">
        {setup ? (
          <Banner tone="bad" title={setup.title}>
            {setup.body}
          </Banner>
        ) : null}
        {pending ? (
          <Banner tone="info" title="Waiting for approval">
            You asked a GitHub organization owner to approve the installation. Once they do, its repos show up under
            Repos.
          </Banner>
        ) : null}
        {installError ? (
          <Banner tone="bad" title="Couldn't start the install">
            {installError}
          </Banner>
        ) : null}
        {!canManage ? (
          <p id="install-role-note" className="text-[13px] text-muted">
            Only an owner or admin of <span className="font-medium text-fg">{org.name}</span> can install the GitHub
            App. Ask one of them to connect your GitHub account.
          </p>
        ) : null}
      </div>

      <div className="mt-4 grid items-start gap-4 md:grid-cols-[minmax(0,1fr)_17rem]">
        <Panel aria-labelledby="installations-heading">
          <h2 id="installations-heading" className="border-b border-line px-3 py-2 text-[13px] font-semibold sm:px-4">
            Installations
          </h2>
          {installs.loading ? (
            <div className="px-3 sm:px-4">
              <Loading label="Loading installations" />
            </div>
          ) : installs.error !== undefined ? (
            <div className="p-3">
              <LoadError title="Couldn't load installations" error={installs.error} onRetry={installs.retry} />
            </div>
          ) : installs.data && installs.data.length > 0 ? (
            <ul className="divide-y divide-line">
              {installs.data.map((i) => {
                const badge = statusBadge[i.status] ?? { label: i.status, tone: "neutral" };
                return (
                  <li key={i.id} className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-2.5 sm:px-4">
                    <GitHubIcon className="text-subtle" />
                    <span className="min-w-0 truncate text-[13px] font-medium">{i.account_login}</span>
                    <Badge tone={badge.tone}>{badge.label}</Badge>
                    <span className="ml-auto text-xs text-subtle">
                      Connected <RelativeTime iso={i.created_at} />
                    </span>
                  </li>
                );
              })}
            </ul>
          ) : (
            <p className="px-3 py-6 text-center text-[13px] text-muted sm:px-4">
              No GitHub accounts are connected yet.
              {canManage ? " Install the app to pick which repos Keith can work on." : null}
            </p>
          )}
        </Panel>

        <Panel aria-labelledby="permissions-heading" className="px-3 py-3 sm:px-4">
          <h2 id="permissions-heading" className="text-[13px] font-semibold">
            What the app can do
          </h2>
          <dl className="mt-2 divide-y divide-line text-[13px]">
            {permissions.map((p) => (
              <div key={p.name} className="flex items-center justify-between gap-3 py-1.5">
                <dt>{p.name}</dt>
                <dd className="text-muted">{p.access}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-2 border-t border-line pt-2 text-xs text-muted">
            The agent pushes only to <code className="font-mono text-fg">agent/*</code> branches and opens a pull request
            for you to review. It never pushes to your default branch.
          </p>
        </Panel>
      </div>
    </Page>
  );
}
