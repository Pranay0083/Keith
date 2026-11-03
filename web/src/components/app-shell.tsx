"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, isApiError, isMock } from "@/lib/api/client";
import type { Me, Org } from "@/lib/api/types";
import { cx } from "@/lib/cx";
import { Badge, Banner, Button, LinkButton, Loading, Select } from "./ui";
import { KeyIcon, ListIcon, LogOutIcon, PlugIcon, PlusIcon, RepoIcon } from "./icons";

const ORG_KEY = "keith.orgId";

export interface AppContextValue {
  me: Me;
  /** The org every org-scoped page works in. */
  org: Org;
  setOrgId(id: string): void;
  /** True when the user may manage the org (owner or admin). */
  canManage: boolean;
}

const AppContext = createContext<AppContextValue | null>(null);

/** The signed-in user and the selected org. Only valid inside AppShell. */
export function useApp(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used inside <AppShell>");
  return ctx;
}

/** The signed-in user, or null outside AppShell. */
export function useMe(): Me | null {
  return useContext(AppContext)?.me ?? null;
}

function readStoredOrg(): string | null {
  try {
    return window.localStorage.getItem(ORG_KEY);
  } catch {
    return null;
  }
}

function storeOrg(id: string) {
  try {
    window.localStorage.setItem(ORG_KEY, id);
  } catch {
    // Storage can be unavailable (private mode); the choice just won't stick.
  }
}

const nav = [
  { href: "/sessions", label: "Sessions", icon: ListIcon },
  { href: "/repos", label: "Repos", icon: RepoIcon },
  { href: "/connect", label: "Connect GitHub", icon: PlugIcon },
  { href: "/settings/secrets", label: "Secrets", icon: KeyIcon },
] as const;

export function AppShell({ children }: { children: ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [me, setMe] = useState<Me | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [orgId, setOrgIdState] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let live = true;
    api.me().then(
      (m) => {
        if (!live) return;
        setMe(m);
        setError(null);
        const stored = readStoredOrg();
        setOrgIdState(m.orgs.some((o) => o.id === stored) ? stored : (m.orgs[0]?.id ?? null));
      },
      (e: unknown) => {
        if (!live) return;
        if (isApiError(e) && e.status === 401) {
          router.replace(`/login?next=${encodeURIComponent(window.location.pathname + window.location.search)}`);
          return;
        }
        setError(e instanceof Error ? e.message : "Something went wrong.");
      },
    );
    return () => {
      live = false;
    };
  }, [router, attempt]);

  const setOrgId = useCallback((id: string) => {
    setOrgIdState(id);
    storeOrg(id);
  }, []);

  const value = useMemo<AppContextValue | null>(() => {
    if (!me) return null;
    const org = me.orgs.find((o) => o.id === orgId) ?? me.orgs[0];
    if (!org) return null;
    return { me, org, setOrgId, canManage: org.role === "owner" || org.role === "admin" };
  }, [me, orgId, setOrgId]);

  async function signOut() {
    try {
      await api.logout();
    } finally {
      router.replace("/login");
    }
  }

  return (
    <div className="flex h-dvh flex-col md:flex-row">
      <a
        href="#main"
        className="sr-only z-50 rounded-md bg-accent px-3 py-1.5 text-accent-fg focus:not-sr-only focus:absolute focus:top-2 focus:left-2"
      >
        Skip to content
      </a>

      <aside className="flex shrink-0 flex-col border-b border-line bg-panel md:w-56 md:border-r md:border-b-0">
        <div className="flex items-center justify-between gap-2 px-3 pt-3 pb-2 md:px-4">
          <Link href="/sessions" className="flex items-center gap-2 rounded-md text-[15px] font-semibold tracking-tight">
            <span className="grid size-6 place-items-center rounded-md bg-fg font-mono text-xs text-canvas" aria-hidden>
              K
            </span>
            Keith
          </Link>
          <div className="flex items-center gap-1">
            {isMock ? (
              <Badge tone="warn" className="font-mono">
                mock
              </Badge>
            ) : null}
            {value ? (
              <Button variant="ghost" size="sm" onClick={signOut} aria-label="Sign out" title="Sign out" className="md:hidden">
                <LogOutIcon />
              </Button>
            ) : null}
          </div>
        </div>

        <nav aria-label="Main" className="flex gap-1 overflow-x-auto px-2 pb-2 md:flex-col md:overflow-visible md:px-3">
          <LinkButton href="/new" variant="primary" size="sm" className="shrink-0 md:mb-2 md:w-full md:justify-start">
            <PlusIcon />
            New task
          </LinkButton>
          {nav.map(({ href, label, icon: Icon }) => {
            const active = pathname === href || pathname.startsWith(`${href}/`);
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={cx(
                  "flex h-7 shrink-0 items-center gap-2 rounded-md px-2 text-[13px] whitespace-nowrap transition-colors",
                  active ? "bg-hover font-medium text-fg" : "text-muted hover:bg-hover hover:text-fg",
                )}
              >
                <Icon className="text-subtle" />
                {label}
              </Link>
            );
          })}
        </nav>

        {value ? (
          <div className="mt-auto hidden flex-col gap-2 border-t border-line p-3 md:flex">
            {value.me.orgs.length > 1 ? (
              <div>
                <label htmlFor="org-switch" className="mb-1 block text-xs text-subtle">
                  Organization
                </label>
                <Select id="org-switch" value={value.org.id} onChange={(e) => setOrgId(e.target.value)}>
                  {value.me.orgs.map((o) => (
                    <option key={o.id} value={o.id}>
                      {o.name}
                    </option>
                  ))}
                </Select>
              </div>
            ) : (
              <p className="truncate text-xs text-subtle" title={value.org.name}>
                {value.org.name} · {value.org.role}
              </p>
            )}
            <div className="flex items-center gap-2">
              {value.me.user.avatar_url ? (
                // eslint-disable-next-line @next/next/no-img-element -- remote avatar, no optimisation needed
                <img src={value.me.user.avatar_url} alt="" className="size-6 rounded-full bg-hover" />
              ) : (
                <span className="size-6 rounded-full bg-hover" aria-hidden />
              )}
              <span className="min-w-0 flex-1 truncate text-[13px]">{value.me.user.login}</span>
              <Button variant="ghost" size="sm" onClick={signOut} aria-label="Sign out" title="Sign out">
                <LogOutIcon />
              </Button>
            </div>
          </div>
        ) : null}
      </aside>

      <main id="main" tabIndex={-1} className="min-h-0 min-w-0 flex-1 overflow-auto focus:outline-none">
        {value ? (
          <AppContext.Provider value={value}>{children}</AppContext.Provider>
        ) : error ? (
          <div className="mx-auto max-w-xl p-6">
            <Banner
              tone="bad"
              title="Couldn't load your account"
              actions={
                <Button size="sm" onClick={() => setAttempt((n) => n + 1)}>
                  Retry
                </Button>
              }
            >
              {error}
            </Banner>
          </div>
        ) : (
          <div className="px-6">
            <Loading />
          </div>
        )}
      </main>
    </div>
  );
}
