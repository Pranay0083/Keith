"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { isMock, loginUrl } from "@/lib/api/client";
import { Badge, Banner, buttonClass } from "@/components/ui";
import { GitHubIcon } from "@/components/icons";

/** The message for `/login?error=<code>` (docs/api.md, OAuth callback). */
function loginErrorMessage(code: string | null): string | null {
  switch (code) {
    case null:
    case "":
      return null;
    case "oauth_state":
      return "Your sign-in link expired. Try again.";
    case "oauth_denied":
      return "GitHub sign-in was cancelled.";
    default:
      return "Something went wrong signing in with GitHub. Try again.";
  }
}

function SignIn({ next, error }: { next?: string; error: string | null }) {
  return (
    <>
      {error ? (
        <Banner tone="bad" className="mt-5">
          {error}
        </Banner>
      ) : null}
      <a href={loginUrl(next)} className={buttonClass("primary", "md", "mt-5 w-full")}>
        <GitHubIcon />
        Sign in with GitHub
      </a>
    </>
  );
}

function SignInFromParams() {
  const params = useSearchParams();
  return <SignIn next={params.get("next") ?? undefined} error={loginErrorMessage(params.get("error"))} />;
}

export function LoginPage() {
  return (
    <main className="grid min-h-dvh place-items-center bg-panel px-4 py-10">
      <div className="w-full max-w-sm rounded-lg border border-line bg-raised p-6">
        <div className="flex items-center gap-2 text-[15px] font-semibold tracking-tight">
          <span className="grid size-6 place-items-center rounded-md bg-fg font-mono text-xs text-canvas" aria-hidden>
            K
          </span>
          Keith
        </div>
        <h1 className="mt-5 text-lg font-semibold tracking-tight">Sign in</h1>
        <p className="mt-1 text-[13px] text-muted">
          Hand a coding task to an agent and watch it work until it opens a pull request.
        </p>

        <Suspense fallback={<SignIn error={null} />}>
          <SignInFromParams />
        </Suspense>

        {isMock ? (
          <div className="mt-5 border-t border-line pt-4 text-[13px] text-muted">
            <p className="flex flex-wrap items-center gap-1.5">
              <Badge tone="warn" className="font-mono">
                mock
              </Badge>
              Mock mode needs no GitHub sign-in.
            </p>
            <p className="mt-1.5">
              <Link href="/sessions/demo" className="text-accent-text hover:underline">
                Open the demo session
              </Link>
            </p>
          </div>
        ) : null}
      </div>
    </main>
  );
}
