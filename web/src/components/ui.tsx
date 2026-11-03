// Presentational primitives. No hooks here, so these render on the server
// or the client. Colours come from the tokens in globals.css.

import Link from "next/link";
import type { ComponentProps, ReactNode } from "react";
import { cx } from "@/lib/cx";
import type { SessionStatusName } from "@/lib/api/events";

// --- Buttons ---

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
type ButtonSize = "sm" | "md";

const buttonBase =
  "inline-flex items-center justify-center gap-1.5 rounded-md font-medium whitespace-nowrap select-none transition-colors disabled:cursor-not-allowed disabled:opacity-55";

const buttonVariants: Record<ButtonVariant, string> = {
  primary: "bg-accent text-accent-fg hover:bg-accent-hover border border-transparent",
  secondary: "bg-raised text-fg border border-line-strong hover:bg-hover",
  ghost: "text-muted hover:text-fg hover:bg-hover border border-transparent",
  danger: "bg-raised text-bad border border-line-strong hover:bg-bad-soft",
};

const buttonSizes: Record<ButtonSize, string> = {
  sm: "h-7 px-2.5 text-[13px]",
  md: "h-8 px-3 text-sm",
};

export function buttonClass(variant: ButtonVariant = "secondary", size: ButtonSize = "md", className?: string) {
  return cx(buttonBase, buttonVariants[variant], buttonSizes[size], className);
}

export interface ButtonProps extends ComponentProps<"button"> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  /** Shows a spinner and disables the button. */
  loading?: boolean;
}

export function Button({ variant, size, loading, disabled, className, children, type, ...rest }: ButtonProps) {
  return (
    <button
      type={type ?? "button"}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={buttonClass(variant, size, className)}
      {...rest}
    >
      {loading ? <Spinner className="size-3.5" /> : null}
      {children}
    </button>
  );
}

export interface LinkButtonProps extends ComponentProps<typeof Link> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

export function LinkButton({ variant, size, className, ...rest }: LinkButtonProps) {
  return <Link className={buttonClass(variant, size, className)} {...rest} />;
}

// --- Form controls ---

const controlBase =
  "w-full rounded-md border border-line-strong bg-raised px-2.5 text-sm text-fg placeholder:text-subtle disabled:opacity-60 aria-invalid:border-bad";

export function Input({ className, ...rest }: ComponentProps<"input">) {
  return <input className={cx(controlBase, "h-8", className)} {...rest} />;
}

export function Textarea({ className, ...rest }: ComponentProps<"textarea">) {
  return <textarea className={cx(controlBase, "py-1.5 leading-relaxed", className)} {...rest} />;
}

export function Select({ className, ...rest }: ComponentProps<"select">) {
  return <select className={cx(controlBase, "h-8 pr-7", className)} {...rest} />;
}

export interface FieldProps {
  label: ReactNode;
  htmlFor: string;
  hint?: ReactNode;
  error?: string;
  children: ReactNode;
  className?: string;
}

/**
 * A label, a control, and an optional hint and error. Give the control
 * `id={htmlFor}` and `aria-describedby={fieldDescribedBy(htmlFor, hint, error)}`.
 */
export function Field({ label, htmlFor, hint, error, children, className }: FieldProps) {
  return (
    <div className={cx("flex flex-col gap-1", className)}>
      <label htmlFor={htmlFor} className="text-[13px] font-medium text-fg">
        {label}
      </label>
      {children}
      {hint && !error ? (
        <p id={`${htmlFor}-hint`} className="text-xs text-subtle">
          {hint}
        </p>
      ) : null}
      {error ? (
        <p id={`${htmlFor}-error`} className="text-xs text-bad">
          {error}
        </p>
      ) : null}
    </div>
  );
}

export function fieldDescribedBy(id: string, hint?: ReactNode, error?: string): string | undefined {
  if (error) return `${id}-error`;
  if (hint) return `${id}-hint`;
  return undefined;
}

// --- Badges ---

export type Tone = "neutral" | "accent" | "ok" | "warn" | "bad" | "wait";

const toneClasses: Record<Tone, string> = {
  neutral: "bg-hover text-muted",
  accent: "bg-accent-soft text-accent-text",
  ok: "bg-ok-soft text-ok",
  warn: "bg-warn-soft text-warn",
  bad: "bg-bad-soft text-bad",
  wait: "bg-wait-soft text-wait",
};

export function Badge({ tone = "neutral", className, children }: { tone?: Tone; className?: string; children: ReactNode }) {
  return (
    <span
      className={cx(
        "inline-flex h-5 items-center gap-1 rounded px-1.5 text-xs font-medium whitespace-nowrap",
        toneClasses[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

const statusMeta: Record<SessionStatusName, { label: string; tone: Tone; live: boolean }> = {
  queued: { label: "Queued", tone: "neutral", live: true },
  provisioning: { label: "Provisioning", tone: "accent", live: true },
  setting_up: { label: "Setting up", tone: "accent", live: true },
  running: { label: "Running", tone: "accent", live: true },
  waiting_for_user: { label: "Needs input", tone: "wait", live: false },
  completed: { label: "Completed", tone: "ok", live: false },
  failed: { label: "Failed", tone: "bad", live: false },
  cancelled: { label: "Cancelled", tone: "neutral", live: false },
  no_changes: { label: "No changes", tone: "neutral", live: false },
};

/**
 * How to show a status. The API can be newer than this build, so a status
 * it doesn't know shows as itself ("needs_review" as "Needs review").
 */
function statusInfo(status: string): { label: string; tone: Tone; live: boolean } {
  if (Object.hasOwn(statusMeta, status)) return statusMeta[status as SessionStatusName];
  const words = status.replaceAll("_", " ");
  return { label: words.charAt(0).toUpperCase() + words.slice(1), tone: "neutral", live: false };
}

export function statusLabel(status: SessionStatusName | string): string {
  return statusInfo(status).label;
}

export function StatusBadge({ status, className }: { status: SessionStatusName | string | null; className?: string }) {
  if (!status) {
    return (
      <Badge className={className}>
        <span className="size-1.5 rounded-full bg-current opacity-60" aria-hidden />
        Loading
      </Badge>
    );
  }
  const m = statusInfo(status);
  return (
    <Badge tone={m.tone} className={className}>
      <span className={cx("size-1.5 rounded-full bg-current", m.live && "pulse-dot")} aria-hidden />
      {m.label}
    </Badge>
  );
}

// --- Banners and surfaces ---

const bannerTones: Record<"info" | "ok" | "warn" | "bad", string> = {
  info: "border-line bg-accent-soft text-fg",
  ok: "border-line bg-ok-soft text-fg",
  warn: "border-line bg-warn-soft text-fg",
  bad: "border-line bg-bad-soft text-fg",
};

const bannerIcon: Record<"info" | "ok" | "warn" | "bad", string> = {
  info: "text-accent-text",
  ok: "text-ok",
  warn: "text-warn",
  bad: "text-bad",
};

export interface BannerProps {
  tone?: "info" | "ok" | "warn" | "bad";
  title?: ReactNode;
  children?: ReactNode;
  actions?: ReactNode;
  className?: string;
}

/** An inline notice. Errors use role="alert"; everything else is a polite status. */
export function Banner({ tone = "info", title, children, actions, className }: BannerProps) {
  return (
    <div
      role={tone === "bad" ? "alert" : "status"}
      className={cx("flex items-start gap-2.5 rounded-md border px-3 py-2 text-[13px]", bannerTones[tone], className)}
    >
      <span className={cx("mt-[5px] size-2 shrink-0 rounded-full bg-current", bannerIcon[tone])} aria-hidden />
      <div className="min-w-0 flex-1">
        {title ? <p className="font-medium">{title}</p> : null}
        {children ? <div className="text-muted">{children}</div> : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-1.5">{actions}</div> : null}
    </div>
  );
}

export function Panel({ className, children, ...rest }: ComponentProps<"section">) {
  return (
    <section className={cx("rounded-lg border border-line bg-raised", className)} {...rest}>
      {children}
    </section>
  );
}

/** The padded, width-limited container for an ordinary page. */
export function Page({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cx("mx-auto w-full max-w-5xl px-4 py-6 sm:px-6", className)}>{children}</div>;
}

export interface PageHeaderProps {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
}

export function PageHeader({ title, description, actions }: PageHeaderProps) {
  return (
    <header className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div className="min-w-0">
        <h1 className="text-lg font-semibold tracking-tight">{title}</h1>
        {description ? <p className="mt-0.5 text-[13px] text-muted">{description}</p> : null}
      </div>
      {actions ? <div className="flex items-center gap-2">{actions}</div> : null}
    </header>
  );
}

export function EmptyState({ title, description, action }: { title: ReactNode; description?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-line-strong px-6 py-10 text-center">
      <p className="text-sm font-medium">{title}</p>
      {description ? <p className="max-w-md text-[13px] text-muted">{description}</p> : null}
      {action ? <div className="mt-2">{action}</div> : null}
    </div>
  );
}

export function Spinner({ className, label }: { className?: string; label?: string }) {
  return (
    <svg
      className={cx("size-4 animate-spin text-current", className)}
      viewBox="0 0 16 16"
      fill="none"
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      <circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeOpacity="0.25" strokeWidth="2" />
      <path d="M14.5 8A6.5 6.5 0 0 0 8 1.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

/** A centred loading row for a page or panel. */
export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 px-1 py-6 text-[13px] text-muted" role="status">
      <Spinner />
      {label}…
    </div>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded border border-line-strong bg-panel px-1 font-mono text-[11px] text-muted">{children}</kbd>
  );
}

/** Text only screen readers hear. */
export function VisuallyHidden({ children }: { children: ReactNode }) {
  return <span className="sr-only">{children}</span>;
}
