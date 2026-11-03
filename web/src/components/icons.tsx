// A small set of 16px stroke icons. Decorative by default (aria-hidden);
// pass `title` to give one an accessible name.

import type { ReactNode } from "react";
import { cx } from "@/lib/cx";

interface IconProps {
  className?: string;
  title?: string;
}

function Svg({ className, title, children }: IconProps & { children: ReactNode }) {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={cx("size-4 shrink-0", className)}
      role={title ? "img" : undefined}
      aria-hidden={title ? undefined : true}
    >
      {title ? <title>{title}</title> : null}
      {children}
    </svg>
  );
}

export const PlusIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M8 3v10M3 8h10" />
  </Svg>
);

export const ListIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M5.5 4h8M5.5 8h8M5.5 12h8M2.5 4h.01M2.5 8h.01M2.5 12h.01" />
  </Svg>
);

export const RepoIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M3.5 13V3.5a1.5 1.5 0 0 1 1.5-1.5h7.5v9.5H5a1.5 1.5 0 0 0-1.5 1.5Zm0 0A1.5 1.5 0 0 0 5 14.5h7.5V11.5" />
  </Svg>
);

export const PlugIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M6 1.5v3M10 1.5v3M4 4.5h8V7a4 4 0 0 1-8 0V4.5ZM8 11v3.5" />
  </Svg>
);

export const KeyIcon = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="5.5" cy="10.5" r="3" />
    <path d="m7.7 8.3 6-6M11.5 4.5l1.5 1.5M10 6l1 1" />
  </Svg>
);

export const LogOutIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M6 14H3.5A1.5 1.5 0 0 1 2 12.5v-9A1.5 1.5 0 0 1 3.5 2H6M10.5 11.5 14 8l-3.5-3.5M14 8H6" />
  </Svg>
);

export const StopIcon = (p: IconProps) => (
  <Svg {...p}>
    <rect x="4" y="4" width="8" height="8" rx="1" />
  </Svg>
);

export const SendIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M14 2 7 9M14 2l-4.5 12L7 9 2 6.5 14 2Z" />
  </Svg>
);

export const TerminalIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="m3 4.5 3 3-3 3M8 11.5h5" />
  </Svg>
);

export const DiffIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M8 2v6M5 5h6M5 13h6" />
  </Svg>
);

export const StepsIcon = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="4" cy="4" r="1.5" />
    <circle cx="4" cy="12" r="1.5" />
    <path d="M4 5.5v5M7.5 4h6M7.5 12h6" />
  </Svg>
);

export const CheckIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="m3 8.5 3 3 7-7" />
  </Svg>
);

export const XIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="m4 4 8 8M12 4l-8 8" />
  </Svg>
);

export const AlertIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M8 1.75 14.5 13.5h-13L8 1.75ZM8 6.5v3M8 11.5h.01" />
  </Svg>
);

export const ExternalIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M9.5 2.5h4v4M13.5 2.5 7.5 8.5M11.5 9.5v3a1 1 0 0 1-1 1h-7a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1h3" />
  </Svg>
);

export const PullRequestIcon = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="4" cy="3.5" r="1.5" />
    <circle cx="4" cy="12.5" r="1.5" />
    <circle cx="12" cy="12.5" r="1.5" />
    <path d="M4 5v6M12 11V6a2 2 0 0 0-2-2H7.5M9 2.5 7.5 4 9 5.5" />
  </Svg>
);

export const ChevronIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="m6 4 4 4-4 4" />
  </Svg>
);

export const LockIcon = (p: IconProps) => (
  <Svg {...p}>
    <rect x="3" y="7" width="10" height="7" rx="1.5" />
    <path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2" />
  </Svg>
);

export const QuestionIcon = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="8" cy="8" r="6.25" />
    <path d="M6.25 6.25a1.75 1.75 0 1 1 2.6 1.53c-.5.28-.85.7-.85 1.22v.25M8 11.5h.01" />
  </Svg>
);

export const GitHubIcon = ({ className, title }: IconProps) => (
  <svg
    viewBox="0 0 16 16"
    fill="currentColor"
    className={cx("size-4 shrink-0", className)}
    role={title ? "img" : undefined}
    aria-hidden={title ? undefined : true}
  >
    {title ? <title>{title}</title> : null}
    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
  </svg>
);
