// A deliberately small renderer for agent messages: paragraphs, ``` fenced
// code blocks, "- " bullet lists and `inline code`. Everything is rendered
// as React text, so nothing in a message can inject markup.

import { Fragment, type ReactNode } from "react";
import { cx } from "@/lib/cx";

type Block = { kind: "code"; text: string } | { kind: "para"; text: string } | { kind: "list"; items: string[] };

function parseBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  // Odd segments are the insides of ``` fences; an unclosed fence (a
  // message still streaming) renders as code too.
  text.split(/```/).forEach((seg, i) => {
    if (i % 2 === 1) {
      blocks.push({ kind: "code", text: seg.replace(/^[\w+-]*\n/, "").replace(/\n$/, "") });
      return;
    }
    for (const para of seg.split(/\n{2,}/)) {
      const t = para.replace(/^\n+|\n+$/g, "");
      if (!t.trim()) continue;
      const lines = t.split("\n");
      if (lines.every((l) => /^\s*[-*] /.test(l))) {
        blocks.push({ kind: "list", items: lines.map((l) => l.replace(/^\s*[-*] /, "")) });
      } else {
        blocks.push({ kind: "para", text: t });
      }
    }
  });
  return blocks;
}

function inline(text: string): ReactNode[] {
  return text.split(/(`[^`\n]+`)/g).map((part, i) =>
    part.length > 2 && part.startsWith("`") && part.endsWith("`") ? (
      <code key={i} className="rounded bg-hover px-1 py-px font-mono text-[0.92em]">
        {part.slice(1, -1)}
      </code>
    ) : (
      <Fragment key={i}>{part}</Fragment>
    ),
  );
}

export interface RichTextProps {
  text: string;
  className?: string;
  /** Rendered at the end of the last paragraph (the streaming caret). */
  trailing?: ReactNode;
}

export function RichText({ text, className, trailing }: RichTextProps) {
  const blocks = parseBlocks(text);
  const last = blocks.length - 1;
  return (
    <div className={cx("flex flex-col gap-1.5 break-words", className)}>
      {blocks.map((b, i) =>
        b.kind === "list" ? (
          <ul key={i} className="list-disc space-y-0.5 pl-5">
            {b.items.map((item, j) => (
              <li key={j}>
                {inline(item)}
                {i === last && j === b.items.length - 1 ? trailing : null}
              </li>
            ))}
          </ul>
        ) : b.kind === "code" ? (
          <pre
            key={i}
            className="overflow-x-auto rounded-md border border-line bg-panel px-2.5 py-2 font-mono text-xs leading-relaxed"
          >
            <code>{b.text}</code>
            {i === last ? trailing : null}
          </pre>
        ) : (
          <p key={i} className="whitespace-pre-wrap">
            {inline(b.text)}
            {i === last ? trailing : null}
          </p>
        ),
      )}
      {blocks.length === 0 && trailing ? <p>{trailing}</p> : null}
    </div>
  );
}
