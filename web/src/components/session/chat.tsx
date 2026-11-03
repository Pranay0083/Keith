"use client";

import type { ChatItem, ChatMessage, ChatQuestion } from "@/lib/session/reducer";
import { cx } from "@/lib/cx";
import { QuestionIcon } from "../icons";
import { RelativeTime } from "../time";
import { messageAuthor, type ChatViewer } from "./format";
import { useStickToBottom } from "./hooks";
import { RichText } from "./rich-text";

export interface ChatProps {
  items: ChatItem[];
  /** Shown when there are no items yet. */
  emptyText: string;
  viewer: ChatViewer;
}

export function Chat({ items, emptyText, viewer }: ChatProps) {
  const last = items[items.length - 1];
  const tick = `${items.length}:${last?.kind === "message" ? last.text.length : ""}:${last && "pending" in last ? last.pending : ""}`;
  const { ref, onScroll } = useStickToBottom<HTMLDivElement>(tick);
  const questions = new Map(items.filter((c): c is ChatQuestion => c.kind === "question").map((q) => [q.id, q]));

  return (
    <div
      ref={ref}
      onScroll={onScroll}
      role="log"
      aria-label="Conversation"
      className="min-h-0 flex-1 overflow-y-auto px-4 py-3"
    >
      {items.length === 0 ? (
        <p className="py-6 text-center text-[13px] text-subtle">{emptyText}</p>
      ) : (
        <ol className="flex flex-col gap-3">
          {items.map((item) =>
            item.kind === "message" ? (
              <li key={`m:${item.id}`}>
                <Message
                  msg={item}
                  author={messageAuthor(item, viewer)}
                  answers={item.questionId ? questions.get(item.questionId) : undefined}
                />
              </li>
            ) : (
              <li key={`q:${item.id}`}>
                <Question q={item} />
              </li>
            ),
          )}
        </ol>
      )}
    </div>
  );
}

function Message({ msg, author, answers }: { msg: ChatMessage; author: string; answers?: ChatQuestion }) {
  const user = msg.role === "user";
  return (
    <article
      aria-busy={msg.streaming || undefined}
      className={cx(
        "text-[13.5px] leading-relaxed",
        user && "rounded-lg border border-line bg-accent-soft px-3 py-2",
        msg.pending && "opacity-70",
      )}
    >
      <header className="mb-0.5 flex items-baseline gap-2 text-xs">
        <span className="font-medium text-fg">{author}</span>
        {answers ? <span className="text-subtle">answered a question</span> : null}
        {msg.pending ? (
          <span className="text-subtle">{msg.questionId ? "Sending answer…" : "Queued for the agent's next turn"}</span>
        ) : msg.ts ? (
          <RelativeTime iso={msg.ts} className="text-subtle" />
        ) : null}
      </header>
      <RichText
        text={msg.text}
        trailing={msg.streaming ? <span className="streaming-caret" aria-hidden /> : undefined}
      />
    </article>
  );
}

function Question({ q }: { q: ChatQuestion }) {
  const state = q.answerId ? "Answered" : q.expired ? "Not answered" : "Waiting for your answer";
  return (
    <article className="rounded-lg border border-line bg-wait-soft px-3 py-2 text-[13.5px]">
      <header className="mb-1 flex items-center gap-1.5 text-xs">
        <QuestionIcon className="size-3.5 text-wait" />
        <span className="font-medium text-wait">Keith asked</span>
        <span className="text-subtle">· {state}</span>
      </header>
      <RichText text={q.question} />
      {q.choices.length > 0 ? (
        <ul className="mt-1.5 flex flex-wrap gap-1.5" aria-label="Suggested answers">
          {q.choices.map((c) => (
            <li key={c} className="rounded border border-line-strong bg-raised px-1.5 py-0.5 text-xs text-muted">
              {c}
            </li>
          ))}
        </ul>
      ) : null}
    </article>
  );
}
