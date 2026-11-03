"use client";

import { useId, useState, type FormEvent, type KeyboardEvent } from "react";
import type { ChatMessage, ChatQuestion } from "@/lib/session/reducer";
import { isTerminal, type SessionStatusName } from "@/lib/api/events";
import { Button, Kbd, Textarea } from "../ui";
import { QuestionIcon, SendIcon } from "../icons";
import { RichText } from "./rich-text";

export interface ComposerProps {
  status: SessionStatusName | null;
  /** The open question, if the agent is waiting on one. */
  question: ChatQuestion | null;
  /** The answer already sent for the open question, while it is pending. */
  pendingAnswer: ChatMessage | null;
  /** True when the stream is gone for good (signed out, no access). */
  disabled: boolean;
  onSend(text: string): Promise<void>;
  onAnswer(questionId: string, text: string): Promise<void>;
}

/** The message input, or the answer form while the agent waits on a question. */
export function Composer(props: ComposerProps) {
  if (props.question) {
    // Keyed by question so a new question starts with an empty form.
    return <AnswerForm key={props.question.id} {...props} question={props.question} />;
  }
  return <MessageForm {...props} />;
}

function submitOnEnter(e: KeyboardEvent<HTMLTextAreaElement>) {
  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
    e.preventDefault();
    e.currentTarget.form?.requestSubmit();
  }
}

function MessageForm({ status, disabled, onSend }: ComposerProps) {
  const id = useId();
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const finished = isTerminal(status);
  const label = finished ? "Send a follow-up" : "Steer the agent";
  const placeholder = finished
    ? "Ask for a change and the session picks up where it left off…"
    : "Redirect or add context. The agent reads this on its next turn…";

  async function submit(e: FormEvent) {
    e.preventDefault();
    const t = text.trim();
    if (!t || sending) return;
    setSending(true);
    setError(null);
    try {
      await onSend(t);
      setText("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't send the message.");
    } finally {
      setSending(false);
    }
  }

  return (
    <form onSubmit={submit} className="border-t border-line bg-panel px-3 py-2.5">
      <label htmlFor={id} className="sr-only">
        {label}
      </label>
      <div className="flex items-end gap-2">
        <Textarea
          id={id}
          rows={2}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={submitOnEnter}
          placeholder={placeholder}
          disabled={disabled}
          aria-describedby={error ? `${id}-error` : `${id}-hint`}
          aria-invalid={error ? true : undefined}
          className="max-h-40 min-h-[2.75rem] resize-y"
          maxLength={16_000}
        />
        <Button type="submit" variant="primary" loading={sending} disabled={disabled || !text.trim()} aria-label="Send">
          {sending ? null : <SendIcon />}
        </Button>
      </div>
      {error ? (
        <p id={`${id}-error`} role="alert" className="mt-1 text-xs text-bad">
          {error}
        </p>
      ) : (
        <p id={`${id}-hint`} className="mt-1 text-[11px] text-subtle">
          <Kbd>Enter</Kbd> to send, <Kbd>Shift</Kbd>+<Kbd>Enter</Kbd> for a new line
        </p>
      )}
    </form>
  );
}

function AnswerForm({ question, pendingAnswer, disabled, onAnswer }: ComposerProps & { question: ChatQuestion }) {
  const id = useId();
  const [text, setText] = useState("");
  const [sending, setSending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const locked = disabled || sending !== null || pendingAnswer !== null;

  async function send(answer: string) {
    const t = answer.trim();
    if (!t || locked) return;
    setSending(t);
    setError(null);
    try {
      await onAnswer(question.id, t);
      setText("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't send the answer.");
    } finally {
      setSending(null);
    }
  }

  return (
    <section aria-labelledby={`${id}-title`} className="border-t border-line bg-wait-soft px-3 py-2.5">
      <h2 id={`${id}-title`} className="mb-1 flex items-center gap-1.5 text-xs font-medium text-wait">
        <QuestionIcon className="size-3.5" />
        Keith is waiting for your answer
      </h2>
      <RichText text={question.question} className="mb-2 text-[13px] text-fg" />

      {pendingAnswer ? (
        <p role="status" className="text-[13px] text-muted">
          Answer sent: “{pendingAnswer.text}”. Waiting for the agent to resume…
        </p>
      ) : (
        <>
          {question.choices.length > 0 ? (
            <div role="group" aria-label="Choose an answer" className="mb-2 flex flex-col gap-1.5">
              {question.choices.map((c) => (
                <Button
                  key={c}
                  className="h-auto min-h-8 justify-start py-1.5 text-left whitespace-normal"
                  disabled={locked}
                  loading={sending === c}
                  onClick={() => void send(c)}
                >
                  {c}
                </Button>
              ))}
            </div>
          ) : null}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void send(text);
            }}
            className="flex items-end gap-2"
          >
            <label htmlFor={id} className="sr-only">
              {question.choices.length > 0 ? "Or write your own answer" : "Your answer"}
            </label>
            <Textarea
              id={id}
              rows={1}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={submitOnEnter}
              placeholder={question.choices.length > 0 ? "Or write your own answer…" : "Your answer…"}
              disabled={locked}
              className="max-h-32 min-h-8 resize-y bg-raised"
              maxLength={16_000}
            />
            <Button
              type="submit"
              variant="primary"
              disabled={locked || !text.trim()}
              loading={sending !== null && sending === text.trim()}
            >
              Answer
            </Button>
          </form>
        </>
      )}
      {error ? (
        <p role="alert" className="mt-1 text-xs text-bad">
          {error}
        </p>
      ) : null}
    </section>
  );
}
