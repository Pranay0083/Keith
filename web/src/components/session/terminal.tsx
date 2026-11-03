"use client";

// One read-only xterm.js terminal for one sandbox shell. xterm touches the
// DOM at import time, so it loads in an effect, on the client only. Output
// is written incrementally: the component remembers how many characters it
// has written and only writes the new tail.

import "@xterm/xterm/css/xterm.css";
import { useEffect, useRef, useState } from "react";
import type { FitAddon } from "@xterm/addon-fit";
import type { ITheme, Terminal } from "@xterm/xterm";

const light: ITheme = {
  background: "#fbfbfc",
  foreground: "#27272a",
  cursor: "#27272a",
  cursorAccent: "#fbfbfc",
  selectionBackground: "#bfd3fb",
  black: "#27272a",
  red: "#c4302b",
  green: "#2f7d32",
  yellow: "#9a6a00",
  blue: "#1d5fd1",
  magenta: "#8e3fb5",
  cyan: "#0f7a8a",
  white: "#6b6b74",
  brightBlack: "#71717a",
  brightRed: "#d9443f",
  brightGreen: "#38913c",
  brightYellow: "#ad7800",
  brightBlue: "#2f6fed",
  brightMagenta: "#a04fc9",
  brightCyan: "#138a9b",
  brightWhite: "#3f3f46",
};

const dark: ITheme = {
  background: "#121215",
  foreground: "#e4e4e7",
  cursor: "#e4e4e7",
  cursorAccent: "#121215",
  selectionBackground: "#2c3d63",
  black: "#3f3f46",
  red: "#f87171",
  green: "#4ade80",
  yellow: "#facc15",
  blue: "#60a5fa",
  magenta: "#c084fc",
  cyan: "#22d3ee",
  white: "#d4d4d8",
  brightBlack: "#71717a",
  brightRed: "#fca5a5",
  brightGreen: "#86efac",
  brightYellow: "#fde047",
  brightBlue: "#93c5fd",
  brightMagenta: "#d8b4fe",
  brightCyan: "#67e8f9",
  brightWhite: "#fafafa",
};

const DARK_QUERY = "(prefers-color-scheme: dark)";

export interface TerminalViewProps {
  /** The shell's retained output. */
  output: string;
  /** Characters dropped from the head of `output`. */
  dropped: number;
  /** Whether the tab is showing; a hidden terminal cannot measure itself. */
  visible: boolean;
  label: string;
}

export function TerminalView({ output, dropped, visible, label }: TerminalViewProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const termRef = useRef<{ term: Terminal; fit: FitAddon } | null>(null);
  /** Absolute count of characters written (dropped + written part of output). */
  const writtenRef = useRef(0);
  const [ready, setReady] = useState(false);

  // Create the terminal once.
  useEffect(() => {
    let disposed = false;
    const mql = window.matchMedia(DARK_QUERY);
    const onScheme = () => {
      if (termRef.current) termRef.current.term.options.theme = mql.matches ? dark : light;
    };
    void Promise.all([import("@xterm/xterm"), import("@xterm/addon-fit")]).then(([xterm, fitMod]) => {
      if (disposed || !hostRef.current) return;
      const term = new xterm.Terminal({
        convertEol: true,
        disableStdin: true,
        cursorBlink: false,
        cursorStyle: "bar",
        cursorInactiveStyle: "none",
        fontFamily: 'ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace',
        fontSize: 12,
        lineHeight: 1.25,
        scrollback: 10_000,
        theme: mql.matches ? dark : light,
      });
      const fit = new fitMod.FitAddon();
      term.loadAddon(fit);
      term.open(hostRef.current);
      termRef.current = { term, fit };
      writtenRef.current = 0;
      mql.addEventListener("change", onScheme);
      setReady(true);
    });
    return () => {
      disposed = true;
      mql.removeEventListener("change", onScheme);
      termRef.current?.term.dispose();
      termRef.current = null;
    };
  }, []);

  // Write whatever is new.
  useEffect(() => {
    const t = termRef.current;
    if (!ready || !t) return;
    const total = dropped + output.length;
    if (writtenRef.current < dropped || writtenRef.current > total) {
      // Our position was trimmed away (or the output was replaced): redraw.
      t.term.reset();
      t.term.write(output);
    } else if (total > writtenRef.current) {
      t.term.write(output.slice(writtenRef.current - dropped));
    }
    writtenRef.current = total;
  }, [output, dropped, ready]);

  // Fit to the container while visible.
  useEffect(() => {
    const t = termRef.current;
    const host = hostRef.current;
    if (!ready || !visible || !t || !host) return;
    const fit = () => {
      try {
        t.fit.fit();
      } catch {
        // The container can be mid-layout; the next resize fits it.
      }
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(host);
    return () => ro.disconnect();
  }, [ready, visible]);

  return <div ref={hostRef} role="log" aria-label={label} className="h-full w-full overflow-hidden" />;
}
