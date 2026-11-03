import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { SESSION_STATUSES } from "@/lib/api/events";
import { StatusBadge, statusLabel } from "./ui";

const badge = (status: string | null) => renderToStaticMarkup(createElement(StatusBadge, { status }));

describe("StatusBadge", () => {
  it("labels every status this build knows", () => {
    expect(SESSION_STATUSES.map(statusLabel)).toEqual([
      "Queued",
      "Provisioning",
      "Setting up",
      "Running",
      "Needs input",
      "Completed",
      "Failed",
      "Cancelled",
      "No changes",
    ]);
    expect(badge("running")).toContain("Running");
    expect(badge(null)).toContain("Loading");
  });

  it("shows a status from a newer API as itself instead of crashing", () => {
    expect(statusLabel("needs_review")).toBe("Needs review");
    expect(badge("needs_review")).toContain("Needs review");
    expect(badge("constructor")).toContain("Constructor"); // not Object.prototype's
  });
});
