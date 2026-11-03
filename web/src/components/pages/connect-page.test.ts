import { describe, expect, it } from "vitest";
import { setupError } from "./connect-page";

describe("setupError", () => {
  it("explains who may connect an installation", () => {
    const { body } = setupError("installation_not_verified");
    expect(body).toContain("the GitHub account the app is installed on");
    expect(body).toContain("an owner of the GitHub organization");
    expect(body).toContain("Access to its repos isn't enough");
    expect(body).not.toMatch(/can access that installation/);
  });

  it("points a failed org setup at the Members permission", () => {
    expect(setupError("setup_failed").body).toContain("“Members: Read-only” organization permission");
  });

  it("falls back for a code this build doesn't know", () => {
    expect(setupError("something_new").title).toBe("The install didn't finish");
  });
});
