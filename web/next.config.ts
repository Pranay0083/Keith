import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // Don't let `next dev` write AGENTS.md into the repo.
  agentRules: false,
  env: {
    // Always defined at build time, so `=== "1"` checks fold to a constant
    // and a normal build drops the mock API entirely.
    NEXT_PUBLIC_API_MOCK: process.env.NEXT_PUBLIC_API_MOCK === "1" ? "1" : "0",
  },
};

export default nextConfig;
