import type { Metadata } from "next";
import { SecretsPage } from "@/components/pages/secrets-page";

export const metadata: Metadata = { title: "Secrets" };

export default function Page() {
  return <SecretsPage />;
}
