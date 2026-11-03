import type { Metadata } from "next";
import { SessionsPage } from "@/components/pages/sessions-page";

export const metadata: Metadata = { title: "Sessions" };

export default function Page() {
  return <SessionsPage />;
}
