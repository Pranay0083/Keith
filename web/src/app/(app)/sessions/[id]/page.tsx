import type { Metadata } from "next";
import { SessionView } from "@/components/session/session-view";

export const metadata: Metadata = { title: "Session" };

export default async function SessionPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  // Keyed by id so moving between sessions starts from a fresh event state.
  return <SessionView key={id} id={id} />;
}
