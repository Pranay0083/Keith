import type { Metadata } from "next";
import { ReposPage } from "@/components/pages/repos-page";

export const metadata: Metadata = { title: "Repos" };

export default function Page() {
  return <ReposPage />;
}
