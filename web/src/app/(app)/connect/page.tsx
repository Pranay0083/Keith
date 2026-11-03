import type { Metadata } from "next";
import { ConnectPage } from "@/components/pages/connect-page";

export const metadata: Metadata = { title: "Connect GitHub" };

export default function Page() {
  return <ConnectPage />;
}
