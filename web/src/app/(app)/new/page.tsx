import type { Metadata } from "next";
import { NewTaskPage } from "@/components/pages/new-task-page";

export const metadata: Metadata = { title: "New task" };

export default function Page() {
  return <NewTaskPage />;
}
