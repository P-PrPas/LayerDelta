import {connect, get, Page, post} from "./api";
import {Checkpoint, CheckpointSource, ImportProgress, InspectError} from "./tensorSource";

type ImportRecord = Omit<ImportProgress, "stage"> & {
  import_id: string; status: "queued" | "running" | "succeeded" | "failed";
  stage: ImportProgress["stage"] | null; checkpoint_id: string | null; existing: boolean | null;
  error: {code: string; message: string} | null;
};

const imports = "/checkpoint-imports";

export const liveCheckpointSource: CheckpointSource = {
  async listCheckpoints() {
    await connect();
    const entries: Checkpoint[] = [];
    let cursor: string | null = null;
    do {
      const page: Page<Checkpoint> = await get("/checkpoints?limit=200" +
        (cursor ? "&cursor=" + encodeURIComponent(cursor) : ""));
      entries.push(...page.items);
      cursor = page.next_cursor;
    } while (cursor);
    return entries;
  },

  async getCheckpoint(id) {
    await connect();
    return get<Checkpoint>("/checkpoints/" + encodeURIComponent(id));
  },

  async importCheckpoint({path, file}, observe) {
    await connect();
    let job = await (file ?
      post<ImportRecord>(imports + "/upload?" + new URLSearchParams({filename: file.name}), file) :
      post<ImportRecord>(imports, {path}));
    const poll = imports + "/" + encodeURIComponent(job.import_id);
    const deadline = Date.now() + 15 * 60_000;
    for (;;) {
      observe(job.stage ? {...job, stage: job.stage} : null);
      if (job.status === "succeeded" && job.checkpoint_id) {
        const checkpoint = await liveCheckpointSource.getCheckpoint(job.checkpoint_id);
        return {checkpoint, existing: job.existing === true};
      }
      if (job.status === "failed") {
        throw new InspectError(job.error?.code ?? "import_failed", job.error?.message ?? "The import failed.");
      }
      if (Date.now() >= deadline)
        throw new InspectError("import_timeout", "The import is still running. Reopen Tensors to refresh the checkpoint list.");
      await new Promise(resolve => setTimeout(resolve, 300));
      job = await get<ImportRecord>(poll);
    }
  },
};

export function checkpointLabel(name: string) {
  const letters = [...name];
  return letters.length <= 36 ? name : letters.slice(0, 19).join("") + "…" + letters.slice(-16).join("");
}
