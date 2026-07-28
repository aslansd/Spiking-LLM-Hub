import { ChatMessage, SNNModel } from "./types";

/**
 * Persistence for chat history and neuromorphic parameters.
 *
 * Every entry point is defensive: localStorage throws rather than returning
 * null in Safari private browsing and when a quota is exceeded, and stored
 * data written by an older build may not match the current shape.
 */

const VERSION = "v1";
const CHATS_KEY = `spiking-hub:${VERSION}:chats`;
const PARAMS_KEY = `spiking-hub:${VERSION}:params`;

/** Keeps a single model's history well inside the ~5MB origin quota. */
const MAX_STORED_MESSAGES_PER_MODEL = 60;

export type ModelParams = { threshold: number; decay: number; leak: number };

let storageWarned = false;

function getStorage(): Storage | null {
  try {
    const probe = "__spiking_hub_probe__";
    window.localStorage.setItem(probe, "1");
    window.localStorage.removeItem(probe);
    return window.localStorage;
  } catch {
    if (!storageWarned) {
      console.info("localStorage unavailable; this session will not be saved.");
      storageWarned = true;
    }
    return null;
  }
}

export function isStorageAvailable(): boolean {
  return getStorage() !== null;
}

function isChatMessage(value: unknown): value is ChatMessage {
  if (!value || typeof value !== "object") return false;
  const m = value as Record<string, unknown>;
  return (
    typeof m.id === "string" &&
    typeof m.content === "string" &&
    (m.role === "user" || m.role === "assistant" || m.role === "system")
  );
}

export function loadChats(): Record<string, ChatMessage[]> | null {
  const storage = getStorage();
  if (!storage) return null;

  try {
    const raw = storage.getItem(CHATS_KEY);
    if (!raw) return null;

    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;

    // Drop anything that does not match the current shape rather than letting
    // a stale record crash the render.
    const clean: Record<string, ChatMessage[]> = {};
    for (const [modelId, messages] of Object.entries(parsed)) {
      if (!Array.isArray(messages)) continue;
      const valid = messages.filter(isChatMessage).map((m) => ({
        ...m,
        timestamp: typeof m.timestamp === "string" ? m.timestamp : "",
      }));
      if (valid.length) clean[modelId] = valid;
    }
    return Object.keys(clean).length ? clean : null;
  } catch (err) {
    console.warn("Could not read saved chats; starting fresh.", err);
    return null;
  }
}

export function saveChats(chats: Record<string, ChatMessage[]>): void {
  const storage = getStorage();
  if (!storage) return;

  const trimmed: Record<string, ChatMessage[]> = {};
  for (const [modelId, messages] of Object.entries(chats)) {
    trimmed[modelId] = messages.slice(-MAX_STORED_MESSAGES_PER_MODEL);
  }

  try {
    storage.setItem(CHATS_KEY, JSON.stringify(trimmed));
  } catch {
    // Almost certainly a quota error. Retry once with much less history
    // before giving up; losing old turns beats losing the whole session.
    try {
      const minimal: Record<string, ChatMessage[]> = {};
      for (const [modelId, messages] of Object.entries(trimmed)) {
        minimal[modelId] = messages.slice(-10).map(({ metrics, ...rest }) => rest);
      }
      storage.setItem(CHATS_KEY, JSON.stringify(minimal));
    } catch (err) {
      console.warn("Chat history could not be saved.", err);
    }
  }
}

export function loadParams(): Record<string, ModelParams> | null {
  const storage = getStorage();
  if (!storage) return null;

  try {
    const raw = storage.getItem(PARAMS_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;

    const clean: Record<string, ModelParams> = {};
    for (const [modelId, value] of Object.entries(parsed)) {
      const p = value as Record<string, unknown>;
      if (
        typeof p?.threshold === "number" &&
        typeof p?.decay === "number" &&
        typeof p?.leak === "number"
      ) {
        clean[modelId] = { threshold: p.threshold, decay: p.decay, leak: p.leak };
      }
    }
    return Object.keys(clean).length ? clean : null;
  } catch {
    return null;
  }
}

export function saveParams(params: Record<string, ModelParams>): void {
  const storage = getStorage();
  if (!storage) return;
  try {
    storage.setItem(PARAMS_KEY, JSON.stringify(params));
  } catch {
    /* non-critical */
  }
}

export function clearStoredData(): void {
  const storage = getStorage();
  if (!storage) return;
  try {
    storage.removeItem(CHATS_KEY);
    storage.removeItem(PARAMS_KEY);
  } catch {
    /* nothing useful to do */
  }
}

/* -------------------------------------------------------------------------- */
/* Export                                                                     */
/* -------------------------------------------------------------------------- */

export function chatToMarkdown(
  model: SNNModel,
  messages: ChatMessage[],
  params: ModelParams,
): string {
  const header = [
    `# ${model.name} playground transcript`,
    "",
    `- Exported: ${new Date().toISOString()}`,
    `- Model entry: ${model.name} (${model.type}, ${model.parameters})`,
    `- Parameters: V_th ${params.threshold}, decay ${params.decay}, leak ${params.leak}`,
    "",
    "> These replies were produced by a general-purpose model role-playing this",
    "> architecture, and the metrics are calculated estimates. No spiking network",
    "> was executed.",
    "",
    "---",
    "",
  ].join("\n");

  const body = messages
    .map((m) => {
      const who = m.role === "user" ? "You" : model.name;
      const time = m.timestamp ? ` · ${m.timestamp}` : "";
      const metrics = m.metrics
        ? `\n\n<sub>Estimated: ${m.metrics.spikeCount.toLocaleString()} spikes, ` +
          `${m.metrics.synapticOps.toLocaleString()} SOPs, ` +
          `${m.metrics.energySavedPercent.toFixed(1)}% modelled energy saving</sub>`
        : "";
      return `**${who}**${time}\n\n${m.content}${metrics}`;
    })
    .join("\n\n---\n\n");

  return header + body + "\n";
}

export function chatToJSON(
  model: SNNModel,
  messages: ChatMessage[],
  params: ModelParams,
): string {
  return JSON.stringify(
    {
      exportedAt: new Date().toISOString(),
      disclaimer:
        "Responses are role-played by a general-purpose model; metrics are calculated estimates, not hardware measurements.",
      model: {
        id: model.id,
        name: model.name,
        type: model.type,
        parameters: model.parameters,
        paper: model.paper,
        github: model.github,
      },
      parameters: params,
      messages,
    },
    null,
    2,
  );
}

export function downloadTextFile(filename: string, content: string, mime: string): void {
  const blob = new Blob([content], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  // Revoking immediately can cancel the download in some browsers.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
