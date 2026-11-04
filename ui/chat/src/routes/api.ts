import { splitSseRecords, extractSseDataPayload } from "$lib/utils/sse";

export function errorText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value && typeof value === "object" && "message" in value)
    return String(value.message);
  return "Request failed";
}

let consoleToken = "";
let browserClient = "";
function clientId() {
  if (browserClient) return browserClient;
  try { browserClient = localStorage.getItem("kvmem-vllm-client") || ""; } catch {}
  if (!browserClient) {
    browserClient = crypto.randomUUID();
    try { localStorage.setItem("kvmem-vllm-client", browserClient); } catch {}
  }
  return browserClient;
}
const clientHeaders = () => ({ "X-KVMem-Client": clientId() });
export async function context() {
  const info = await fetch("/api/launcher/info");
  if (!info.ok) throw Error("Open chat from the KVMem console");
  const launcher = await info.json();
  consoleToken = launcher.token;
  const response = await fetch("/api/chat/context", { headers: clientHeaders() });
  if (!response.ok) throw Error("Service settings unavailable");
  const result = await response.json();
  // Display the loaded file, without changing the API model identity or reading
  // the editable launch draft. A service change must not mix the two snapshots.
  const running = launcher.status;
  if (
    running?.state === "ready" &&
    result.instance &&
    running.instance === result.instance
  ) {
    const model = running.active_config?.parameters?.Model;
    if (typeof model === "string" && model)
      result.display_model = model.split(/[\\/]/).pop();
  }
  return result;
}
export async function requestRecord(id: string) {
  const response = await fetch("/api/chat/requests/" + encodeURIComponent(id), { headers: clientHeaders() });
  return response.ok ? response.json() : null;
}
export async function cancelRequest(id: string) {
  await fetch("/api/chat/cancel", {
    method: "POST",
    keepalive: true,
    headers: {
      "Content-Type": "application/json",
      "X-KVMem-Token": consoleToken,
      ...clientHeaders(),
    },
    body: JSON.stringify({ id, client_id: clientId() }),
  }).catch(() => {});
}
export async function releaseConversation(conversationId: string) {
  const response = await fetch("/api/chat/release", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-KVMem-Token": consoleToken, ...clientHeaders() },
    body: JSON.stringify({ conversation_id: conversationId, client_id: clientId() }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw Error(errorText(body.error || response.statusText));
  }
}
export async function streamChat(
  body: object,
  signal: AbortSignal,
  event: (data: any) => void,
  id: string,
  instance: string,
) {
  const response = await fetch("/api/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-KVMem-Token": consoleToken,
      "X-KVMem-Request": id,
      "X-KVMem-Instance": instance,
      ...clientHeaders(),
    },
    body: JSON.stringify({ ...body, client_id: clientId() }),
    signal,
  });
  if (!response.ok) {
    const text = await response.text();
    let detail;
    try {
      detail = errorText(JSON.parse(text).error);
    } catch {
      detail = text || response.statusText;
    }
    throw new Error(detail);
  }
  const reader = response.body?.getReader();
  if (!reader) throw new Error("Empty response");
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done)
        throw new Error(
          "Connection interrupted. Send a new message to continue.",
        );
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      const parsed = splitSseRecords(buffer);
      buffer = parsed.rest;
      for (const record of parsed.records) {
        const payload = extractSseDataPayload(record);
        if (!payload) continue;
        if (payload === "[DONE]") return;
        const data = JSON.parse(payload);
        if (data.error) throw new Error(errorText(data.error));
        event(data);
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

