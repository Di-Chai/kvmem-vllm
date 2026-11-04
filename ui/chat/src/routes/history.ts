// Normalize old exports without changing their source databases. Preserve unknown
// fields and message trees; namespace identifiers to prevent cross-source clashes.
export async function normalizeHistory(value: any) {
  let items = Array.isArray(value)
    ? value
    : value.conversations || (value.conv ? [value] : undefined);
  if (
    value.format === "polystrata" ||
    (Array.isArray(items) && items[0]?.role)
  ) {
    const rows = value.messages || items;
    items = [
      {
        conv: {
          id: "strata-current",
          name: "PolyStrata",
          lastModified: rows.at(-1)?.time || Date.now(),
        },
        messages: rows.map((m: any, i: number) => ({
          ...m,
          id: String(i),
          convId: "strata-current",
          role: m.role,
          content: m.text || "",
          reasoningContent: m.reasoning || "",
          timestamp: m.time || i,
          parent: i ? String(i - 1) : null,
          children: i + 1 < rows.length ? [String(i + 1)] : [],
          extra: (m.images || [])
            .filter((x: any) => x.url)
            .map((x: any) => ({
              type: "image",
              name: x.name,
              base64Url: x.url,
            })),
          uiStatus: m.error ? "error" : m.stopped ? "interrupted" : "complete",
          legacySource: m,
        })),
      },
    ];
  }
  if (!Array.isArray(items) || items.length > 10000)
    throw Error("Unsupported history format");
  const result: any[] = [];
  let warnings = 0;
  for (const item of items) {
    const original = item.conv || item.conversation;
    if (!original?.id || !Array.isArray(item.messages))
      throw Error("Conversation or messages missing");
    const source = item.source || value.source || "import";
    const bytes = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(source + ":" + original.id),
    );
    const prefix =
      source === location.origin
        ? String(original.id)
        : "import-" +
          Array.from(new Uint8Array(bytes))
            .map((n) => n.toString(16).padStart(2, "0"))
            .join("")
            .slice(0, 24);
    const id = (v: any) =>
      v == null
        ? null
        : source === location.origin
          ? String(v)
          : prefix + ":" + String(v);
    const seen = new Set();
    const messages = item.messages.map((m: any, i: number) => {
      if (
        typeof m.id !== "string" ||
        seen.has(m.id) ||
        !["user", "assistant", "system", "tool"].includes(m.role)
      )
        throw Error("Invalid message identity or role");
      seen.add(m.id);
      if (
        (m.extra || []).some((e: any) => e.type !== "image") ||
        m.tool_calls ||
        m.toolCalls ||
        m.tools?.length ||
        m.legacySource?.images?.some((x: any) => !x.url) ||
        (m.content != null && typeof m.content !== "string")
      )
        warnings++;
      return {
        ...m,
        id: id(m.id),
        convId: prefix,
        parent: id(m.parent),
        children: (m.children || []).map(id),
        content: typeof m.content === "string" ? m.content : "",
        ...(m.content != null && typeof m.content !== "string"
          ? { legacyContent: m.content }
          : {}),
        reasoningContent: m.reasoningContent || m.reasoning_content || "",
        timestamp: Number.isFinite(m.timestamp) ? m.timestamp : i,
        type: m.type || "text",
        uiStatus: m.uiStatus === "pending" ? "interrupted" : m.uiStatus,
      };
    });
    // Keep all branches in storage, and select the original leaf when available.
    const current = original.currNode || item.messages.at(-1)?.id || null;
    if (current != null && !seen.has(current))
      throw Error("History leaf is missing");
    const mapped = new Set(messages.map((m: any) => m.id));
    if (messages.some((m: any) => m.parent && !mapped.has(m.parent)))
      throw Error("History parent is missing");
    const parents = new Map(messages.map((m: any) => [m.id, m.parent])),
      checked = new Set();
    for (const m of messages) {
      let key = m.id;
      const path = new Set();
      while (key && !checked.has(key)) {
        if (path.has(key)) throw Error("History contains a cycle");
        path.add(key);
        key = parents.get(key);
      }
      for (const id of path) checked.add(id);
    }
    result.push({
      conv: {
        ...original,
        id: prefix,
        currNode: id(current),
        name: String(original.name || "Imported chat"),
        lastModified: Number(original.lastModified) || Date.now(),
        importSource: { source, id: original.id },
      },
      messages,
    });
  }
  return { items: result, warnings, legacy: !!value.legacy?.length };
}
