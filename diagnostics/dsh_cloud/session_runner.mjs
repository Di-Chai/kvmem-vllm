// Replaces only the one-shot headless driver; all prompt/tool plugins stay put.
import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { setTimeout as delay } from "node:timers/promises";

export const name = "gc-single-session-runner";
export const inject = ["agentDefaultModel", "agents", "sessions"];

function writeJSON(file, value) {
  const temp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(temp, JSON.stringify(value, null, 2) + "\n", { mode: 0o600 });
  fs.renameSync(temp, file);
}

export function summarize(events, firstSeq) {
  let started = false, text = "", reason;
  for (const event of events) {
    if (event.seq < firstSeq) continue;
    if (event.type === "turn/start") started = true;
    if (!started) continue;
    if (event.type === "assistant/message") {
      const joined = event.data.message.content
        .filter(block => block.type === "text").map(block => block.text).join("");
      if (joined) text = joined;
    }
    if (event.type === "turn/end") reason = event.data.reason;
  }
  return { text, reason };
}

export function validateCommand(command, index) {
  if (!command || command.index !== index || !["user", "close"].includes(command.kind)) {
    throw new Error(`Invalid queue command ${index}`);
  }
  if (command.kind === "user" &&
      (typeof command.text !== "string" || !command.text.trim())) {
    throw new Error(`Queue command ${index} requires nonempty user text`);
  }
  return command;
}

async function run(ctx, runDir, log, exit) {
  if (process.env.DSH_GC_ALLOW_MODEL_REQUESTS !== "1") {
    throw new Error("Explicit launch required: DSH_GC_ALLOW_MODEL_REQUESTS=1");
  }
  const harness = process.env.DSH_GC_HARNESS;
  if (!harness || !path.isAbsolute(harness)) throw new Error("DSH_GC_HARNESS must be absolute");
  const require = createRequire(path.join(harness,
    "node_modules/.pnpm/node_modules/@deepseek-ai/dsh-headless/package.json"));
  const load = specifier => import(pathToFileURL(require.resolve(specifier)).href);
  const [{ installModelSelection }, { createUserMessage }, { SessionId }] =
    await Promise.all([load("@deepseek-ai/dsh-agent"), load("@deepseek-ai/dsh-llm"),
      load("@deepseek-ai/dsh-session")]);
  await ctx.get("loader")?.await();
  const agents = ctx.get("agents"), sessions = ctx.get("sessions");
  const selection = ctx.get("agentDefaultModel").currentSelection();
  if (selection.provider !== "gc-local" || selection.model !== "kvmem-gc-qwen38") {
    throw new Error("Unexpected default provider/model");
  }
  const { agent } = await agents.create({
    sessionId: SessionId(`session-${randomUUID()}`),
    meta: { cwd: process.cwd() },
    agentOptions: { provider: selection.provider, model: selection.model },
    setup: agentCtx => {
      // setup may return a transaction with commit(); installModelSelection
      // returns a disposer instead. Match the official headless void callback.
      installModelSelection(agentCtx, { current: selection, assembled: undefined });
    },
  });
  await agent.whenIdle();
  const sessionId = agent.session.id;
  writeJSON(path.join(runDir, "session.json"), {
    session_id: sessionId, agent_id: agent.id, pid: process.pid,
    cwd: process.cwd(), dsh_home: process.env.DSH_HOME, provider: selection.provider,
    model: selection.model, started_at: new Date().toISOString(),
  });
  log({ event: "runner.ready", session_id: sessionId });
  for (let index = 1; ; index++) {
    const key = String(index).padStart(6, "0");
    const commandPath = path.join(runDir, "commands", `${key}.json`);
    writeJSON(path.join(runDir, "status.json"), { state: "waiting", session_id: sessionId, next_index: index });
    while (!fs.existsSync(commandPath)) await delay(500);
    const command = validateCommand(JSON.parse(fs.readFileSync(commandPath, "utf8")), index);
    if (command.kind === "close") {
      await sessions.flush(agent.session);
      writeJSON(path.join(runDir, "status.json"), { state: "closed", session_id: sessionId, last_index: index });
      log({ event: "runner.closed", session_id: sessionId, index });
      exit(0);
      return;
    }
    const firstSeq = agent.session.seq;
    const startedAt = new Date().toISOString();
    writeJSON(path.join(runDir, "status.json"), { state: "running", session_id: sessionId, index });
    log({ event: "runner.followup", session_id: sessionId, index, first_seq: firstSeq });
    // User provenance is required by the installed route-specific Query hook.
    agent.followup(createUserMessage({ content: [{ type: "text", text: command.text }], source: { kind: "user" } }));
    await agent.whenIdle();
    await sessions.flush(agent.session);
    if (agent.session.id !== sessionId) throw new Error("Agent changed session during followup");
    const events = [...agent.session.events].filter(event => event.seq >= firstSeq);
    const outcome = summarize(events, firstSeq);
    writeJSON(path.join(runDir, "turns", `${key}.events.json`), events);
    writeJSON(path.join(runDir, "turns", `${key}.result.json`), {
      index, session_id: sessionId, first_seq: firstSeq, last_seq: agent.session.seq,
      started_at: startedAt, finished_at: new Date().toISOString(), ...outcome,
    });
    process.stdout.write(JSON.stringify({ index, session_id: sessionId, ...outcome }) + "\n");
    log({ event: "runner.turn_finished", session_id: sessionId, index, reason: outcome.reason });
    if (outcome.reason?.kind !== "completed") {
      writeJSON(path.join(runDir, "status.json"), { state: "failed", session_id: sessionId, index, reason: outcome.reason });
      exit(1);
      return;
    }
  }
}

export function apply(ctx) {
  const runDir = process.env.DSH_GC_RUN_DIR;
  if (!runDir || !path.isAbsolute(runDir)) throw new Error("DSH_GC_RUN_DIR must be absolute");
  const exit = ctx.get("appExit");
  if (!exit) throw new Error("Launcher appExit is required");
  const log = value => fs.appendFileSync(path.join(runDir, "driver-events.jsonl"),
    JSON.stringify({ at: new Date().toISOString(), ...value }) + "\n", { mode: 0o600 });
  ctx.on("session/event", (session, event) => log({ event: event.type, session_id: session.id, seq: event.seq }));
  ctx.on("agent/error", ({ agent, error }) => log({ event: "agent/error", agent_id: agent.id, error: String(error) }));
  run(ctx, runDir, log, exit).catch(error => {
    log({ event: "runner.error", error: String(error) });
    writeJSON(path.join(runDir, "status.json"), { state: "failed", error: String(error) });
    process.stderr.write(`gc-single-session-runner: ${error.stack ?? error}\n`);
    exit(1);
  });
}
