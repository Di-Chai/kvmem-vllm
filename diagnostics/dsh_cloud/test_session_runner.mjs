// No server, model, GPU, Agent implementation, or tool executor is started here.
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { apply } from "./session_runner.mjs";

async function simulate({ failFirst = false, allow = true, invalid = false,
  turnTimeoutMs, turnDelayMs = 5, followupThrows = false, observeAfterMs = 0, cancelReportsCompleted = false } = {}) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "dsh-gc-driver-test-"));
  fs.mkdirSync(path.join(directory, "commands"));
  fs.mkdirSync(path.join(directory, "turns"));
  const commands = [
    { index: 1, kind: "user", text: "initial task" },
    { index: 2, kind: "user", text: "repair the existing task" },
    { index: 3, kind: "close" },
  ];
  if (invalid) commands[0].index = 99;
  commands.forEach((command, index) => fs.writeFileSync(
    path.join(directory, "commands", `${String(index + 1).padStart(6, "0")}.json`), JSON.stringify(command)));
  const previousDir = process.env.DSH_GC_RUN_DIR;
  const previousAllow = process.env.DSH_GC_ALLOW_MODEL_REQUESTS;
  const previousTimeout = process.env.DSH_GC_TURN_TIMEOUT_MS;
  if (turnTimeoutMs === undefined) delete process.env.DSH_GC_TURN_TIMEOUT_MS;
  else process.env.DSH_GC_TURN_TIMEOUT_MS = String(turnTimeoutMs);
  process.env.DSH_GC_RUN_DIR = directory;
  if (allow) process.env.DSH_GC_ALLOW_MODEL_REQUESTS = "1";
  else delete process.env.DSH_GC_ALLOW_MODEL_REQUESTS;
  let createCount = 0, flushCount = 0, busy = false, pending = Promise.resolve();
  let resolveExit;
  const exited = new Promise(resolve => { resolveExit = resolve; });
  const messages = [], cancellations = [], session = { id: "same-session", seq: 0, events: [] };
  const event = (type, data) => session.events.push({ type, seq: session.seq++, data });
  let activeTimer, settle;
  const agent = {
    id: "same-agent", session,
    whenIdle: () => pending,
    followup(message) {
      if (followupThrows) throw new Error("mock followup failure");
      assert.equal(busy, false, "a followup overlapped an unfinished turn");
      busy = true;
      messages.push(message);
      event("turn/start", {});
      pending = new Promise(resolve => {
        settle = reason => {
          if (!busy) return;
          event("turn/end", { reason });
          busy = false;
          resolve();
        };
        activeTimer = setTimeout(() => {
          event("assistant/message", { message: { content: [{ type: "text", text: `answer ${messages.length}` }] } });
          settle(failFirst && messages.length === 1
            ? { kind: "error", error: { code: "mock", message: "mock failure" } }
            : { kind: "completed" });
        }, turnDelayMs);
      });
    },
    cancel(cause) {
      cancellations.push(cause);
      clearTimeout(activeTimer);
      settle?.(cancelReportsCompleted ? { kind: "completed" } : { kind: "aborted", cause });
    },
  };
  const services = {
    appExit: code => resolveExit(code),
    agentDefaultModel: { currentSelection: () => ({ provider: "gc-local", model: "kvmem-gc-qwen38" }) },
    agents: { create: async options => {
      createCount++;
      const setupResult = await options.setup({ on: () => () => {} });
      assert.equal(setupResult, undefined, "setup must not return the model-selection disposer as a commit transaction");
      return { agent };
    } },
    sessions: { flush: async current => { assert.equal(current, session); flushCount++; } },
  };
  try {
    apply({ get: key => services[key], on: () => {} });
    const code = await exited;
    if (observeAfterMs) await new Promise(resolve => setTimeout(resolve, observeAfterMs));
    return { code, createCount, flushCount, messages, cancellations,
      budgets: fs.readdirSync(path.join(directory, "turns")).filter(name => name.endsWith(".budget.json"))
        .map(name => JSON.parse(fs.readFileSync(path.join(directory, "turns", name)))),
      status: JSON.parse(fs.readFileSync(path.join(directory, "status.json"))),
      results: fs.readdirSync(path.join(directory, "turns")).filter(name => name.endsWith(".result.json"))
        .map(name => JSON.parse(fs.readFileSync(path.join(directory, "turns", name)))),
    };
  } finally {
    if (previousDir === undefined) delete process.env.DSH_GC_RUN_DIR;
    else process.env.DSH_GC_RUN_DIR = previousDir;
    if (previousAllow === undefined) delete process.env.DSH_GC_ALLOW_MODEL_REQUESTS;
    else process.env.DSH_GC_ALLOW_MODEL_REQUESTS = previousAllow;
    if (previousTimeout === undefined) delete process.env.DSH_GC_TURN_TIMEOUT_MS;
    else process.env.DSH_GC_TURN_TIMEOUT_MS = previousTimeout;
    clearTimeout(activeTimer);
    fs.rmSync(directory, { recursive: true });
  }
}

test("two serial user tasks retain one Agent/session and user provenance", async () => {
  const run = await simulate();
  assert.equal(run.code, 0);
  assert.equal(run.createCount, 1);
  assert.equal(run.flushCount, 3);
  assert.deepEqual(run.messages.map(message => message.source.kind), ["user", "user"]);
  assert.notEqual(run.messages[0].id, run.messages[1].id);
  assert.deepEqual(run.results.map(result => result.session_id), ["same-session", "same-session"]);
  assert.ok(run.results[1].first_seq >= run.results[0].last_seq);
  assert.deepEqual(run.results.map(result => result.text), ["answer 1", "answer 2"]);
  assert.equal(run.status.state, "closed");
});

test("failed first turn prevents later queued tasks", async () => {
  const run = await simulate({ failFirst: true });
  assert.equal(run.code, 1);
  assert.equal(run.createCount, 1);
  assert.equal(run.messages.length, 1);
  assert.equal(run.status.state, "failed");
});

test("invalid queue index is rejected before followup", async () => {
  const run = await simulate({ invalid: true });
  assert.equal(run.code, 1);
  assert.equal(run.messages.length, 0);
});

test("launch guard prevents even Agent creation", async () => {
  const run = await simulate({ allow: false });
  assert.equal(run.code, 1);
  assert.equal(run.createCount, 0);
  assert.equal(run.messages.length, 0);
});


test("deadline cancels one active turn with explicit budget evidence and flush", async () => {
  const run = await simulate({ turnTimeoutMs: 20, turnDelayMs: 200 });
  assert.equal(run.code, 1);
  assert.equal(run.messages.length, 1, "later queued tasks must not start");
  assert.equal(run.flushCount, 1);
  assert.deepEqual(run.cancellations, [{ kind: "hook", reason: "resource-budget:turn-timeout" }]);
  assert.equal(run.budgets.length, 1);
  const budget = run.budgets[0];
  assert.equal(budget.limit_ms, 20);
  assert.ok(budget.elapsed_ms >= 15);
  assert.deepEqual(run.results[0].reason, budget);
  assert.equal(run.results[0].reason.kind, "resource-budget");
  assert.deepEqual(run.results[0].agent_reason, {
    kind: "aborted", cause: { kind: "hook", reason: "resource-budget:turn-timeout" },
  });
  assert.deepEqual(run.status.reason, budget);
  assert.equal(run.status.state, "failed");
});

test("early completion clears each timer before later turns and idle", async () => {
  const run = await simulate({ turnTimeoutMs: 60, observeAfterMs: 100 });
  assert.equal(run.code, 0);
  assert.equal(run.results.length, 2);
  assert.deepEqual(run.cancellations, []);
  assert.deepEqual(run.budgets, []);
  assert.deepEqual(run.results.map(result => result.reason), [{ kind: "completed" }, { kind: "completed" }]);
});

test("zero timeout preserves normal serial followups without cancellation", async () => {
  const run = await simulate({ turnTimeoutMs: 0, turnDelayMs: 30 });
  assert.equal(run.code, 0);
  assert.deepEqual(run.cancellations, []);
  assert.deepEqual(run.budgets, []);
  assert.equal(run.flushCount, 3);
});

test("followup exception clears timer and still flushes the official session", async () => {
  const run = await simulate({ turnTimeoutMs: 20, followupThrows: true, observeAfterMs: 50 });
  assert.equal(run.code, 1);
  assert.equal(run.flushCount, 1);
  assert.deepEqual(run.cancellations, []);
  assert.deepEqual(run.budgets, []);
  assert.match(run.status.error, /mock followup failure/);
});

test("invalid timer environment rejects before Agent creation", async () => {
  for (const turnTimeoutMs of ["-1", "1.5", "2147483648", "", "NaN", "01"]) {
    const run = await simulate({ turnTimeoutMs });
    assert.equal(run.code, 1);
    assert.equal(run.createCount, 0);
    assert.deepEqual(run.cancellations, []);
  }
});


test("deadline winning a completion race is never reported as natural success", async () => {
  const run = await simulate({ turnTimeoutMs: 20, turnDelayMs: 200, cancelReportsCompleted: true });
  assert.equal(run.code, 1);
  assert.equal(run.results[0].reason.kind, "resource-budget");
  assert.deepEqual(run.results[0].agent_reason, { kind: "completed" });
  assert.equal(run.status.reason.kind, "resource-budget");
});
