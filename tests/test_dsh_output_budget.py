"""CPU-only exact-token bridge checks near logical limits and with G=16384."""
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from starlette.requests import Request

from benchmarks.deepswe.claude_bridge import Planner, Trajectory
import scripts.dsh_bridge as bridge
from scripts.dsh_model_protocol import TokenizerProtocol
from test_dsh_model_protocol import wire_body, tool_output


@pytest.fixture(scope="module")
def assets():
    directory = os.environ.get("KVMEM_QWEN38_TOKENIZER")
    if not directory:
        pytest.skip("Set KVMEM_QWEN38_TOKENIZER to audited local Qwen3.8 assets")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(directory, local_files_only=True), directory


def proof_body(body):
    body = copy.deepcopy(body)
    hashes = [hashlib.sha256(m["content"].encode()).hexdigest()
              for m in body["messages"] if m["role"] == "user"]
    body["kvmem_query_provenance"] = dict(
        schema_version=1, session_id=body["prompt_cache_key"],
        query_user_ordinal=0, query_message_id="budget-human-query",
        user_messages_sha256=hashlib.sha256(json.dumps(hashes, separators=(",", ":")).encode()).hexdigest())
    return body


async def run_bridge(assets, monkeypatch, directory, raw, *, remaining=None,
                     cap=16384, segments=1, streaming=False, pinned=False):
    tokenizer, path = assets
    current = TokenizerProtocol(tokenizer, path, enable_thinking=False,
        pin_system=pinned, select_budget=36864 if pinned else None, gen_budget=16384)
    body = proof_body(wire_body())
    body.update(max_tokens=cap, stream=streaming)
    prepared = Planner(current, tokenizer, 16384).prepare(bridge.to_anthropic(body), Trajectory())
    original = prepared[0]
    accepted = tokenizer.encode(raw, add_special_tokens=False)
    context = 262144 if remaining is None else len(original) + remaining
    calls = []

    async def handle(request):
        data = json.loads(request.content)
        used = len(data["prompt"]) - len(original)
        assert data["prompt"] == original + accepted[:used]
        assert 0 < data["max_tokens"] <= min(16384, context - len(data["prompt"]))
        assert "thinking_token_budget" not in data
        calls.append(data)
        ids = accepted[used:used + data["max_tokens"]]
        finish = "stop" if used + len(ids) == len(accepted) else "length"
        # Continuations must decode authoritative IDs, not supplied text.
        text = tokenizer.decode(ids) if segments == 1 else "not authoritative"
        events = [
            dict(id="cpu-budget", choices=[dict(index=0, text=text, token_ids=ids, finish_reason=finish)]),
            dict(choices=[], usage=dict(prompt_tokens=len(data["prompt"]), completion_tokens=len(ids))),
        ]
        return httpx.Response(200, content="".join(
            "data: " + json.dumps(e, ensure_ascii=False) + "\n\n" for e in events) + "data: [DONE]\n\n")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(bridge.httpx, "AsyncClient", lambda **_: client)
    args = SimpleNamespace(tokenizer_protocol=True, tokenizer=path, disable_thinking=True,
        pin_system=pinned, select_budget=36864, require_query_provenance=True,
        gen_budget=16384, max_output_segments=segments, max_inflight=1,
        output=directory, backend="http://cpu-fake", model="local",
        max_model_len=context, max_sessions=4)
    app = bridge.create_app(args, tokenizer)
    endpoint = next(r.endpoint for r in app.routes if r.path == "/v1/chat/completions")

    async def receive():
        return dict(type="http.request", body=json.dumps(body).encode(), more_body=False)

    try:
        result = await endpoint(Request(dict(type="http", method="POST",
            path="/v1/chat/completions", headers=[]), receive))
        if streaming:
            result = "".join([piece async for piece in result.body_iterator])
    finally:
        await client.aclose()
    return SimpleNamespace(result=result, calls=calls, original=original, accepted=accepted,
                           protocol=current, body=body, context=context)


@pytest.mark.parametrize("streaming", [False, True])
def test_first_segment_uses_positive_remainder_without_truncating_input(assets, monkeypatch, tmp_path, streaming):
    async def check():
        raw = "OK<|im_end|>"
        expected = assets[0].encode(raw, add_special_tokens=False)
        run = await run_bridge(assets, monkeypatch, tmp_path, raw, remaining=len(expected), streaming=streaming)
        assert len(run.calls) == 1 and run.calls[0]["max_tokens"] == len(expected) < 16384
        assert run.calls[0]["prompt"] == run.original
        saved = json.loads(next(tmp_path.glob("*/00001/accepted.json")).read_text())
        assert saved["token_ids"] == expected
        plan = json.loads(next(tmp_path.glob("*/00001/plan.json")).read_text())
        assert plan["requested_output_cap"] == 16384
        assert plan["effective_output_cap"] == plan["logical_remaining_tokens"] == len(expected)
        if streaming:
            assert "data: [DONE]" in run.result and '"finish_reason": "stop"' in run.result
        else:
            assert run.result["choices"][0]["message"]["content"] == "OK"
            assert run.result["usage"]["completion_tokens"] == len(expected)
    asyncio.run(check())


@pytest.mark.parametrize("remaining", [0, -1])
def test_exhausted_logical_context_rejects_without_upstream_or_commit(assets, monkeypatch, tmp_path, remaining):
    async def check():
        run = await run_bridge(assets, monkeypatch, tmp_path, "OK<|im_end|>", remaining=remaining)
        assert run.result.status_code == 502
        assert "no output space" in run.result.body.decode()
        assert run.calls == [] and not list(tmp_path.glob("**/accepted.json"))
    asyncio.run(check())


def test_short_positive_space_can_return_length_for_text(assets, monkeypatch, tmp_path):
    async def check():
        run = await run_bridge(assets, monkeypatch, tmp_path, "One two three four.<|im_end|>", remaining=1)
        assert run.calls[0]["max_tokens"] == 1
        assert run.result["choices"][0]["finish_reason"] == "length"
        assert run.result["usage"]["completion_tokens"] == 1
        saved = json.loads(next(tmp_path.glob("*/00001/accepted.json")).read_text())
        assert saved["token_ids"] == run.accepted[:1]
    asyncio.run(check())


@pytest.mark.parametrize("streaming", [False, True])
def test_short_first_segment_only_exposes_a_complete_tool(assets, monkeypatch, tmp_path, streaming):
    async def check():
        raw = tool_output().split("</think>\n", 1)[1]
        count = len(assets[0].encode(raw, add_special_tokens=False))
        run = await run_bridge(assets, monkeypatch, tmp_path, raw, remaining=count, streaming=streaming)
        assert len(run.calls) == 1 and run.calls[0]["max_tokens"] == count < 16384
        assert json.loads(next(tmp_path.glob("*/00001/accepted.json")).read_text())["token_ids"] == run.accepted
        if streaming:
            assert "data: [DONE]" in run.result and '"finish_reason": "tool_calls"' in run.result
        else:
            tool, = run.result["choices"][0]["message"]["tool_calls"]
            arguments = json.loads(tool["function"]["arguments"])
            assert arguments["path"] == "中文.txt" and arguments["content"] == "café 和 🙂\n第二行"
    asyncio.run(check())


@pytest.mark.parametrize("streaming", [False, True])
def test_context_truncated_tool_fails_without_accepted_commit(assets, monkeypatch, tmp_path, streaming):
    async def check():
        raw = tool_output().split("</think>\n", 1)[1]
        count = len(assets[0].encode(raw, add_special_tokens=False)) // 2
        run = await run_bridge(assets, monkeypatch, tmp_path, raw, remaining=count, streaming=streaming)
        assert len(run.calls) == 1 and run.calls[0]["max_tokens"] == count
        assert not list(tmp_path.glob("**/accepted.json"))
        assert not list(tmp_path.glob("**/response.json"))
        native = json.loads(next(tmp_path.glob("*/00001/native.json")).read_text())
        assert native["token_ids"] == run.accepted[:count] and native["finish_reason"] == "length"
        if streaming:
            assert '"error"' in run.result and "data: [DONE]" not in run.result
            chunks = [json.loads(line[6:]) for line in run.result.splitlines() if line.startswith("data: {")]
            for item in chunks:
                for choice in item.get("choices", []):
                    assert not choice.get("finish_reason")
                    for tool in choice.get("delta", {}).get("tool_calls", []):
                        assert not tool.get("function", {}).get("arguments")
        else:
            assert run.result.status_code == 502 and "tool call" in run.result.body.decode().lower()
    asyncio.run(check())


def test_16k_native_segments_keep_ids_and_shorten_final_complete_tool(assets, monkeypatch, tmp_path):
    async def check():
        raw = tool_output().split("</think>\n", 1)[1].replace(
            "café 和 🙂\n第二行", "x " * 16384 + "中文🙂")
        count = len(assets[0].encode(raw, add_special_tokens=False))
        assert 16384 < count < 32768
        run = await run_bridge(assets, monkeypatch, tmp_path, raw,
                               remaining=count, cap=32768, segments=2, streaming=True)
        assert [c["max_tokens"] for c in run.calls] == [16384, count - 16384]
        assert run.calls[1]["prompt"] == run.original + run.accepted[:16384]
        assert "data: [DONE]" in run.result and '"finish_reason": "tool_calls"' in run.result
        assert json.loads(next(tmp_path.glob("*/00001/accepted.json")).read_text())["token_ids"] == run.accepted
        assert len(list(tmp_path.glob("**/segment-*.json"))) == 2
        response = json.loads(next(tmp_path.glob("*/00001/response.json")).read_text())
        tool, = response["choices"][0]["message"]["tool_calls"]
        assert json.loads(tool["function"]["arguments"])["content"] == "x " * 16384 + "中文🙂"
    asyncio.run(check())


def test_large_pin_budget_uses_actual_g_and_restore_rejects_cross_g(assets, monkeypatch, tmp_path):
    async def check():
        run = await run_bridge(assets, monkeypatch, tmp_path, "OK<|im_end|>", pinned=True)
        assert run.protocol.system_pin_policy["gen_budget"] == 16384
        assert run.calls[0]["max_tokens"] == 16384
        proof = bridge.validate_system_pin_budget(run.calls[0], run.protocol)
        assert proof["declared_select_budget"] == 36864
        assert proof["required_select_tokens"] < 36864
        with pytest.raises(ValueError, match="declared generation"):
            bridge.validate_system_pin_budget(dict(run.calls[0], max_tokens=16385), run.protocol)
        owner = run.calls[0]["session_id"]
        planner = Planner(run.protocol, assets[0], 16384)
        state = bridge.restore_trajectory(planner, owner, [tmp_path])
        assert state.output == run.accepted and state.turns == 1
        other = TokenizerProtocol(assets[0], assets[1], enable_thinking=False,
                                  pin_system=True, select_budget=36864, gen_budget=4096)
        with pytest.raises(ValueError, match="protocol differs"):
            bridge.restore_trajectory(Planner(other, assets[0], 4096), owner, [tmp_path])
    asyncio.run(check())
