"""Offline checks for the isolated DSH adapter installer; no provider requests."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "install_dsh_query_hook", REPO / "scripts/install_dsh_query_hook.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)

ORIGINAL = b"""function requestHeaders(headers) { return headers; }
export function streamOptions(profile, options, model) {
  return {
    headers: requestHeaders(profile.headers)
  };
}
"""
ROUTE = {
    "provider": "gc-local",
    "model": "kvmem-gc-qwen38",
    "baseURL": "http://127.0.0.1:18387/v1",
}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    root = tmp_path / "dsh-harness"
    cli = root / "node_modules/@deepseek-ai/dsh"
    cli.mkdir(parents=True)
    (cli / "package.json").write_text(json.dumps({
        "name": "@deepseek-ai/dsh", "version": installer.EXPECTED_CLI_VERSION}))
    package = root / "node_modules/@deepseek-ai/dsh-llm-pi-ai"
    lib = package / "lib"
    lib.mkdir(parents=True)
    (package / "package.json").write_text(json.dumps({
        "name": "@deepseek-ai/dsh-llm-pi-ai",
        "version": installer.EXPECTED_ADAPTER_VERSION, "type": "module"}))
    index = lib / "index.js"
    index.write_bytes(ORIGINAL)
    monkeypatch.setattr(installer, "ALLOWED_HARNESS_ROOT", root)
    monkeypatch.setattr(installer, "EXPECTED_ADAPTER_SHA256", installer.digest(ORIGINAL))
    return root, index


def snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def test_dry_run_and_idempotent_install(harness):
    root, index = harness
    before = snapshot(root)
    preview = installer.install(root, ROUTE, dry_run=True)
    assert preview["status"] == "would_install"
    assert preview["dry_run"] and preview["files_to_write"] > 0
    assert snapshot(root) == before

    result = installer.install(root, ROUTE)
    assert result["status"] == "installed"
    assert (root / installer.BACKUP_NAME).read_bytes() == ORIGINAL
    installed = snapshot(root)
    again = installer.install(root, {**ROUTE, "baseURL": ROUTE["baseURL"] + "/"})
    assert again["status"] == "unchanged" and again["files_to_write"] == 0
    assert snapshot(root) == installed


def test_original_hash_mismatch_refuses_all_writes(harness):
    root, index = harness
    index.write_bytes(ORIGINAL + b"// unexpected upstream change\n")
    before = snapshot(root)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        installer.install(root, ROUTE)
    assert snapshot(root) == before


def test_target_and_symlink_escape_are_rejected(harness, tmp_path):
    root, index = harness
    outside = tmp_path / "other-dsh"
    outside.mkdir()
    with pytest.raises(ValueError, match="isolated cloud"):
        installer.install(outside, ROUTE)
    sentinel = tmp_path / "outside.js"
    sentinel.write_bytes(ORIGINAL)
    index.unlink()
    index.symlink_to(sentinel)
    with pytest.raises(ValueError, match="non-regular"):
        installer.install(root, ROUTE)
    assert sentinel.read_bytes() == ORIGINAL
    assert not (root / installer.STATE_NAME).exists()


def test_modified_install_or_changed_route_is_not_overwritten(harness):
    root, index = harness
    installer.install(root, ROUTE)
    before = snapshot(root)
    with pytest.raises(ValueError, match="different route/source"):
        installer.install(root, {**ROUTE, "model": "another-model"})
    assert snapshot(root) == before
    hook = index.parent / installer.HOOK_NAME
    hook.write_bytes(hook.read_bytes() + b"// unexpected mutation\n")
    modified = snapshot(root)
    with pytest.raises(ValueError, match="was modified"):
        installer.install(root, ROUTE)
    assert snapshot(root) == modified


def test_interrupted_manifest_write_can_resume_exact_install(harness):
    root, index = harness
    installer.install(root, ROUTE)
    state = root / installer.STATE_NAME
    state.unlink()
    before = snapshot(root)
    result = installer.install(root, ROUTE)
    assert result["files_to_write"] == 1
    after = snapshot(root)
    assert after.pop(installer.STATE_NAME)
    assert after == before


@pytest.mark.parametrize("route", [
    {"provider": "", "model": "m", "baseURL": "http://127.0.0.1:18387/v1"},
    {"provider": "p", "model": "m"},
    {**ROUTE, "baseURL": "file:///tmp/provider"},
    {**ROUTE, "baseURL": "http://user:password@127.0.0.1:18387/v1"},
    {**ROUTE, "baseURL": ROUTE["baseURL"] + "?token=not-a-credential"},
    {**ROUTE, "baseURL": "http://127.0.0.1:99999/v1"},
])
def test_invalid_route_fails_without_mutation(harness, route):
    root, _ = harness
    before = snapshot(root)
    with pytest.raises(ValueError):
        installer.install(root, route)
    assert snapshot(root) == before


def test_patched_adapter_attaches_proof_only_to_explicit_route(harness):
    root, index = harness
    node = os.environ.get("KVMEM_DSH_TEST_NODE") or shutil.which("node")
    if not node:
        pytest.skip("Node is required for the offline adapter integration check")
    installer.install(root, ROUTE)
    script = """
import assert from 'node:assert/strict';
const {streamOptions} = await import(process.argv[1]);
const route = JSON.parse(process.argv[2]);
const model = {id:route.model, baseUrl:route.baseURL};
const options = {provider:route.provider, sessionId:'session-1', messages:[
  {id:'query-1', role:'user', source:{kind:'user'}, content:[{type:'text',text:'任务'}]}
]};
const configured = streamOptions({headers:{}}, options, model);
const body = {prompt_cache_key:'session-1', temperature:0.6,
              messages:[{role:'user',content:'任务'}]};
const before = JSON.stringify(body.messages);
configured.onPayload(body);
assert.equal(body.kvmem_query_provenance.query_message_id, 'query-1');
assert.equal(body.kvmem_query_provenance.session_id, 'session-1');
assert.equal(JSON.stringify(body.messages), before);
assert.equal(body.temperature, 0.6);
const other = streamOptions({headers:{}}, {...options,provider:'other'}, model);
assert.equal(other.onPayload, undefined);
const wrongEndpoint = streamOptions({headers:{}}, options, {...model,baseUrl:'http://127.0.0.1:18388/v1'});
assert.equal(wrongEndpoint.onPayload, undefined);
const bad = {prompt_cache_key:'another-session', messages:[{role:'user',content:'任务'}]};
assert.throws(() => configured.onPayload(bad), /session identity/);
assert.equal(bad.kvmem_query_provenance, undefined);
"""
    result = subprocess.run(
        [node, "--input-type=module", "-e", script, index.as_uri(), json.dumps(ROUTE)],
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
