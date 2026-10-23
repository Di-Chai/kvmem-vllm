# CPU validation of a DSH-generated static game

This observer serves an existing game directory on an ephemeral loopback port and
runs Chromium with SwiftShader software rendering. It does not generate or edit
game source, invoke an LLM, or measure GPU/model throughput. Keep it outside
timed inference benchmarks because CPU work can affect their measurements.

The isolated cloud dependency installation is
/root/data/kvmem-vllm-gc/dsh-harness/browser-validation.
It contains Playwright 1.64.1, pngjs 7.0.0 and the matching official Chrome Headless Shell.
The installation does not change the DSH dependency lock. Reproduction:

    npm install --save-exact playwright@1.64.1 pngjs@7.0.0
    PLAYWRIGHT_BROWSERS_PATH="$PWD/browsers" npx playwright install --only-shell chromium

On the prepared cloud machine:

    python3 /root/data/kvmem-vllm-gc/repo/diagnostics/dsh_validation/run.py \
      --root /root/data/kvmem-vllm-gc/runs/dsh-live-01/csgo/workspace \
      --entry index.html \
      --runtime-root /tmp/kvmem-gc-browser-runtime-1248 \
      --output /root/data/kvmem-vllm-gc/runs/dsh-live-01/csgo/browser-validation-01

The launcher selects the isolated Node binary, software-only browser settings,
local shared-library search path and local font configuration. Missing Chromium
shared libraries on this machine were downloaded as Ubuntu packages and unpacked
under browser-validation/sysroot; no system packages or services were installed.
The local temporary runtime avoids initial executable mmap stalls on the Ceph
HDD volume mounted at /root/data. It is copied from the same verified artifacts.
Exact package versions and browser download hash are recorded in that tool
directory. The official Chrome Headless Shell 156.0.8078.4 ZIP was checked against
its upstream MD5/ETag before extraction.

The source directory must already contain the model's actual artifacts. If the
model chose a subdirectory, pass that directory or adjust --entry. The output
must be outside the source tree. It records the browser/renderer, hashes of
served files, HTTP failures, console errors, uncaught page errors, DOM/canvas
metadata, delivered input events, before/after screenshots, and pixel deltas.

The default sequence observes idle animation, tries a visible start/play button,
focuses the game, holds W and D, clicks to fire, and presses R. These are
observations, not proof that the game moves or shoots. In particular, animation
can change screenshots without input having any effect. The report explicitly
sets specified_state_checks_pass to null when no state checks were supplied.

After inspecting the generated code, pass --plan a JSON file that provides a
read-only probe_expression appropriate to the actual artifact, plus explicit
checks. No framework or state variable is assumed. Example plan for an artifact
that really exposes those state fields:

    {
      "probe_expression": "({x: window.game.player.x, ammo: window.game.ammo})",
      "actions": [
        {"name": "idle-control", "type": "wait", "ms": 700},
        {"name": "focus", "type": "focus-game"},
        {"name": "move", "type": "key-hold", "key": "d", "ms": 700,
         "checks": [{"path": "x", "op": "changed"}]},
        {"name": "fire", "type": "mouse-click", "button": "left",
         "checks": [{"path": "ammo", "op": "decreased"}]}
      ]
    }

Probe expressions are trusted developer input and evaluated in the page;
keep them read-only. The script never injects game-state accessors into source.
Supported check operations: changed, decreased, increased, equals. Paths are
dot-separated keys relative to the probe result. Missing values fail checks.
A passing check means only that the specified property behaved as asserted.
Judge whether it establishes movement/shooting from the game's source and
compare the idle-control step. Do not label all gameplay correct from a single
property change.

Additional actions: click (selector), key-press (key), mouse-move (x/y/steps).
Optional viewport, initial_wait_ms, hold_ms and per-action settle_ms are supported.
All unspecified delays are bounded. Page exceptions cause a failed report.
HTTP 4xx/5xx responses are recorded separately as http_errors (URL and status),
and fail the no_http_errors gate even if the browser reports no transport failure
or console error. Favicon or other missing assets are not silently ignored.

A minimal local 404 script-resource regression check is available:

    python3 diagnostics/dsh_validation/test_http_errors.py \
      --runtime-root /tmp/kvmem-gc-browser-runtime-1248 \
      --output /root/data/kvmem-vllm-gc/runs/browser-validator-http404-01

It creates an infrastructure fixture outside any model artifact directory, fetches
a missing .js resource, and checks that transport success with HTTP 404 still fails
the report. It never sends a model request.

External requests are blocked by default, including all model/API endpoints.
If the actual source depends on a known CDN, an explicitly reviewed plan may set
allowed_asset_hosts; only GET script/stylesheet/image/font/media resources on
those hosts are permitted. Fetch/XHR, WebSockets and external POST are not
permitted by that exception. A blocked dependency is a visible validation issue,
not evidence that the model's code itself is broken.

Artifacts must be assessed separately from DSH task continuity and GC quality.
The current GC task has no real semantic signal producer: a browser pass cannot
qualify semantic GC safety, decode overhead, or long-horizon completion.

Form observers may also use these explicit actions:

- `fill`: `{"type":"fill","selector":"#description","value":"Lunch"}`.
  The string replaces an input/textarea/contenteditable value; an empty string
  clears it. Playwright delivers the input behavior, rather than typing every
  key. It does not submit a form. Use a separate click or key-press to submit.
- `select-option`: `{"type":"select-option","selector":"#category","value":"food"}`.
  Selects a native HTML select option by its exact value, not its visible label.
  The selected values are recorded. Custom dropdowns still use their actual
  click/key interactions; the observer does not invent application APIs.
- `reload`: `{"type":"reload"}`. Reloads the current URL, waits for load, then
  observes the page after settle_ms. It keeps the same origin and browser
  context, so localStorage survives. The response status is recorded and all
  HTTP/console/page/network error gates remain active across reloads.

Fill and select-option require one matching element and a string value.
Missing, ambiguous, disabled, or otherwise invalid targets cause an action error
and a failed report. Input/change events are recorded without reading field
values into the event log; probes explicitly choose which values to observe.

To check persistence, first fill/select, submit using the application's UI, and
assert the expected visible result. Then reload and assert the result again
with read-only DOM or storage probes. An action completing without errors does
not establish successful saving. A reload check proves persistence only within
this browser context and origin; each observer run uses a fresh context and
ephemeral server port. It does not prove persistence across browser restarts.
Per-page input event history resets on reload; each action retains its before
and after snapshots.

A CPU browser infrastructure regression uses a minimal form fixture outside
model artifacts. It checks replacement/clearing, native selection, submission,
DOM restoration from localStorage after reload, a deliberately broken restore
negative, and an invalid selector negative:

    python3 diagnostics/dsh_validation/test_form_actions.py \
      --runtime-root /tmp/kvmem-gc-browser-runtime-1248 \
      --output /root/data/kvmem-vllm-gc/runs/browser-validator-forms-01

The fixture is test infrastructure, not a generated task application. Continue
running test_http_errors.py when changing action/navigation behavior, because
HTTP errors must fail the report even when an interaction or reload succeeds.

A confirmation dialog is handled only when its click explicitly declares it:

    {"type":"click","selector":"#delete",
     "expect_dialog":{"type":"confirm","accept":true},
     "checks":[{"path":"rows","op":"decreased"}]}

The optional `expect_dialog` accepts only `type:"confirm"` and a boolean
`accept`. It is supported only on `click`. The observer waits up to three
seconds from the click's start for the expected dialog, records its type,
message, decision and whether handling succeeded, and removes the listener
after that action, including failures. A missing dialog, wrong type, duplicate
dialog or handling failure fails the action. Unexpected types and additional
dialogs are dismissed, never accepted. Use `accept:false` to test cancellation.
Clicks without `expect_dialog` keep Playwright's normal auto-dismiss behavior;
there is no global auto-accept. Successful acceptance alone does not prove a
deletion: assert the visible row/count state after the click.

CPU fixture coverage includes explicit acceptance/cancellation, unchanged
default dismissal before and after scoped handlers, missing/wrong/duplicate
dialog failures and invalid decisions:

    python3 diagnostics/dsh_validation/test_dialog_actions.py \
      --runtime-root /tmp/kvmem-gc-browser-runtime-1248 \
      --output /root/data/kvmem-vllm-gc/runs/browser-validator-dialogs-01
