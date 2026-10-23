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
Favicon or other missing assets are recorded rather than silently ignored.

External requests are blocked by default, including all model/API endpoints.
If the actual source depends on a known CDN, an explicitly reviewed plan may set
allowed_asset_hosts; only GET script/stylesheet/image/font/media resources on
those hosts are permitted. Fetch/XHR, WebSockets and external POST are not
permitted by that exception. A blocked dependency is a visible validation issue,
not evidence that the model's code itself is broken.

Artifacts must be assessed separately from DSH task continuity and GC quality.
The current GC task has no real semantic signal producer: a browser pass cannot
qualify semantic GC safety, decode overhead, or long-horizon completion.
