# Model-produced expense tracker after real KV GC

The three web files were produced and edited by Qwen3.8-27B-GPTQ-Int4 through DSH, without human code repairs. The same session completed creation and a later search-feature task after 111 historical KV blocks were retired and their CPU payloads released.

The independent CPU Chromium run executed 48 actions and passed 86 state checks. browser-results.json retains every action, before/after probe and assertion; the original full browser report is retained on the cloud machine, with its SHA256 in this bundle. search-verified.png is an unedited screenshot.

Monthly summary cards use the current calendar month. Month/keyword filters affect the record list, and clearing search also resets the month filter. The recorded test used September/October 2026 and ran on 2026-10-24; replay on another calendar month requires adjusting test dates or fixing the browser clock.

Run a local static server from this directory, for example python3 -m http.server 8000, then visit http://127.0.0.1:8000/. No external dependency is required. Chinese/emoji rendering requires suitable system fonts; the qualification browser runtime was supplied with Noto Sans CJK and Noto Color Emoji.
