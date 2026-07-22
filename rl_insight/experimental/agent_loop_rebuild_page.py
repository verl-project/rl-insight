# Copyright (c) 2026 verl-project authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Landing page for the Grafana dashboard «Rebuild Agent Loop» link.

Opened from the dashboard header (near time range / Refresh) with
``?from=${__from}&to=${__to}&auto=1``. Grafana passes epoch **milliseconds**.
"""

from __future__ import annotations

REBUILD_PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Rebuild Agent Loop</title>
<style>
  :root { color-scheme: light; --bg:#f6f4ef; --ink:#1a1a1a; --muted:#5c5c5c; --accent:#0b6e4f; --card:#fff; --line:#ddd; --ok:#0b6e4f; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }
  main { max-width: 720px; margin: 40px auto; padding: 0 20px; }
  h1 { font-size: 1.35rem; margin: 0 0 0.35rem; }
  p.lead { color: var(--muted); margin: 0 0 1.25rem; line-height: 1.45; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 1.25rem; }
  label { display:block; font-size: 0.85rem; color: var(--muted); margin-bottom: 0.35rem; }
  .row { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
  input[type=datetime-local], input[type=text] {
    width:100%; padding: 0.55rem 0.65rem; border:1px solid var(--line); border-radius: 6px; font: inherit;
  }
  .presets { display:flex; flex-wrap:wrap; gap:8px; margin: 0 0 14px; }
  .presets button, .go {
    border:1px solid var(--line); background:#fff; border-radius:999px; padding:0.4rem 0.85rem;
    cursor:pointer; font: inherit;
  }
  .go { background: var(--accent); color:#fff; border-color: var(--accent); border-radius: 8px; padding: 0.65rem 1.1rem; }
  .go:disabled { opacity: 0.6; cursor: wait; }
  pre { background:#111; color:#e8e8e8; padding: 12px; border-radius: 8px; overflow:auto; font-size: 0.8rem; min-height: 4rem; }
  .banner { padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1rem; background: #e7f5ef; color: var(--ok); display:none; }
  .banner.show { display:block; }
  .hint { font-size: 0.85rem; color: var(--muted); margin-top: 1rem; line-height: 1.4; }
</style>
</head>
<body>
<main>
  <h1>Rebuild Agent Loop</h1>
  <p class="lead">
    由 Grafana 仪表盘顶栏链接打开（时间窗 / Refresh 旁的
    <strong>Rebuild Agent Loop</strong>）。按<strong>当前仪表盘时间窗</strong>
    从 Tempo 重建嵌套轨迹树；窗内每个 run 各自可展开。
  </p>
  <div id="banner" class="banner"></div>
  <div class="card">
    <div class="presets" id="presets">
      <button type="button" data-hours="0.25">Last 15m</button>
      <button type="button" data-hours="1">Last 1h</button>
      <button type="button" data-hours="3">Last 3h</button>
      <button type="button" data-hours="6">Last 6h</button>
      <button type="button" data-hours="24">Last 24h</button>
    </div>
    <div class="row">
      <div>
        <label for="fromLocal">From</label>
        <input id="fromLocal" type="datetime-local"/>
      </div>
      <div>
        <label for="toLocal">To</label>
        <input id="toLocal" type="datetime-local"/>
      </div>
    </div>
    <div style="margin-bottom:14px">
      <label for="runId">run_id（可选，留空 = 窗内全部 run）</label>
      <input id="runId" type="text" placeholder="export-…"/>
    </div>
    <button class="go" id="go" type="button">Rebuild dashboard</button>
    <pre id="out">Ready.</pre>
    <p class="hint">
      完成后回到 Grafana，对仪表盘 <strong>硬刷新</strong>（约 10s 内 provisioning 生效）。
      Grafana 原生 Refresh 只会重查已有面板，不会重建 Sample/Session 嵌套行。
    </p>
  </div>
</main>
<script>
/** Grafana ${__from}/${__to} are epoch ms; also accept unix seconds. */
function toUnixSeconds(raw) {
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) return null;
  return n > 1e12 ? Math.floor(n / 1000) : Math.floor(n);
}
function toLocalValue(d) {
  const pad = n => String(n).padStart(2, "0");
  return d.getFullYear() + "-" + pad(d.getMonth()+1) + "-" + pad(d.getDate())
    + "T" + pad(d.getHours()) + ":" + pad(d.getMinutes());
}
function setPreset(hours) {
  const to = new Date();
  const from = new Date(to.getTime() - hours * 3600 * 1000);
  document.getElementById("fromLocal").value = toLocalValue(from);
  document.getElementById("toLocal").value = toLocalValue(to);
}
document.querySelectorAll("#presets button").forEach(btn => {
  btn.addEventListener("click", () => setPreset(parseFloat(btn.dataset.hours)));
});
setPreset(3);

const params = new URLSearchParams(location.search);
const fromSec = toUnixSeconds(params.get("from"));
const toSec = toUnixSeconds(params.get("to"));
if (fromSec && toSec) {
  document.getElementById("fromLocal").value = toLocalValue(new Date(fromSec * 1000));
  document.getElementById("toLocal").value = toLocalValue(new Date(toSec * 1000));
}
if (params.get("run_id")) document.getElementById("runId").value = params.get("run_id");

async function doRebuild() {
  const go = document.getElementById("go");
  const out = document.getElementById("out");
  const banner = document.getElementById("banner");
  const fromMs = Date.parse(document.getElementById("fromLocal").value);
  const toMs = Date.parse(document.getElementById("toLocal").value);
  if (isNaN(fromMs) || isNaN(toMs)) {
    out.textContent = "Invalid from/to";
    return;
  }
  const body = {
    from: Math.floor(fromMs / 1000),
    to: Math.floor(toMs / 1000),
  };
  const rid = document.getElementById("runId").value.trim();
  if (rid) body.run_id = rid;
  go.disabled = true;
  out.textContent = "Rebuilding for dashboard time range…";
  banner.classList.remove("show");
  try {
    const resp = await fetch("/api/v1/agent-loop/rebuild", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    const text = await resp.text();
    let data;
    try { data = JSON.parse(text); } catch { data = { raw: text }; }
    out.textContent = JSON.stringify(data, null, 2);
    if (resp.ok) {
      const n = (data.runs && data.runs.length) || data.runs || 0;
      banner.textContent = "Done · " + n + " run(s) expanded. Return to Grafana and hard-refresh the dashboard.";
      banner.classList.add("show");
    }
  } catch (err) {
    out.textContent = String(err);
  } finally {
    go.disabled = false;
  }
}

document.getElementById("go").addEventListener("click", doRebuild);
if (params.get("auto") === "1" && fromSec && toSec) {
  setTimeout(doRebuild, 150);
}
</script>
</body>
</html>
"""
