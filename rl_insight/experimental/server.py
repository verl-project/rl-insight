#!/usr/bin/env python3

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

"""Timeline visualization server for FileSampleRecord data.

Usage::

    python server.py /path/to/traj/root --port 8080

Then open http://localhost:8080 in a browser.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path
from pathlib import Path as _Path

_project_root = _Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from http.server import HTTPServer, SimpleHTTPRequestHandler  # noqa: E402
from typing import Any  # noqa: E402

from rl_insight.experimental.samples import FileSampleRecord  # noqa: E402

# ---------------------------------------------------------------------------
# Data conversion: FileSampleRecord → timeline JSON
# ---------------------------------------------------------------------------


def build_timeline_data(root_dir: Path) -> dict[str, Any]:
    """Scan root_dir for FileSampleRecord directories and build timeline JSON."""
    samples: dict[str, list[dict[str, Any]]] = {}
    overview: dict[str, dict[str, int]] = {}

    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        uid = child.name
        if not (child / "_index.json").exists():
            continue

        fs = FileSampleRecord.open(str(root_dir), uid)
        mem = fs.load()

        overview[uid] = {
            "session_count": mem.num_sessions,
            "total_turns": 0,
            "total_trajs": mem.num_trajectories,
            "success_count": 0,
        }

        sample_sessions: list[dict[str, Any]] = []
        for session in mem.sessions:
            session_data: dict[str, Any] = {
                "sample": mem.sample_index,
                "session": session.session_index,
                "session_id": session.session_id
                or f"session-{mem.sample_index}-{session.session_index}",
                "turns": [],
                "traj_rewards": {},
                "turn_count": 0,
                "traj_count": session.num_trajectories,
            }

            turn_counter = 0
            for traj in session.trajectories:
                session_data["traj_rewards"][str(traj.trajectory_index)] = (
                    traj.reward_score or 0.0
                )
                if traj.reward_score and traj.reward_score > 0:
                    overview[uid]["success_count"] += 1

                for step in traj.steps:
                    tool_names = [tr.name for tr in step.tool_results]

                    # Determine if this is a tool step or LLM-only step.
                    if tool_names:
                        step_type = "tool"
                    else:
                        step_type = "llm"

                    turn = {
                        "turn": turn_counter,
                        "traj": traj.trajectory_index,
                        "type": step_type,
                        "tools": tool_names,
                        "finish_reason": step.exit_reason or "tool_calls"
                        if not step.done
                        else (traj.exit_reason or "stop"),
                        "content": step.thought[:200] if step.thought else "",
                        "ts": time.time(),
                    }
                    session_data["turns"].append(turn)
                    turn_counter += 1

            session_data["turn_count"] = turn_counter
            overview[uid]["total_turns"] += turn_counter
            sample_sessions.append(session_data)

        samples[uid] = sample_sessions

    return {
        "samples": samples,
        "overview": overview,
        "updated_at": time.time(),
    }


def _classify_tool(name: str) -> str:
    """Map tool name to a display category for color coding."""
    name_lower = name.lower()
    if "bash" in name_lower:
        return "Bash"
    if "read" in name_lower:
        return "Read"
    if "edit" in name_lower or "write" in name_lower or "str_replace" in name_lower:
        return "Edit"
    return name


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class TimelineHandler(SimpleHTTPRequestHandler):
    data_dir: Path | None = None

    def do_GET(self) -> None:
        if self.path == "/api/data":
            self._serve_api()
        elif self.path == "/" or self.path == "/index.html":
            self._serve_html()
        else:
            super().do_GET()

    def _serve_api(self) -> None:
        if self.data_dir is None:
            self._json_response({"error": "no data directory configured"}, 500)
            return
        try:
            data = build_timeline_data(self.data_dir)
            self._json_response(data)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _serve_html(self) -> None:
        html = _load_html()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _json_response(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, default=str)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:
        pass  # suppress access logs


def _load_html() -> str:
    """Return the HTML page template (inline for portability)."""
    return HTML_TEMPLATE


# ---------------------------------------------------------------------------
# HTML template (modified from original timeline.html)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trajectory Timeline</title>
<style>
:root{--bg:#0f1117;--card:#1a1d27;--border:#2a2d3a;--text:#c9cdd4;--dim:#7b8191;--accent:#5b8def;--green:#4ade80;--red:#f87171;--yellow:#fbbf24;--purple:#a78bfa}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:20px 24px}
.header{margin-bottom:14px;display:flex;align-items:center;gap:16px}
.header h1{font-size:20px;font-weight:600}
.header .status{font-size:11px;color:var(--dim)}
.header .status.live{color:var(--green)}
.sample-section{margin-bottom:20px}
.sample-header{display:flex;align-items:center;gap:10px;margin-bottom:8px;cursor:pointer;user-select:none;padding:8px 12px;border-radius:6px;transition:background .15s}
.sample-header:hover{background:rgba(91,141,239,0.06)}
.sample-header .arrow{font-size:11px;color:var(--dim);display:inline-block;transition:transform .2s;width:14px;text-align:center}
.sample-header .arrow.open{transform:rotate(90deg)}
.sample-header .stitle{font-size:15px;font-weight:600}
.sample-header .smeta{font-size:11px;color:var(--dim)}
.sample-header .sresolved{font-size:11px;margin-left:auto}
.sample-header .sresolved.ok{color:var(--green)}
.sample-header .sresolved.fail{color:var(--red)}
.session-bar{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 8px 26px}
.session-tab{padding:5px 12px;border:1px solid var(--border);border-radius:4px;background:var(--card);color:var(--dim);font-size:11px;cursor:pointer;transition:all .15s}
.session-tab:hover{border-color:var(--accent);color:var(--text)}
.session-tab.active{background:var(--accent);border-color:var(--accent);color:#fff}
.sample-timeline{display:none;margin-left:26px}
.sample-timeline.open{display:block}
.legend{display:flex;gap:14px;margin:0 0 10px 26px;font-size:11px;color:var(--dim);align-items:center;flex-wrap:wrap}
.legend .dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
.tl-container{background:var(--card);border:1px solid var(--border);border-radius:6px;overflow:auto;max-height:400px}
.tl-header{position:sticky;top:0;z-index:2;background:var(--card);border-bottom:1px solid var(--border);padding:6px 12px;font-size:10px;color:var(--dim);display:flex;gap:8px}
.tl-row{display:flex;align-items:center;padding:4px 12px;border-bottom:1px solid rgba(42,45,58,0.4);min-height:32px;transition:background .08s}
.tl-row:hover{background:rgba(91,141,239,0.04)}
.tl-label{width:50px;flex-shrink:0;font-size:10px;color:var(--dim);display:flex;align-items:center;gap:4px}
.tl-label .dot{width:5px;height:5px;border-radius:50%;flex-shrink:0}
.tl-blocks{display:flex;gap:2px;flex-wrap:wrap;flex:1}
.tl-block{width:14px;height:14px;border-radius:2px;cursor:pointer;flex-shrink:0;transition:transform .1s}
.tl-block:hover{transform:scale(1.6);z-index:3;box-shadow:0 0 6px rgba(0,0,0,0.4)}
.tl-block.bash{background:#4ade80}
.tl-block.read{background:#34d399}
.tl-block.edit{background:#fbbf24}
.tl-block.write{background:#fb923c}
.tl-block.other{background:#a78bfa}
.tl-block.llm{background:#5b8def}
.tl-block.cut{border:2px solid var(--red)}
.tooltip{display:none;position:fixed;z-index:999;background:#1e2130;border:1px solid var(--border);border-radius:6px;padding:8px 12px;max-width:320px;font-size:11px;line-height:1.5;box-shadow:0 8px 24px rgba(0,0,0,0.4);pointer-events:none}
.tooltip .tt-head{color:var(--dim);font-size:10px;margin-bottom:3px}
.tooltip .tt-type{font-weight:600;margin-bottom:2px}
.tooltip .tt-content{color:var(--dim);margin-top:4px;font-size:10px;max-height:60px;overflow:hidden}
.detail-pane{margin:8px 0 0 26px;background:var(--card);border:1px solid var(--border);border-radius:6px;padding:12px 16px;font-size:12px;display:none}
.detail-pane.show{display:block}
.detail-pane .dp-head{display:flex;justify-content:space-between;margin-bottom:6px}
.detail-pane .dp-turn{color:var(--accent);font-weight:600}
.detail-pane .dp-tools{color:var(--green)}
.detail-pane .dp-content{color:var(--text);margin-top:6px;line-height:1.5;white-space:pre-wrap;max-height:100px;overflow-y:auto;font-size:11px}
.detail-pane .dp-meta{color:var(--dim);font-size:10px;margin-top:4px}
.collapse-all{font-size:11px;color:var(--dim);cursor:pointer;margin-left:12px}
.collapse-all:hover{color:var(--text)}
.empty-state{text-align:center;padding:40px;color:var(--dim);font-size:13px}
</style>
</head>
<body>
<div class="header">
  <h1>Trajectory Timeline <span class="collapse-all" onclick="collapseAll()">collapse all</span></h1>
  <span class="status" id="status">loading...</span>
</div>
<div class="legend">
  <span><span class="dot" style="background:#4ade80"></span>Bash</span>
  <span><span class="dot" style="background:#34d399"></span>Read</span>
  <span><span class="dot" style="background:#fbbf24"></span>Edit</span>
  <span><span class="dot" style="background:#fb923c"></span>Write</span>
  <span><span class="dot" style="background:#a78bfa"></span>Other</span>
  <span><span class="dot" style="background:#5b8def"></span>LLM</span>
  <span style="color:var(--red)">&#x25A2; length cutoff</span>
</div>
<div id="all-samples"></div>
<div class="detail-pane" id="detail-pane"></div>
<div class="tooltip" id="tooltip"></div>

<script>
var toolColors = {
  'Bash':'bash','Read':'read','Edit':'edit','Write':'write',
  'TaskCreate':'other','TaskGet':'other','TaskUpdate':'other','TaskList':'other',
  'ExitWorktree':'other','AskUserQuestion':'other','Find':'other','find':'other','Grep':'other'
};

var activeSessions = {};

function fetchData() {
  var st = document.getElementById('status');
  st.textContent = 'fetching...';
  st.className = 'status';
  fetch('/api/data')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.error) { st.textContent = 'error: ' + data.error; return; }
      render(data);
      st.textContent = 'live \u2022 ' + new Date(data.updated_at * 1000).toLocaleTimeString();
      st.className = 'status live';
    })
    .catch(function(e) {
      st.textContent = 'error: ' + e;
      st.className = 'status';
    });
}

function render(data) {
  var container = document.getElementById('all-samples');
  var samples = data.samples;
  var overview = data.overview || {};
  var html = '';

  var keys = Object.keys(samples).sort();
  if (keys.length === 0) {
    container.innerHTML = '<div class="empty-state">No trajectory data yet. Waiting for FileSampleRecord writes...</div>';
    return;
  }

  for (var i = 0; i < keys.length; i++) {
    var uid = keys[i];
    var sessions = samples[uid];
    var ov = overview[uid] || {};
    var totalTurns = ov.total_turns || 0;
    var totalTrajs = ov.total_trajs || 0;
    var success = ov.success_count || 0;
    var sc = success > 0 ? 'ok' : 'fail';
    var shortUid = uid.length > 12 ? uid.substring(0, 12) + '...' : uid;

    html += '<div class="sample-section">';
    html += '<div class="sample-header" onclick="toggleSample(this)">';
    html += '<span class="arrow open">&#9654;</span>';
    html += '<span class="stitle">Sample ' + i + ' <span style="font-weight:400;color:var(--dim)">' + shortUid + '</span></span>';
    html += '<span class="smeta">' + sessions.length + ' sessions &middot; ' + totalTurns + ' turns &middot; ' + totalTrajs + ' trajs</span>';
    html += '<span class="sresolved ' + sc + '">' + success + '/' + totalTrajs + ' solved</span>';
    html += '</div>';

    // Session tabs
    html += '<div class="session-bar" id="sbar-' + i + '">';
    for (var j = 0; j < sessions.length; j++) {
      var s = sessions[j];
      var solved = 0;
      for (var k in s.traj_rewards) { if (s.traj_rewards[k] > 0) solved++; }
      var activeClass = j === 0 ? ' active' : '';
      html += '<span class="session-tab' + activeClass + '" data-si="' + i + '" data-sj="' + j + '" onclick="selectSession(event, this)">S' + s.session;
      html += ' <span class="' + (solved > 0 ? 'ok' : 'fail') + '">' + solved + '/' + s.traj_count + '</span>';
      html += '</span>';
    }
    html += '</div>';

    // Timeline per session
    html += '<div class="sample-timeline open" id="stl-' + i + '">';
    for (var j = 0; j < sessions.length; j++) {
      var s = sessions[j];
      var display = j === 0 ? 'block' : 'none';
      html += '<div class="tl-container" id="stl-' + i + '-' + j + '" style="display:' + display + '">';
      html += '<div class="tl-header">Session ' + s.session + ' &middot; ' + s.turn_count + ' turns &middot; ' + s.traj_count + ' trajs</div>';

      var turns = s.turns;
      var rows = {};
      for (var t = 0; t < turns.length; t++) {
        var turn = turns[t];
        var trajKey = 'traj' + turn.traj;
        if (!rows[trajKey]) rows[trajKey] = [];
        rows[trajKey].push(turn);
      }

      var trajKeys = Object.keys(rows).sort();
      for (var tk = 0; tk < trajKeys.length; tk++) {
        var rowTurns = rows[trajKeys[tk]];
        html += '<div class="tl-row">';
        html += '<div class="tl-label"><span class="dot"></span>' + trajKeys[tk] + '</div>';
        html += '<div class="tl-blocks">';
        for (var rt = 0; rt < rowTurns.length; rt++) {
          var rturn = rowTurns[rt];
          var cls = 'tl-block ';
          if (rturn.type === 'llm') {
            cls += 'llm';
          } else if (rturn.tools && rturn.tools.length > 0) {
            var c = toolColors[rturn.tools[0]] || 'other';
            cls += c;
          } else {
            cls += 'other';
          }
          if (rturn.finish_reason === 'length') cls += ' cut';
          var title = 'T' + rturn.turn + ' traj' + rturn.traj;
          if (rturn.tools && rturn.tools.length > 0) title += ' [' + rturn.tools.join(',') + ']';
          title += ' ' + (rturn.finish_reason || '');
          html += '<div class="' + cls + '" title="' + title.replace(/"/g,'&quot;') + '"';
          html += ' data-turn=\'' + JSON.stringify(rturn).replace(/'/g,"&#39;") + '\'';
          html += ' onmouseenter="showTooltip(event,this)" onmouseleave="hideTooltip()"';
          html += ' onclick="showDetail(this)"></div>';
        }
        html += '</div></div>';
      }
      html += '</div>';
    }
    html += '</div></div>';
  }
  container.innerHTML = html;

  // Restore active session selections after re-render
  for (var si in activeSessions) {
    var tab = document.querySelector('.session-tab[data-si="' + si + '"][data-sj="' + activeSessions[si] + '"]');
    if (tab) selectSession(null, tab);
  }
}

function toggleSample(el) {
  var arrow = el.querySelector('.arrow');
  var open = arrow.classList.toggle('open');
  var si = el.parentElement.querySelector('.sample-timeline');
  if (si) si.classList.toggle('open', open);
}

function selectSession(ev, el) {
  var bar = el.parentElement;
  var tabs = bar.querySelectorAll('.session-tab');
  for (var i = 0; i < tabs.length; i++) tabs[i].classList.remove('active');
  el.classList.add('active');
  var si = el.getAttribute('data-si');
  var sj = el.getAttribute('data-sj');
  activeSessions[si] = sj;
  var containers = document.querySelectorAll('#stl-' + si + ' > div');
  for (var j = 0; j < containers.length; j++) {
    containers[j].style.display = (j == sj) ? 'block' : 'none';
  }
}

function collapseAll() {
  var arrows = document.querySelectorAll('.sample-header .arrow');
  for (var i = 0; i < arrows.length; i++) {
    arrows[i].classList.remove('open');
  }
  var timelines = document.querySelectorAll('.sample-timeline');
  for (var j = 0; j < timelines.length; j++) {
    timelines[j].classList.remove('open');
  }
}

function showTooltip(ev, el) {
  var d = JSON.parse(el.getAttribute('data-turn'));
  var tt = document.getElementById('tooltip');
  tt.innerHTML = '<div class="tt-head">Turn ' + d.turn + ' (traj ' + d.traj + ')</div>'
    + '<div class="tt-type">' + (d.tools && d.tools.length ? d.tools.join(', ') : 'LLM') + '</div>'
    + '<div class="tt-content">' + (d.content || '(empty)').substring(0, 120) + '</div>';
  tt.style.display = 'block';
  tt.style.left = (ev.clientX + 12) + 'px';
  tt.style.top = (ev.clientY + 12) + 'px';
}

function hideTooltip() {
  document.getElementById('tooltip').style.display = 'none';
}

function showDetail(el) {
  var d = JSON.parse(el.getAttribute('data-turn'));
  var pane = document.getElementById('detail-pane');
  pane.innerHTML = '<div class="dp-head">'
    + '<span class="dp-turn">Turn ' + d.turn + ' &middot; traj ' + d.traj + '</span>'
    + '<span class="dp-tools">' + (d.tools && d.tools.length ? d.tools.join(', ') : 'LLM') + '</span>'
    + '</div>'
    + '<div class="dp-meta">finish_reason: ' + (d.finish_reason || '?') + '</div>'
    + '<div class="dp-content">' + (d.content || '(empty)') + '</div>';
  pane.classList.add('show');
}

// Auto-refresh every 1 second
fetchData();
setInterval(fetchData, 1000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Timeline visualization server")
    parser.add_argument(
        "data_dir", help="Root directory containing FileSampleRecord data"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="HTTP port (default: 8080)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)"
    )
    args = parser.parse_args()

    data_path = Path(args.data_dir).resolve()
    if not data_path.is_dir():
        print(f"Error: {data_path} is not a directory")
        return

    # Set data directory on handler class
    TimelineHandler.data_dir = data_path

    server = HTTPServer((args.host, args.port), TimelineHandler)
    print(f"Serving timeline at http://localhost:{args.port}")
    print(f"Data directory: {data_path}")
    print("Auto-refresh: every 1 second")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
