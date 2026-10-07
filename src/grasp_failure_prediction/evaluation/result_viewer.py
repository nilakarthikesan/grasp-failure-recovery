"""Read-only local web viewer for evaluation result bundles."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlparse
import webbrowser


def discover_results(runs_root: str | Path) -> list[dict]:
    """Return normalized summaries for every result.json below ``runs_root``."""

    root = Path(runs_root).resolve()
    results: list[dict] = []
    if not root.is_dir():
        return results
    for result_path in root.rglob("result.json"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_dir = result_path.parent
        try:
            relative_dir = run_dir.relative_to(root)
        except ValueError:
            continue
        resolved = {}
        resolved_path = run_dir / "resolved_case.json"
        if resolved_path.is_file():
            try:
                resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                resolved = {}
        case = resolved.get("case", {})
        outcome = result.get("outcome", {})
        retargeting = result.get("retargeting", {})
        execution = result.get("resolved_execution", {})
        artifacts = result.get("artifacts", {})
        video_relative = artifacts.get("video")
        video_path = run_dir / video_relative if video_relative else None
        has_video = bool(video_path and video_path.is_file())
        file_relative = (
            (relative_dir / video_relative).as_posix() if has_video else None
        )
        results.append(
            {
                "run_id": relative_dir.as_posix(),
                "case_id": result.get("case_id", relative_dir.name),
                "status": result.get("status", "unknown"),
                "success": bool(result.get("success", False)),
                "failure_type": result.get("failure_type"),
                "environment_id": execution.get("environment_id"),
                "execution_protocol_id": execution.get("execution_protocol_id"),
                "code_commit": execution.get("code_commit"),
                "mujoco_version": execution.get("mujoco_version"),
                "object_id": case.get("object", {}).get("id"),
                "mass_kg": case.get("object", {}).get("mass_kg"),
                "grasp_id": case.get("grasp", {}).get("id"),
                "inference_seed": case.get("grasp", {}).get("inference_seed"),
                "simulator_seed": case.get("seed"),
                "maximum_lift_m": outcome.get("maximum_lift_m"),
                "hold_duration_s": outcome.get("hold_duration_s"),
                "opposing_contact_duration_s": outcome.get(
                    "opposing_contact_duration_s"
                ),
                "peak_contact_normal_force_n": outcome.get(
                    "peak_contact_normal_force_n"
                ),
                "peak_actuator_force_fraction": outcome.get(
                    "peak_actuator_force_fraction"
                ),
                "mean_fingertip_error_m": retargeting.get(
                    "mean_fingertip_error_m"
                ),
                "maximum_fingertip_error_m": retargeting.get(
                    "maximum_fingertip_error_m"
                ),
                "has_video": has_video,
                "video_url": f"/files/{quote(file_relative)}" if has_video else None,
                "result_url": f"/files/{quote((relative_dir / 'result.json').as_posix())}",
                "modified_ns": result_path.stat().st_mtime_ns,
            }
        )
    results.sort(key=lambda item: item["modified_ns"], reverse=True)
    return results


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grasp evaluation results</title><style>
:root{color-scheme:dark;--bg:#0a0d12;--panel:#121722;--panel2:#181f2d;--text:#ecf1f8;--muted:#929db0;--line:#273144;--green:#42d392;--red:#ff6b75;--blue:#6aa9ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif}
header{position:sticky;top:0;z-index:3;display:flex;gap:18px;align-items:center;padding:16px 24px;background:#0a0d12e8;border-bottom:1px solid var(--line);backdrop-filter:blur(14px)}
h1{font-size:18px;margin:0}.summary{color:var(--muted)}.controls{margin-left:auto;display:flex;gap:8px;align-items:center}button,input{border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:9px;padding:8px 11px}button{cursor:pointer}button.active{border-color:var(--blue);color:#fff;background:#17325a}input{width:250px}
main{display:grid;grid-template-columns:minmax(340px,42%) 1fr;min-height:calc(100vh - 67px)}.list{border-right:1px solid var(--line);padding:16px;overflow:auto}.cards{display:grid;gap:10px}.card{padding:14px;border:1px solid var(--line);border-radius:12px;background:var(--panel);cursor:pointer}.card:hover,.card.selected{border-color:var(--blue);background:var(--panel2)}.row{display:flex;align-items:center;gap:9px}.grow{flex:1}.case{font-weight:650;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.run{font-size:12px;color:var(--muted);margin-top:3px}.badge{padding:3px 8px;border-radius:999px;font-size:11px;font-weight:700}.pass{color:var(--green);background:#123929}.fail{color:var(--red);background:#421d25}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:11px}.metric{font-size:11px;color:var(--muted)}.metric b{display:block;color:var(--text);font-size:13px}.video-dot{color:var(--blue);font-size:12px}
.detail{padding:24px;overflow:auto}.empty{height:70vh;display:grid;place-items:center;color:var(--muted)}video{display:block;width:100%;max-height:56vh;background:#000;border:1px solid var(--line);border-radius:14px}.detail h2{font-size:22px;margin:18px 0 3px}.detail .sub{color:var(--muted);margin-bottom:18px}.grid{display:grid;grid-template-columns:repeat(3,minmax(130px,1fr));gap:10px}.tile{padding:13px;border:1px solid var(--line);background:var(--panel);border-radius:10px;color:var(--muted)}.tile b{display:block;color:var(--text);font-size:16px;margin-top:3px}.links{display:flex;gap:12px;margin-top:18px}.links a{color:var(--blue)}
@media(max-width:900px){main{grid-template-columns:1fr}.list{border-right:0;border-bottom:1px solid var(--line);max-height:45vh}.controls{flex-wrap:wrap}input{width:170px}.grid{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<header><h1>Grasp evaluations</h1><span class="summary" id="summary">Loading…</span><div class="controls"><button data-filter="all" class="active">All</button><button data-filter="success">Success</button><button data-filter="failure">Failure</button><input id="search" placeholder="Search case, run, object…"></div></header>
<main><section class="list"><div class="cards" id="cards"></div></section><section class="detail" id="detail"><div class="empty">Select an evaluation result</div></section></main>
<script>
const state={items:[],filter:'all',query:'',selected:null};
const n=(v,d=3)=>v==null?'—':Number(v).toFixed(d); const mm=v=>v==null?'—':`${(1000*v).toFixed(1)} mm`; const pct=v=>v==null?'—':`${(100*v).toFixed(1)}%`;
function visible(){return state.items.filter(x=>(state.filter==='all'||(state.filter==='success'&&x.success)||(state.filter==='failure'&&!x.success))&&JSON.stringify(x).toLowerCase().includes(state.query))}
function render(){const items=visible();document.querySelector('#summary').textContent=`${state.items.length} results · ${state.items.filter(x=>x.success).length} success · ${state.items.filter(x=>!x.success).length} failure`;const cards=document.querySelector('#cards');cards.innerHTML=items.map(x=>`<article class="card ${state.selected===x.run_id?'selected':''}" data-id="${x.run_id}"><div class="row"><div class="grow"><div class="case">${x.case_id}</div><div class="run">${x.run_id}</div></div><span class="badge ${x.success?'pass':'fail'}">${x.success?'SUCCESS':(x.failure_type||'FAILED')}</span></div><div class="metrics"><span class="metric">Lift<b>${mm(x.maximum_lift_m)}</b></span><span class="metric">Hold<b>${n(x.hold_duration_s,2)} s</b></span><span class="metric">Alignment<b>${mm(x.mean_fingertip_error_m)}</b></span></div>${x.has_video?'<div class="video-dot">● Video attached</div>':''}</article>`).join('')||'<div class="empty">No matching results</div>';cards.querySelectorAll('.card').forEach(el=>el.onclick=()=>select(el.dataset.id));}
function select(id){state.selected=id;const x=state.items.find(v=>v.run_id===id);render();if(!x)return;document.querySelector('#detail').innerHTML=`${x.video_url?`<video controls preload="metadata" src="${x.video_url}"></video>`:'<div class="empty">No video attached to this historical result</div>'}<h2>${x.case_id}</h2><div class="sub">${x.run_id} · ${x.environment_id||'unknown environment'} · ${x.execution_protocol_id||'unknown protocol'}</div><div class="grid"><div class="tile">Outcome<b style="color:${x.success?'var(--green)':'var(--red)'}">${x.success?'Success':(x.failure_type||'Failure')}</b></div><div class="tile">Maximum lift<b>${mm(x.maximum_lift_m)}</b></div><div class="tile">Stable hold<b>${n(x.hold_duration_s,2)} s</b></div><div class="tile">Mean alignment error<b>${mm(x.mean_fingertip_error_m)}</b></div><div class="tile">Opposing contact<b>${n(x.opposing_contact_duration_s,2)} s</b></div><div class="tile">Peak normal force<b>${n(x.peak_contact_normal_force_n,2)} N</b></div><div class="tile">Peak actuator limit<b>${pct(x.peak_actuator_force_fraction)}</b></div><div class="tile">Object / mass<b>${x.object_id||'—'} · ${x.mass_kg??'—'} kg</b></div><div class="tile">Inference / sim seed<b>${x.inference_seed??'—'} / ${x.simulator_seed??'—'}</b></div></div><div class="links"><a href="${x.result_url}" target="_blank">Open result.json</a>${x.video_url?`<a href="${x.video_url}" target="_blank">Open video</a>`:''}</div>`;}
document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{state.filter=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));render()});document.querySelector('#search').oninput=e=>{state.query=e.target.value.toLowerCase();render()};
fetch('/api/results').then(r=>r.json()).then(items=>{state.items=items;render();if(items.length)select(items[0].run_id)}).catch(e=>document.querySelector('#summary').textContent=e);
</script></body></html>"""


class ResultViewerHandler(BaseHTTPRequestHandler):
    runs_root: Path

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(HTML.encode(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/results":
            payload = json.dumps(discover_results(self.runs_root)).encode()
            self._send_bytes(payload, "application/json")
            return
        if parsed.path.startswith("/files/"):
            self._send_file(unquote(parsed.path.removeprefix("/files/")))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, relative: str) -> None:
        root = self.runs_root.resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
            if not match:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else end
            elif match.group(2):
                start = max(0, size - int(match.group(2)))
            if start > end or start >= size:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            end = min(end, size - 1)
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status is HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, format: str, *args) -> None:
        return


def serve_results(runs_root: Path, host: str, port: int, *, open_browser: bool) -> None:
    root = runs_root.resolve()
    handler = type("ConfiguredResultViewerHandler", (ResultViewerHandler,), {"runs_root": root})
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{server.server_port}/"
    print(f"Serving evaluation results from {root}\n{url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    serve_results(args.runs_root, args.host, args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
