"""Read-only local web viewer for evaluation result bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlparse
import webbrowser


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_evaluation(runs_root: Path, evaluation_root: Path) -> dict[str, dict]:
    """Validate saved predictions and bind them to completed raw attempts.

    This viewer neither loads model weights nor computes new predictions. The
    checked report, manifest, and CSV must describe the same complete collection.
    """
    report = json.loads((evaluation_root / "report.json").read_text())
    if (report.get("schema_version") != "frozen_fresh_task_outcome_evaluation_v1"
            or report.get("prediction_time_s") != 2.96):
        raise ValueError("unsupported frozen evaluation report or prediction time")
    for filename, key in (("manifest.json", "manifest_sha256"),
                          ("predictions.csv", "predictions_sha256")):
        if _sha256(evaluation_root / filename) != report.get(key):
            raise ValueError(f"frozen evaluation {filename} hash mismatch")
    manifest = json.loads((evaluation_root / "manifest.json").read_text())
    counts = report.get("counts", {})
    included = counts.get("included")
    if (manifest.get("schema_version") != "fresh_task_prefix_dataset_v1"
            or manifest.get("counts") != counts or type(included) is not int or included < 1
            or any(type(counts.get(key)) is not int or counts[key] < 0
                   for key in ("indexed", "excluded", "success", "failure"))
            or counts.get("indexed") != included + counts.get("excluded", -1)
            or counts.get("success", -1) + counts.get("failure", -1) != included):
        raise ValueError("frozen evaluation report counts disagree with its manifest")
    episodes = manifest.get("episodes", [])
    if not isinstance(episodes, list) or len(episodes) != included:
        raise ValueError("frozen evaluation episode count differs from the report")
    model_names = {"training_prevalence", "proprioception_and_commands", "commands_only"}
    if set(report.get("models", {})) != model_names:
        raise ValueError("frozen evaluation must contain all three saved baselines")
    predictions: dict[str, dict[str, dict]] = {name: {} for name in model_names}
    with (evaluation_root / "predictions.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["episode_id", "partition", "label_success", "model",
                                "p_success", "p_failure", "failure_alert"]:
            raise ValueError("unsupported frozen evaluation prediction columns")
        for row in reader:
            name, episode_id = row["model"], row["episode_id"]
            if (name not in predictions or not re.fullmatch(r"episode_\d{6}", episode_id)
                    or episode_id in predictions[name] or row["partition"] != "fresh_evaluation"
                    or row["label_success"] not in ("0", "1")
                    or row["failure_alert"] not in ("0", "1")):
                raise ValueError("invalid or duplicate frozen evaluation prediction")
            p_success, p_failure = float(row["p_success"]), float(row["p_failure"])
            if (not all(math.isfinite(p) and 0 <= p <= 1 for p in (p_success, p_failure))
                    or not math.isclose(p_success + p_failure, 1, rel_tol=0, abs_tol=1e-12)
                    or int(row["failure_alert"]) != int(p_failure >= .5)):
                raise ValueError("invalid frozen evaluation probability or fixed alert")
            predictions[name][episode_id] = dict(label=int(row["label_success"]),
                                                p_failure=p_failure, failure_alert=p_failure >= .5)
    annotations = {}
    for episode in episodes:
        episode_id = episode.get("episode_id")
        if (not isinstance(episode_id, str) or not re.fullmatch(r"episode_\d{6}", episode_id)
                or episode_id in annotations):
            raise ValueError("invalid or duplicate frozen evaluation episode")
        label = episode.get("source_task_label")
        novelty = episode.get("geometry_novelty")
        noise = episode.get("noise_amplitude_rad")
        observed_at = episode.get("observed_at_s")
        if (type(label) is not int or label not in (0, 1)
                or novelty not in ("new_geometry", "new_scene_seen_geometry")
                or not isinstance(noise, (int, float)) or isinstance(noise, bool)
                or not math.isfinite(noise) or noise < 0
                or not isinstance(observed_at, (int, float)) or isinstance(observed_at, bool)
                or not math.isclose(observed_at, 2.96, rel_tol=0, abs_tol=1e-9)):
            raise ValueError("invalid frozen evaluation outcome, novelty, or noise")
        if any(episode_id not in rows or rows[episode_id]["label"] != label
               for rows in predictions.values()):
            raise ValueError("frozen evaluation predictions disagree with episode labels")
        raw = runs_root / episode_id
        metadata_path = raw / "collection_metadata.json"
        for filename in ("collection_metadata.json", "result.json"):
            if _sha256(raw / filename) != episode.get("raw_files_sha256", {}).get(filename):
                raise ValueError(f"frozen evaluation raw {filename} hash mismatch")
        metadata = json.loads(metadata_path.read_text())
        result = json.loads((raw / "result.json").read_text())
        if (metadata.get("status") != "completed" or type(metadata.get("label")) is not int
                or metadata["label"] != label or metadata.get("noise_amplitude_rad") != noise
                or metadata.get("geometry_group_id") != episode.get("geometry_group_id")
                or result.get("status") != "completed" or type(result.get("success")) is not bool
                or int(result["success"]) != label):
            raise ValueError("frozen evaluation differs from the raw attempt")
        full = predictions["proprioception_and_commands"][episode_id]
        error = ("missed_failure" if label == 0 and not full["failure_alert"] else
                 "false_alarm" if label == 1 and full["failure_alert"] else "correct")
        annotations[episode_id] = dict(full, observed_at_s=observed_at, model_error=error,
                                       geometry_novelty=novelty, noise_amplitude_rad=noise)
    if (any(set(rows) != set(annotations) for rows in predictions.values())
            or sum(row["label"] for row in annotations.values()) != counts["success"]):
        raise ValueError("frozen evaluation prediction or label counts disagree with report")
    return annotations


def discover_results(runs_root: str | Path, evaluation_root: str | Path | None = None) -> list[dict]:
    """Return normalized summaries for every result.json below ``runs_root``."""

    root = Path(runs_root).resolve()
    annotations = (load_frozen_evaluation(root, Path(evaluation_root).resolve())
                   if evaluation_root is not None else None)
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
        if annotations is not None:
            if relative_dir.as_posix() not in annotations:
                raise ValueError("use the evaluated collection directory as --runs-root")
            results[-1]["evaluation"] = annotations[relative_dir.as_posix()]
    if annotations is not None and {item["run_id"] for item in results} != set(annotations):
        raise ValueError("frozen evaluation includes missing or unreadable raw results")
    results.sort(key=lambda item: item["modified_ns"], reverse=True)
    if annotations is not None:
        results.sort(key=lambda item: item["has_video"], reverse=True)
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
.controls label{white-space:nowrap;font-size:12px;color:var(--muted)}.controls input[type=checkbox]{width:auto}.prediction{margin:18px 0;padding:16px;border:1px solid var(--blue);border-radius:12px;background:var(--panel)}.prediction h3{margin:0 0 12px}.prediction p,.walkthrough li{color:var(--muted)}.walkthrough{padding-left:22px;margin-bottom:0}.walkthrough li{margin-top:9px}.walkthrough strong{color:var(--text)}.load-more{width:100%;margin-top:12px}.prediction-note{font-size:12px;margin-top:9px;color:var(--blue)}
@media(max-width:900px){main{grid-template-columns:1fr}.list{border-right:0;border-bottom:1px solid var(--line);max-height:45vh}.controls{flex-wrap:wrap}input{width:170px}.grid{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<header><h1>Grasp evaluations</h1><span class="summary" id="summary">Loading…</span><div class="controls"><button data-filter="all" class="active">All</button><button data-filter="success">Success</button><button data-filter="failure">Failure</button><label><input id="video-only" type="checkbox"> Videos only</label><input id="search" placeholder="Search case, run, object…"></div></header>
<main><section class="list"><div class="cards" id="cards"></div><button class="load-more" id="load-more" hidden>Show 100 more</button></section><section class="detail" id="detail"><div class="empty">Select an evaluation result</div></section></main>
<script>
const state={items:[],filter:'all',query:'',selected:null,videoOnly:false,limit:100};
const n=(v,d=3)=>v==null?'—':Number(v).toFixed(d); const mm=v=>v==null?'—':`${(1000*v).toFixed(1)} mm`; const pct=v=>v==null?'—':`${(100*v).toFixed(1)}%`;
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const errorName=e=>({correct:'Correct prediction',missed_failure:'Missed failure',false_alarm:'False alarm'}[e]);
function visible(){return state.items.filter(x=>(state.filter==='all'||(state.filter==='success'&&x.success)||(state.filter==='failure'&&!x.success))&&(!state.videoOnly||x.has_video)&&JSON.stringify(x).toLowerCase().includes(state.query))}
function render(){const matching=visible();const items=matching.slice(0,state.limit);document.querySelector('#summary').textContent=`${state.items.length} results · ${state.items.filter(x=>x.success).length} success · ${state.items.filter(x=>!x.success).length} failure · showing ${items.length}/${matching.length}`;document.querySelector('#load-more').hidden=matching.length<=state.limit;const cards=document.querySelector('#cards');cards.innerHTML=items.map(x=>`<article class="card ${state.selected===x.run_id?'selected':''}" data-id="${esc(x.run_id)}"><div class="row"><div class="grow"><div class="case">${esc(x.case_id)}</div><div class="run">${esc(x.run_id)}</div></div><span class="badge ${x.success?'pass':'fail'}">${esc(x.success?'SUCCESS':(x.failure_type||'FAILED'))}</span></div><div class="metrics"><span class="metric">Lift<b>${mm(x.maximum_lift_m)}</b></span><span class="metric">Hold<b>${n(x.hold_duration_s,2)} s</b></span><span class="metric">Alignment<b>${mm(x.mean_fingertip_error_m)}</b></span></div>${x.evaluation?`<div class="prediction-note">${pct(x.evaluation.p_failure)} predicted failure · ${errorName(x.evaluation.model_error)}</div>`:''}${x.has_video?'<div class="video-dot">● Video attached</div>':''}</article>`).join('')||'<div class="empty">No matching results</div>';cards.querySelectorAll('.card').forEach(el=>el.onclick=()=>select(el.dataset.id));}
function predictionDetail(x){const e=x.evaluation;if(!e)return '';const novelty=e.geometry_novelty==='new_geometry'?'New object geometry':'New scene, familiar geometry';return `<section class="prediction"><h3>Frozen model prediction</h3><div class="grid"><div class="tile">Failure probability from observations through ${n(e.observed_at_s,2)} s<b>${pct(e.p_failure)}</b></div><div class="tile">Fixed 50% alert threshold<b>${e.failure_alert?'Predict failure':'Predict success'}</b></div><div class="tile">Prediction versus actual outcome<b>${errorName(e.model_error)}</b></div><div class="tile">Object novelty<b>${novelty}</b></div><div class="tile">Requested finger angle noise<b>${n(e.noise_amplitude_rad,2)} rad</b></div><div class="tile">Recorded label<b>${e.label} = ${e.label?'success':'failure'}</b></div></div><p>This is a prediction of eventual grasp, lift, and hold success from saved observations before lifting. The model weights, scaling, and alert threshold are unchanged.</p><h3>What happened in this attempt</h3><ol class="walkthrough"><li><strong>Capture the starting scene.</strong> Save calibrated RGB and depth for ${esc(x.object_id||'this object')}.</li><li><strong>Generate a fresh HUG grasp.</strong> Turn that observation into finger and palm targets for the Shadow Hand.</li><li><strong>Execute the fixed controller with noise.</strong> Add bounded finger target offsets up to ${n(e.noise_amplitude_rad,2)} rad. Each offset stays fixed during the attempt; joint limits clip the commands. Noise can improve or weaken the grasp.</li><li><strong>Use only observations through ${n(e.observed_at_s,2)} seconds.</strong> The saved predictor uses finger angles, velocities, and commands from closing, before lifting. It assigns ${pct(e.p_failure)} probability of failure.</li><li><strong>Compare with the completed simulation.</strong> The frozen contact, lift, and hold rule assigns ${e.label?'success (1)':'failure (0)'}. ${e.model_error==='correct'?'The prediction agrees.':e.model_error==='false_alarm'?'The model warned, but the attempt succeeded.':'The model missed a failed attempt.'}</li></ol></section>`;}
function select(id){state.selected=id;const x=state.items.find(v=>v.run_id===id);render();if(!x)return;document.querySelector('#detail').innerHTML=`${x.video_url?`<video controls preload="metadata" src="${x.video_url}"></video>`:'<div class="empty">No video attached to this historical result</div>'}<h2>${x.case_id}</h2><div class="sub">${x.run_id} · ${x.environment_id||'unknown environment'} · ${x.execution_protocol_id||'unknown protocol'}</div>${predictionDetail(x)}<div class="grid"><div class="tile">Outcome<b style="color:${x.success?'var(--green)':'var(--red)'}">${x.success?'Success':(x.failure_type||'Failure')}</b></div><div class="tile">Maximum lift<b>${mm(x.maximum_lift_m)}</b></div><div class="tile">Stable hold<b>${n(x.hold_duration_s,2)} s</b></div><div class="tile">Mean alignment error<b>${mm(x.mean_fingertip_error_m)}</b></div><div class="tile">Opposing contact<b>${n(x.opposing_contact_duration_s,2)} s</b></div><div class="tile">Peak normal force<b>${n(x.peak_contact_normal_force_n,2)} N</b></div><div class="tile">Peak actuator limit<b>${pct(x.peak_actuator_force_fraction)}</b></div><div class="tile">Object / mass<b>${x.object_id||'—'} · ${x.mass_kg??'—'} kg</b></div><div class="tile">Inference / sim seed<b>${x.inference_seed??'—'} / ${x.simulator_seed??'—'}</b></div></div><div class="links"><a href="${x.result_url}" target="_blank">Open result.json</a>${x.video_url?`<a href="${x.video_url}" target="_blank">Open video</a>`:''}</div>`;}
document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{state.filter=b.dataset.filter;state.limit=100;document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));render()});document.querySelector('#search').oninput=e=>{state.query=e.target.value.toLowerCase();state.limit=100;render()};
document.querySelector('#video-only').onchange=e=>{state.videoOnly=e.target.checked;state.limit=100;render()};document.querySelector('#load-more').onclick=()=>{state.limit+=100;render()};
fetch('/api/results').then(r=>{if(!r.ok)throw new Error('Saved evaluation could not be verified. Check the server output and collection path.');return r.json()}).then(items=>{state.items=items;render();if(items.length)select(items[0].run_id)}).catch(e=>document.querySelector('#summary').textContent=e.message);
</script></body></html>"""


class ResultViewerHandler(BaseHTTPRequestHandler):
    runs_root: Path
    evaluation_root: Path | None = None

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(HTML.encode(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/results":
            try:
                payload = json.dumps(discover_results(self.runs_root, self.evaluation_root)).encode()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.send_error(HTTPStatus.CONFLICT, str(exc))
                return
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


def serve_results(runs_root: Path, host: str, port: int, *, open_browser: bool,
                  evaluation_root: Path | None = None) -> None:
    root = runs_root.resolve()
    evaluation = evaluation_root.resolve() if evaluation_root is not None else None
    if evaluation is not None:
        discover_results(root, evaluation)
    handler = type("ConfiguredResultViewerHandler", (ResultViewerHandler,),
                   {"runs_root": root, "evaluation_root": evaluation})
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
    parser.add_argument("--evaluation-root", type=Path,
                        help="optional frozen evaluation report, manifest, and predictions directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    serve_results(args.runs_root, args.host, args.port, open_browser=not args.no_open,
                  evaluation_root=args.evaluation_root)


if __name__ == "__main__":
    main()
