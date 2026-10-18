#!/usr/bin/env python3
"""Click the calibration anchors straight onto each dataset folder's image.

The inverse of ``label_anchors.py``: instead of reading an ``anchor.txt``, this
walks the dataset folder by folder, shows one reference frame per folder in a
browser, and lets you click the cup anchors. Every click becomes a pixel pair;
on Save it writes that folder's ``anchor.txt`` in the format the pipeline
already parses::

    cup1 anchor (636, 663, 40cm)
    cup2 anchor (1097, 605, 100cm)

No Paint, no jotting down numbers, no transposed coordinates.

    python pick_anchors.py                          # dataset/ -> per-folder anchor.txt
    python pick_anchors.py --overwrite              # redo folders that already have one
    python pick_anchors.py --match "*/b1"           # only some folders
    python pick_anchors.py --dist-cm 40,100 --names cup1,cup2
    python pick_anchors.py --preview                # draw labelled copies when done

The page supports wheel-zoom (zoom into the cup rim for pixel accuracy),
drag-to-pan, backtracking, and switching to another frame in the same folder when
one shot is blurrier than the rest. Everything else is server-side: the browser
never gets a filesystem path, it posts pixel pairs to a local-only HTTP server
that does the writing.

Keyboard: click = place point, Backspace = undo, Enter = save & next,
N = next frame, S = skip folder, Q = quit.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import mimetypes
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# arguments / dataset discovery                                               #
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", type=Path, default=_SCRIPT_DIR / "dataset",
                   help="Dataset root to walk (default: ./dataset).")
    p.add_argument("--match", default=None, metavar="GLOB",
                   help="Only folders whose relative path matches this glob, e.g. '*/b1'.")
    p.add_argument("--overwrite", action="store_true",
                   help="Also redo folders that already have an anchor.txt.")
    p.add_argument("--names", default="cup1,cup2",
                   help="Anchor names, comma separated (default: cup1,cup2).")
    p.add_argument("--dist-cm", default="40,100", metavar="CM,CM",
                   help="True camera distance in cm for each anchor (default: 40,100).")
    p.add_argument("--frame", type=int, default=0, metavar="N",
                   help="Start on this frame index inside each folder (default: 0).")
    p.add_argument("--port", type=int, default=8765, help="Local HTTP port (default: 8765).")
    p.add_argument("--no-open", dest="open_browser", action="store_false",
                   help="Do not launch a browser automatically; just print the URL.")
    p.add_argument("--preview", dest="preview", action="store_true",
                   help="Afterwards run label_anchors.py to draw the labelled copies.")
    p.add_argument("--out", type=Path, default=_SCRIPT_DIR / "anchordataset",
                   help="Preview output folder for --preview (default: ./anchordataset).")
    p.set_defaults(open_browser=True, preview=False)
    args = p.parse_args(argv)

    args.names = [n.strip() for n in args.names.split(",") if n.strip()]
    try:
        args.dist_cm = [float(x) for x in args.dist_cm.split(",")]
    except ValueError:
        sys.exit(f"--dist-cm expects comma-separated numbers, got '{args.dist_cm}'")
    if len(args.names) != len(args.dist_cm):
        sys.exit(f"--names ({len(args.names)}) and --dist-cm ({len(args.dist_cm)}) "
                 "must have the same length")
    if not args.names:
        sys.exit("--names is empty; need at least one anchor")
    if any(d <= 0 for d in args.dist_cm):
        sys.exit("--dist-cm values must be positive")
    return args


def find_folders(args):
    """``[(folder, [images...]), ...]`` - folders that actually contain images."""
    root = args.dataset
    if not root.is_dir():
        sys.exit(f"--dataset is not a directory: {root}")
    folders = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            folders.setdefault(p.parent, []).append(p)
    resolved = root.resolve()
    picked = []
    for folder, images in sorted(folders.items()):
        try:
            rel = folder.resolve().relative_to(resolved).as_posix()
        except ValueError:
            rel = folder.name
        if args.match and not fnmatch.fnmatch(rel, args.match):
            continue
        picked.append({
            "folder": folder,
            "rel": rel or ".",
            "images": images,
            "has_anchor": (folder / "anchor.txt").is_file(),
        })
    return picked


# --------------------------------------------------------------------------- #
# anchor.txt writing                                                          #
# --------------------------------------------------------------------------- #
def fmt_cm(value):
    return f"{value:g}cm"


def anchor_lines(points, names, dist_cm):
    return [f"{name} anchor ({int(round(u))}, {int(round(v))}, {fmt_cm(d)})"
            for name, (u, v), d in zip(names, points, dist_cm)]


def write_anchor_file(folder, points, names, dist_cm):
    path = Path(folder) / "anchor.txt"
    path.write_text("\n".join(anchor_lines(points, names, dist_cm)) + "\n")
    return path


# --------------------------------------------------------------------------- #
# server state                                                                #
# --------------------------------------------------------------------------- #
class Session:
    def __init__(self, args, folders):
        self.args = args
        self.all_folders = folders
        self.queue = [f for f in folders if args.overwrite or not f["has_anchor"]]
        self.skipped_existing = [f for f in folders if not args.overwrite and f["has_anchor"]]
        self.index = 0
        self.frame = {id(f): min(args.frame, len(f["images"]) - 1) for f in folders}
        self.written = []
        self.skipped = []
        self.done = threading.Event()
        self.lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------- #
    def _maybe_finish(self):
        """Queue exhausted -> shut the server down, after a beat.

        The delay matters: the page fetches /api/state once more to render the
        summary, so tearing the server down inside the request handler would
        kill that fetch.
        """
        if self.index >= len(self.queue) and not self.done.is_set():
            threading.Timer(1.5, self.done.set).start()

    # -- current ---------------------------------------------------------- #
    def current(self):
        if self.index >= len(self.queue):
            return None
        return self.queue[self.index]

    def state(self):
        entry = self.current()
        if entry is None:
            return {
                "done": True,
                "written": [{"folder": str(p), "lines": lines} for p, lines in self.written],
                "skipped": self.skipped,
            }
        frame = self.frame[id(entry)]
        return {
            "done": False,
            "index": self.index,
            "total": len(self.queue),
            "folder": str(entry["folder"]),
            "rel": entry["rel"],
            "anchor_file": str(entry["folder"] / "anchor.txt"),
            "frame": frame,
            "frame_count": len(entry["images"]),
            "image_name": entry["images"][frame].name,
            "image_url": f"/api/image?i={self.index}&f={frame}",
            "names": self.args.names,
            "dist_cm": self.args.dist_cm,
            "written": len(self.written),
            "existing": self._existing(entry),
        }

    def _existing(self, entry):
        path = Path(entry["folder"]) / "anchor.txt"
        if not path.is_file():
            return None
        return path.read_text().splitlines()

    # -- actions ---------------------------------------------------------- #
    def save(self, points):
        with self.lock:
            entry = self.current()
            if entry is None:
                return {"ok": False, "error": "no folder left"}
            if len(points) != len(self.args.names):
                return {"ok": False, "error":
                        f"expected {len(self.args.names)} points, got {len(points)}"}
            for u, v in points:
                if not (0 <= u <= 100000 and 0 <= v <= 100000):
                    return {"ok": False, "error": f"point out of range: {u},{v}"}
            path = write_anchor_file(entry["folder"], points, self.args.names, self.args.dist_cm)
            self.written.append((path, anchor_lines(points, self.args.names, self.args.dist_cm)))
            self.index += 1
            self._maybe_finish()
            return {"ok": True, "path": str(path),
                    "lines": anchor_lines(points, self.args.names, self.args.dist_cm)}

    def skip(self):
        with self.lock:
            entry = self.current()
            if entry is None:
                return {"ok": False, "error": "no folder left"}
            self.skipped.append(str(entry["folder"]))
            self.index += 1
            self._maybe_finish()
            return {"ok": True}

    def next_frame(self):
        with self.lock:
            entry = self.current()
            if entry is None:
                return {"ok": False, "error": "no folder left"}
            self.frame[id(entry)] = (self.frame[id(entry)] + 1) % len(entry["images"])
            return {"ok": True, "frame": self.frame[id(entry)]}

    def finish(self):
        self.index = len(self.queue)
        self.done.set()
        return {"ok": True}


def image_meta(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return list(im.size)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# HTTP layer                                                                  #
# --------------------------------------------------------------------------- #
def make_handler(session):
    from http.server import BaseHTTPRequestHandler

    class _H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # keep the terminal readable
            pass

        # -- replies ------------------------------------------------------ #
        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(json.dumps(obj).encode(), "application/json", code)

        def _body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return {}

        # -- GET ---------------------------------------------------------- #
        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/api/state":
                return self._json(session.state())
            if path == "/api/image":
                return self._image()
            return self._send(b"not found", "text/plain", 404)

        def _image(self):
            entry = session.current()
            if entry is None:
                return self._send(b"done", "text/plain", 404)
            frame = session.frame[id(entry)]
            src = entry["images"][frame]
            try:
                data = src.read_bytes()
            except OSError as exc:
                return self._send(str(exc).encode(), "text/plain", 500)
            ctype = mimetypes.guess_type(src.name)[0] or "application/octet-stream"
            size = image_meta(src)
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if size:
                self.send_header("X-Image-Size", f"{size[0]}x{size[1]}")
            self.end_headers()
            self.wfile.write(data)

        # -- POST --------------------------------------------------------- #
        def do_POST(self):
            path = self.path.split("?")[0]
            body = self._body()
            if path == "/api/save":
                points = body.get("points") or []
                try:
                    points = [(float(p[0]), float(p[1])) for p in points]
                except (TypeError, ValueError, IndexError):
                    return self._json({"ok": False, "error": "bad points"}, 400)
                return self._json(session.save(points))
            if path == "/api/skip":
                return self._json(session.skip())
            if path == "/api/next-frame":
                return self._json(session.next_frame())
            if path == "/api/quit":
                self._json(session.finish())
                return
            return self._send(b"not found", "text/plain", 404)

    return _H


def serve(session, port):
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(session))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


def open_browser(url):
    """WSL-friendly: try the Windows side first, then XDG, then webbrowser."""
    for cmd in (["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
                 "-NoProfile", "-Command", f"Start-Process '{url}'"],
                ["explorer.exe", url],
                ["xdg-open", url]):
        try:
            if cmd[0].startswith("/") and not Path(cmd[0]).exists():
                continue
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            continue
    try:
        return webbrowser.open(url)
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# page                                                                        #
# --------------------------------------------------------------------------- #
PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>anchor picker</title>
<style>
  :root { --bg:#0f1117; --fg:#e8e8ea; --dim:#8b8fa3; --accent:#5ee0c0; --warn:#ffb454; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 ui-sans-serif,system-ui,"Segoe UI",sans-serif; height:100vh; display:flex; flex-direction:column; }
  header { padding:10px 16px; border-bottom:1px solid #23262f; display:flex; gap:18px; align-items:baseline; flex-wrap:wrap; }
  header h1 { font-size:15px; margin:0; font-weight:600; letter-spacing:.02em; }
  .box { color:var(--dim); }
  .box b { color:var(--fg); font-weight:600; }
  #rel { color:var(--accent); font-weight:600; }
  main { flex:1; position:relative; overflow:hidden; cursor:crosshair; background:#07080b; }
  #wrap { position:absolute; left:0; top:0; transform-origin:0 0; will-change:transform; }
  #img { display:block; image-rendering:auto; user-select:none; -webkit-user-drag:none; }
  .marker { position:absolute; border-radius:50%; border:2px solid var(--c);
            box-shadow:0 0 12px var(--c); pointer-events:none;
            transform:translate(-50%,-50%); }
  .marker span { position:absolute; left:50%; top:100%; transform:translate(-50%,4px);
                 background:#000c; color:var(--c); border:1px solid var(--c); border-radius:4px;
                 padding:1px 5px; font-size:11px; white-space:nowrap; font-variant-numeric:tabular-nums; }
  footer { border-top:1px solid #23262f; padding:10px 16px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  button { background:#1b1e26; color:var(--fg); border:1px solid #2e323d; border-radius:7px;
           padding:7px 13px; font:inherit; cursor:pointer; }
  button:hover:not(:disabled) { border-color:#4a5060; background:#222634; }
  button:disabled { opacity:.4; cursor:not-allowed; }
  button.primary { background:var(--accent); color:#06120f; border-color:var(--accent); font-weight:700; }
  #status { margin-left:auto; color:var(--dim); font-variant-numeric:tabular-nums; }
  #err { color:var(--warn); }
  #readout { color:var(--accent); font-family:ui-monospace,monospace; }
  #done { padding:40px; overflow:auto; }
  #done pre { background:#151821; padding:14px; border-radius:8px; overflow:auto; }
  kbd { background:#1b1e26; border:1px solid #2e323d; border-radius:4px; padding:0 5px; font-size:12px; }
</style></head><body>
<header>
  <h1>anchor picker</h1>
  <span class="box"><b id="rel">-</b> <span id="prog"></span></span>
  <span class="box">frame <b id="frame">1/1</b> <span id="imgname"></span></span>
  <span class="box">write to <b id="target">-</b></span>
</header>
<main id="view">
  <div id="wrap"><img id="img" alt=""><div id="overlay"></div></div>
</main>
<footer>
  <button id="undo">Undo</button>
  <button id="clear">Clear</button>
  <button id="nextframe">Next frame (N)</button>
  <button id="skip">Skip folder (S)</button>
  <button id="save" class="primary" disabled>Save &amp; next (Enter)</button>
  <button id="quit">Finish (Q)</button>
  <span id="readout">-</span>
  <span id="err"></span>
  <span id="status"></span>
</footer>
<script>
const COLORS = ['#ff4040','#20d6ff','#ffc400','#aa6eff','#3ce678','#ff6ebe'];
const view = document.getElementById('view');
const wrap = document.getElementById('wrap');
const img = document.getElementById('img');
const overlay = document.getElementById('overlay');
const saveBtn = document.getElementById('save');
const readout = document.getElementById('readout');
const errBox = document.getElementById('err');
let state = null, points = [], scale = 1, tx = 0, ty = 0;
let dragging = false, moved = false, lastX = 0, lastY = 0;

function setErr(msg) { errBox.textContent = msg || ''; }
function fit() {
  const nw = img.naturalWidth, nh = img.naturalHeight;
  if (!nw) return;
  scale = Math.min(view.clientWidth / nw, view.clientHeight / nh) * 0.98;
  tx = (view.clientWidth - nw * scale) / 2;
  ty = (view.clientHeight - nh * scale) / 2;
  apply();
}
function apply() {
  wrap.style.transform = `translate(${tx}px,${ty}px) scale(${scale})`;
  wrap.style.width = img.naturalWidth + 'px';
  wrap.style.height = img.naturalHeight + 'px';
  for (const m of overlay.children) m.style.transform =
    `translate(-50%,-50%) scale(${1 / Math.max(scale, 1e-6)})`;
}
function renderMarkers() {
  overlay.innerHTML = '';
  points.forEach((p, i) => {
    const d = document.createElement('div');
    d.className = 'marker';
    d.style.cssText = `left:${p[0]}px;top:${p[1]}px;width:${26}px;height:${26}px;
      border-color:${COLORS[i % COLORS.length]};--c:${COLORS[i % COLORS.length]}`;
    d.style.borderColor = COLORS[i % COLORS.length];
    d.style.setProperty('--c', COLORS[i % COLORS.length]);
    d.innerHTML = `<span>${state.names[i]} ${Math.round(p[0])},${Math.round(p[1])}</span>`;
    overlay.appendChild(d);
  });
  apply();
  saveBtn.disabled = points.length !== state.names.length;
  document.getElementById('undo').disabled = !points.length;
  document.getElementById('clear').disabled = !points.length;
}
function toImage(clientX, clientY) {
  const r = img.getBoundingClientRect();
  return [(clientX - r.left) * img.naturalWidth / r.width,
          (clientY - r.top) * img.naturalHeight / r.height];
}
async function loadState() {
  const r = await fetch('/api/state');
  state = await r.json();
  setErr('');
  if (state.done) return showDone();
  document.getElementById('rel').textContent = state.rel;
  document.getElementById('prog').textContent = `(${state.index + 1}/${state.total})`;
  document.getElementById('frame').textContent = `${state.frame + 1}/${state.frame_count}`;
  document.getElementById('imgname').textContent = state.image_name;
  document.getElementById('target').textContent = state.anchor_file;
  points = [];
  errBox.textContent = state.existing ? 'folder had an anchor.txt - overwriting' : '';
  img.onload = () => { overlay.style.width = '0'; fit(); renderMarkers(); };
  img.src = state.image_url + '&t=' + Date.now();
  document.getElementById('undo').disabled = true;
  document.getElementById('clear').disabled = true;
  saveBtn.disabled = true;
  document.getElementById('status').textContent = `written ${state.written}`;
}
function showDone() {
  view.innerHTML = `<div id="done"><h2>All folders done</h2>
    <p>Wrote ${state.written.length} anchor.txt file(s). You can close this tab.</p>
    <pre>${state.written.map(w => w.folder + '\n  ' + w.lines.join('\n  ')).join('\n\n') || '(none)'}</pre>
    ${state.skipped.length ? '<p>Skipped: ' + state.skipped.join(', ') + '</p>' : ''}</div>`;
  document.querySelector('footer').style.display = 'none';
  document.querySelector('header').style.display = 'none';
}
// --- interaction ----------------------------------------------------------
view.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (!img.naturalWidth) return;
  const factor = Math.exp(-e.deltaY * 0.0016);
  const ns = Math.min(12, Math.max(0.03, scale * factor));
  const k = ns / scale;
  const r = view.getBoundingClientRect();
  tx = (e.clientX - r.left) - ((e.clientX - r.left) - tx) * k;
  ty = (e.clientY - r.top) - ((e.clientY - r.top) - ty) * k;
  scale = ns;
  apply();
}, { passive: false });

view.addEventListener('mousedown', (e) => {
  dragging = true; moved = false; lastX = e.clientX; lastY = e.clientY;
  if (e.button === 1 || e.button === 2) e.preventDefault();
});
window.addEventListener('mousemove', (e) => {
  if (img.naturalWidth) {
    const [u, v] = toImage(e.clientX, e.clientY);
    const inside = u >= 0 && v >= 0 && u < img.naturalWidth && v < img.naturalHeight;
    readout.textContent = inside ? `x ${Math.round(u)}  y ${Math.round(v)}` : '-';
  }
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
  tx += dx; ty += dy; lastX = e.clientX; lastY = e.clientY;
  if (moved) apply();
});
window.addEventListener('mouseup', (e) => {
  const wasDragging = dragging, didMove = moved;
  dragging = false;
  if (!wasDragging || didMove) return;
  if (e.target !== img && e.target !== overlay) return;
  if (!state || state.done) return;
  if (e.button !== 0) return;
  if (points.length >= state.names.length) return;
  const [u, v] = toImage(e.clientX, e.clientY);
  if (u < 0 || v < 0 || u >= img.naturalWidth || v >= img.naturalHeight) return;
  points.push([u, v]);
  renderMarkers();
});
view.addEventListener('contextmenu', (e) => e.preventDefault());

// --- actions --------------------------------------------------------------
async function post(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                               body: JSON.stringify(body || {}) });
  return r.json();
}
async function doSave() {
  if (points.length !== state.names.length) return;
  const res = await post('/api/save', { points });
  if (!res.ok) { setErr(res.error || 'save failed'); return; }
  await loadState();
}
async function doSkip() { await post('/api/skip'); await loadState(); }
async function doNextFrame() { await post('/api/next-frame'); await loadState(); }
async function doQuit() { await post('/api/quit'); await loadState(); }

saveBtn.onclick = doSave;
document.getElementById('undo').onclick = () => { points.pop(); renderMarkers(); };
document.getElementById('clear').onclick = () => { points = []; renderMarkers(); };
document.getElementById('skip').onclick = doSkip;
document.getElementById('nextframe').onclick = doNextFrame;
document.getElementById('quit').onclick = doQuit;

window.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); if (!saveBtn.disabled) doSave(); }
  else if (e.key === 'Backspace') { e.preventDefault(); points.pop(); renderMarkers(); }
  else if (e.key === 'Escape') { points = []; renderMarkers(); }
  else if (e.key === 'n' || e.key === 'N') doNextFrame();
  else if (e.key === 's' || e.key === 'S') doSkip();
  else if (e.key === 'q' || e.key === 'Q') doQuit();
});
window.addEventListener('resize', () => { });
loadState();
</script></body></html>
"""


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)
    folders = find_folders(args)
    if not folders:
        sys.exit(f"No image folders found under {args.dataset}"
                 + (f" matching {args.match}" if args.match else ""))

    session = Session(args, folders)
    print(f"Dataset: {args.dataset.resolve()}")
    print(f"Object folders with images: {len(folders)}")
    if session.skipped_existing:
        print(f"  already have anchor.txt (skipped, use --overwrite to redo): "
              f"{len(session.skipped_existing)}")
        for f in session.skipped_existing:
            print(f"    {f['rel']}")
    if not session.queue:
        sys.exit("Nothing to pick: every folder already has an anchor.txt. "
                 "Add --overwrite to redo them.")

    print(f"To label: {len(session.queue)} folder(s)")
    for f in session.queue:
        print(f"    {f['rel']}  ({len(f['images'])} image(s))")
    print("\nAnchors per folder: " + ", ".join(
        f"{n} = {d:g}cm" for n, d in zip(args.names, args.dist_cm)))

    httpd = serve(session, args.port)
    url = f"http://localhost:{args.port}/"
    print(f"\nOpen {url} and click the anchors in order "
          f"({', '.join(args.names)}).")
    if args.open_browser and open_browser(url):
        print("(browser launched)")
    print("Waiting for clicks... Ctrl+C here also quits.\n")

    try:
        session.done.wait()
    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        httpd.shutdown()

    print(f"Wrote {len(session.written)} anchor.txt file(s):")
    for path, lines in session.written:
        print(f"  {path}")
        for line in lines:
            print(f"    {line}")
    if session.skipped:
        print(f"Skipped {len(session.skipped)} folder(s):")
        for f in session.skipped:
            print(f"  {f}")
    remaining = len(session.queue) - session.index
    if remaining > 0:
        print(f"\n{remaining} folder(s) were NOT written.")

    if args.preview and session.written:
        print("\nDrawing labelled copies via label_anchors.py ...")
        cmd = [sys.executable, str(_SCRIPT_DIR / "label_anchors.py"),
               "--dataset", str(args.dataset), "--out", str(args.out)]
        subprocess.run(cmd, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
