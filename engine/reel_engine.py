#!/usr/bin/env python3
"""Berelyn Reel engine v3. General-purpose: HTML/CSS/SVG scenes (gradients, hand-drawn marks, camera, transitions, sequences,
character poses, photos, cutouts, video clips) -> frames (headless Chromium) -> MP4 (FFmpeg), with generated or real sound effects
synced to the animation. Only the Inter font is fetched (npm). Keep this file topic-agnostic: topic-specific drawings are supplied
per Reel as SVG inside scenes.json (diagram type "svg" or sequence items)."""
import os, sys, json, subprocess, tarfile, glob, threading, functools, http.server, socketserver, wave, re, io, shutil
import numpy as np
import imageio_ffmpeg
from PIL import Image, ImageOps
from playwright.sync_api import sync_playwright

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS, SR = 1080, 1920, 30, 44100
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("REEL_WORK", "/home/claude/reel")
UPLOADS = os.environ.get("REEL_UPLOADS", "/mnt/user-data/uploads")
VIDEO_EXT = (".mp4", ".mov", ".webm", ".m4v")

# ------------------------------------------------------------------ assets
def setup_assets(work=WORK):
    a = os.path.join(work, "assets"); os.makedirs(os.path.join(a, "img"), exist_ok=True)
    if not os.path.exists(os.path.join(a, "inter-latin-800-normal.woff")):
        subprocess.run(["npm", "pack", "@fontsource/inter", "--silent"], cwd=work, check=True, capture_output=True)
        for t in glob.glob(os.path.join(work, "fontsource-inter-*.tgz")):
            with tarfile.open(t) as tf:
                for m in tf.getmembers():
                    n = m.name
                    if m.isfile() and n.endswith("-normal.woff") and "inter-latin-" in n and any(f"-{w}-" in n for w in ("500", "600", "800")):
                        m.name = os.path.basename(n); tf.extract(m, a)
    shutil.copy(os.path.join(HERE, "engine.html"), os.path.join(work, "engine.html"))
    return a

def _find(src):
    for base in (UPLOADS, WORK):
        p = os.path.join(base, src)
        if os.path.exists(p): return p
    for base in (UPLOADS, WORK):
        hits = glob.glob(os.path.join(base, "**", os.path.basename(src)), recursive=True)
        if hits: return hits[0]
    return None

def _safe(p): return re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(os.path.basename(p))[0])

def prepare_images(scenes, work=WORK, fps=FPS):
    """Find each referenced image or clip (uploads first), fix rotation, trim transparent margins, limit size, copy into assets.
    Video clips are converted to numbered JPEG frames. Returns (scenes, missing). A missing file is not an error: the engine draws a
    fallback graphic and continues."""
    out = os.path.join(work, "assets", "img"); os.makedirs(out, exist_ok=True); missing = []
    for sc in scenes:
        for im in sc.get("imgs", []):
            if im.get("file") and not im.get("src"): continue
            p = _find(im["src"])
            if not p: im["file"] = None; missing.append(im["src"]); continue
            if p.lower().endswith(VIDEO_EXT):
                name = _safe(p); d = os.path.join(work, "assets", "vid", name); os.makedirs(d, exist_ok=True)
                if not glob.glob(os.path.join(d, "*.jpg")):
                    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-t", str(im.get("maxsec", 20)), "-i", p, "-vf", f"fps={fps},scale='min(1080,iw)':-2",
                                    "-q:v", "3", os.path.join(d, "%04d.jpg")], check=True)
                n = len(glob.glob(os.path.join(d, "*.jpg")))
                im.update(mode="video", frames=n, base=f"assets/vid/{name}/", fps=fps, file=f"assets/vid/{name}/0001.jpg"); continue
            img = ImageOps.exif_transpose(Image.open(p)); img.load()
            alpha = img.mode in ("RGBA", "LA") or "transparency" in img.info
            if alpha:
                img = img.convert("RGBA"); bb = img.split()[-1].point(lambda v: 255 if v > 8 else 0).getbbox()
                if bb:
                    pad = int(0.02 * max(img.size)); img = img.crop((max(0, bb[0] - pad), max(0, bb[1] - pad), min(img.width, bb[2] + pad), min(img.height, bb[3] + pad)))
            img.thumbnail((2160, 2160))
            name = _safe(p) + (".png" if alpha else ".jpg")
            (img if alpha else img.convert("RGB")).save(os.path.join(out, name), **({} if alpha else {"quality": 92}))
            im["file"] = "assets/img/" + name
    return scenes, missing

def describe(path):
    """Size, orientation and whether a file has transparency (cutout or character candidate)."""
    im = Image.open(path); alpha = im.mode in ("RGBA", "LA") or "transparency" in im.info
    return {"size": im.size, "orientation": "portrait" if im.height > im.width else "landscape", "transparent": alpha}

def cutout(src, dst, model="u2net"):
    """Remove the background of a user-supplied photo -> transparent PNG. Needs: pip install rembg onnxruntime
    (the model downloads from GitHub on first use). Always inspect the result before using it."""
    from rembg import remove, new_session
    out = remove(Image.open(src).convert("RGB"), session=new_session(model)); out.save(dst); return dst

# ------------------------------------------------------------------ sound effects (generated)
_rng = np.random.default_rng(7)
def _t(n): return np.arange(n) / SR
def _norm(x, peak=0.9): return x / (np.max(np.abs(x)) + 1e-9) * peak
def _reverb(x, tail=0.45, mix=0.22):
    n = int(SR * tail); ir = _rng.standard_normal(n) * np.exp(-np.arange(n) / (SR * tail / 5)); ir /= np.sqrt((ir ** 2).sum())
    L = len(x) + n; y = np.fft.irfft(np.fft.rfft(x, L) * np.fft.rfft(ir, L), L)
    y = y[:L] * (np.abs(x).max() / (np.abs(y).max() + 1e-9)); out = np.zeros(L); out[:len(x)] += x * (1 - mix); out += y * mix
    return out
def _lp(x, cutoff):                      # one-pole low-pass, cutoff may vary per sample
    a = 1 - np.exp(-2 * np.pi * cutoff / SR); y = np.zeros_like(x); s = 0.0
    for i in range(len(x)): s += a[i] * (x[i] - s); y[i] = s
    return y
def sfx_pop():
    t = _t(int(SR * .14)); f = 180 + 620 * np.exp(-t * 38)
    return _norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 32) + 0.15 * _rng.standard_normal(len(t)) * np.exp(-t * 400))
def sfx_tick():
    t = _t(int(SR * .05)); return _norm((np.sin(2 * np.pi * 2400 * t) + .4 * np.sin(2 * np.pi * 4800 * t)) * np.exp(-t * 110), .7)
def sfx_whoosh(dur=.55):
    n = int(SR * dur); t = np.linspace(0, 1, n, False); noise = _rng.standard_normal(n)
    cut = 250 + 5200 * np.sin(np.pi * t) ** 1.5; y = _lp(_lp(noise, cut), cut * .9) - .3 * _lp(noise, np.full(n, 160.))
    return _norm(y * np.sin(np.pi * t) ** 2, .8)
def sfx_hit():
    t = _t(int(SR * .5)); f = 48 + 90 * np.exp(-t * 28)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)
    top = _lp(_rng.standard_normal(len(t)), np.full(len(t), 3000.)) * np.exp(-t * 60) * .6
    return _norm(_reverb(body + top, .5, .2)[: int(SR * .8)])
def sfx_chime():
    t = _t(int(SR * 1.2)); y = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * d) for f, a, d in ((1318.5, 1, 3.2), (1975.5, .6, 4.0), (2637, .25, 6)))
    return _norm(_reverb(y, .7, .3)[: int(SR * 1.6)], .6)
SFX = {"pop": sfx_pop, "tick": sfx_tick, "whoosh": sfx_whoosh, "hit": sfx_hit, "chime": sfx_chime}


def default_sfx(i, n):                    # fallback cues when the page cannot supply any
    if i == 0: return [{"name": "hit", "at": 0.0, "gain": .7}]
    if i == n - 1: return [{"name": "chime", "at": 0.25, "gain": .6}]
    return [{"name": "whoosh", "at": 0.0, "gain": .5}]

def _decode(path):                        # any audio file -> mono float array at SR
    tmp = os.path.join(WORK, "_dec.wav")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-acodec", "pcm_s16le", tmp], check=True)
    with wave.open(tmp) as w: return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768

def sfx_folder(folder):
    """Map every audio file in a folder (an unzipped ZIP or the Reel Kit sfx folder) by file name without extension."""
    return {os.path.splitext(f)[0]: os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith((".mp3", ".wav", ".ogg", ".m4a"))}

def _sfx(lib, name):
    """Find an effect by exact name, else by a keyword in the name (whoosh, hit, pop, tick, chime), else a generated pop."""
    if name in lib: return lib[name]
    for k in ("whoosh", "hit", "pop", "tick", "chime"):
        if k in name.lower(): return lib[k]
    return lib["pop"]

def build_audio(scenes, out_wav, voice=None, music=None, music_db=-22, user_sfx=None, cues=None):
    """Mix voiceover, music and effects. Effects come from a scene's explicit "sfx" list, else from cues generated by the animation
    timeline (cues), unless the scene sets "autosfx": false."""
    total = sum(s["dur"] for s in scenes); N = int(SR * (total + .6)); mix = np.zeros(N, np.float32)
    lib = {k: f() for k, f in SFX.items()}
    for k, p in (user_sfx or {}).items(): lib[k] = _decode(p)
    acc = 0.0
    for i, s in enumerate(scenes):
        ev = s["sfx"] if "sfx" in s else ([] if s.get("autosfx") is False else (cues[i] if cues else default_sfx(i, len(scenes))))
        for e in ev:
            x = _sfx(lib, e["name"]) * e.get("gain", .6) * 10 ** (-9 / 20); st = int(SR * (acc + e.get("at", e.get("t", 0))))
            if st < N: mix[st: st + len(x)] += x[: max(0, N - st)]
        acc += s["dur"]
    if music: m = _decode(music)[:N] * 10 ** (music_db / 20); mix[: len(m)] += m
    if voice: v = _decode(voice)[:N]; mix[: len(v)] += v
    pk = np.abs(mix).max()
    if pk > .95: mix *= .95 / pk
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.repeat(mix[:, None], 2, 1) * 32767).astype(np.int16).tobytes())
    return total

# ------------------------------------------------------------------ browser, preview, render
def _serve(work):
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=work)
    h.log_message = lambda *a, **k: None
    class S(socketserver.TCPServer): allow_reuse_address = True
    srv = S(("127.0.0.1", 0), h); threading.Thread(target=srv.serve_forever, daemon=True).start(); return srv

def _page(scenes, work=WORK):
    srv = _serve(work); p = sync_playwright().start(); b = p.chromium.launch()
    pg = b.new_page(viewport={"width": W, "height": H}); errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/engine.html")
    pg.evaluate("d => loadData(d)", scenes); pg.wait_for_function("window.ready===true", timeout=90000)
    return p, b, pg, srv, errs

def _close(p, b, srv): b.close(); p.stop(); srv.shutdown()

def get_cues(scenes):
    """Sound-effect cues generated from the animation timeline: one list per scene of {t, name, gain}."""
    p, b, pg, srv, errs = _page(scenes); c = pg.evaluate("getCues()"); _close(p, b, srv)
    return [[{"at": e["t"], "name": e["name"], "gain": e["gain"]} for e in sc] for sc in c]

def preview(scenes, times, outdir=None):
    """Still PNGs at given times (seconds) for approval or QA. Returns (paths, missing_images)."""
    outdir = outdir or os.path.join(WORK, "stills"); os.makedirs(outdir, exist_ok=True)
    setup_assets(); scenes, missing = prepare_images(scenes)
    p, b, pg, srv, errs = _page(scenes); paths = []
    for t in times:
        pg.evaluate(f"renderAt({t})"); f = os.path.join(outdir, f"still_{t:05.2f}s.png"); pg.screenshot(path=f); paths.append(f)
    _close(p, b, srv)
    if errs: print("PAGE ERRORS:", errs)
    return paths, missing

def _shot(pg, t, blur, fps):
    if blur < 2 or not pg.evaluate(f"fastAt({t})"):
        pg.evaluate(f"renderAt({t})"); return pg.screenshot(type="jpeg", quality=93)
    acc = None
    for j in range(blur):
        tj = max(0, t + (j / (blur - 1) - .5) * (.5 / fps)); pg.evaluate(f"renderAt({tj})")
        a = np.asarray(Image.open(io.BytesIO(pg.screenshot(type="png"))).convert("RGB"), dtype=np.float32)
        acc = a if acc is None else acc + a
    buf = io.BytesIO(); Image.fromarray((acc / blur).astype(np.uint8)).save(buf, "JPEG", quality=93); return buf.getvalue()

def _render_chunk(scenes, start, end, out, fps, blur, grain=5):
    p, b, pg, srv, errs = _page(scenes)
    ff = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", str(fps), "-i", "-",
                           *(["-vf", f"noise=alls={grain}:allf=t"] if grain else []), "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-g", "30", "-keyint_min", "30", "-sc_threshold", "0", "-an", out], stdin=subprocess.PIPE)
    for f in range(start, end): ff.stdin.write(_shot(pg, f / fps, blur, fps))
    ff.stdin.close(); ff.wait(); _close(p, b, srv)
    if errs: print("PAGE ERRORS:", errs)

PLAN = os.path.join(WORK, "render_plan.json")

def start_render(scenes, out="/mnt/user-data/outputs/reel.mp4", voice=None, music=None, music_db=-22, user_sfx=None, fps=FPS,
                 workers=None, motion_blur=0, grain=5, chunk_seconds=10):
    """Prepare a render: find images and clips, build the sound mix (effects synced to the animation), and split the video into chunks.
    Nothing is rendered yet. Then call continue_render() until it reports finished."""
    os.makedirs(os.path.dirname(out), exist_ok=True); setup_assets()
    for f in glob.glob(os.path.join(WORK, "_chunk_*.mp4")): os.remove(f)
    scenes, missing = prepare_images(scenes, fps=fps)
    cues = None if all("sfx" in s or s.get("autosfx") is False for s in scenes) else get_cues(scenes)
    wav = os.path.join(WORK, "mix.wav"); total = build_audio(scenes, wav, voice, music, music_db, user_sfx, cues)
    n = int(round(total * fps)); per = max(30, int(chunk_seconds * fps)); cj = os.path.join(WORK, "_scenes.json"); json.dump(scenes, open(cj, "w"))
    chunks = [{"s": s0, "e": min(n, s0 + per), "file": os.path.join(WORK, f"_chunk_{k:03d}.mp4"), "done": False} for k, s0 in enumerate(range(0, n, per))]
    plan = {"out": out, "fps": fps, "blur": motion_blur, "grain": grain, "n": n, "chunks": chunks, "scenes": cj, "wav": wav, "missing": missing,
            "workers": workers or max(1, min(4, os.cpu_count() or 1)), "est": 0.0, "t_used": 0.0}
    json.dump(plan, open(PLAN, "w")); return {"frames": n, "chunks": len(chunks), "missing": missing, "workers": plan["workers"]}

def continue_render(budget=240):
    """Render pending chunks for up to `budget` seconds (keep it under 280: one command may run at most 300 s), then stop and save progress.
    Call again until it returns finished=True. When every chunk is done it joins them with the audio into the final MP4."""
    import time; t0 = time.time(); plan = json.load(open(PLAN)); todo = [c for c in plan["chunks"] if not c["done"]]
    while todo:
        el = time.time() - t0
        if el > 0 and el + plan["est"] * 1.2 > budget: break
        batch, todo = todo[:plan["workers"]], todo[plan["workers"]:]; ts = time.time()
        procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--chunk", plan["scenes"], str(c["s"]), str(c["e"]), c["file"], str(plan["fps"]), str(plan["blur"]), str(plan["grain"])]) for c in batch]
        for pr, c in zip(procs, batch):
            if pr.wait() != 0: raise RuntimeError("a render worker failed")
            c["done"] = True
        plan["est"] = max(plan["est"], time.time() - ts); json.dump(plan, open(PLAN, "w"))
    plan["t_used"] += time.time() - t0; json.dump(plan, open(PLAN, "w"))
    if todo: return {"finished": False, "remaining_chunks": len(todo), "done_chunks": len(plan["chunks"]) - len(todo), "next": "call continue_render() again"}
    lst = os.path.join(WORK, "_parts.txt"); open(lst, "w").write("".join(f"file '{c['file']}'\n" for c in plan["chunks"]))
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", plan["wav"], "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", plan["out"]], check=True)
    return {"finished": True, "path": plan["out"], "missing": plan["missing"], "frames": plan["n"], "render_seconds": round(plan["t_used"], 1)}

def render(scenes, out="/mnt/user-data/outputs/reel.mp4", budget=240, **kw):
    """start_render + continue_render. Safe inside one command: it stops after `budget` seconds. If it returns finished=False, call continue_render() again."""
    info = start_render(scenes, out, **kw); r = continue_render(budget); r.update(chunks=info["chunks"], workers=info["workers"]); return r

def verify(path):
    """Duration, resolution and audio-stream check from FFmpeg."""
    err = subprocess.run([FFMPEG, "-i", path], capture_output=True, text=True).stderr
    return {"duration": re.search(r"Duration: (\S+),", err).group(1), "video": re.search(r"Video: [^\n]+", err).group(0)[:90], "audio": bool(re.search(r"Audio:", err))}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--chunk":
        _, _, cj, s, e, out, fps, blur, grain = sys.argv; _render_chunk(json.load(open(cj)), int(s), int(e), out, int(fps), int(blur), int(grain))
    else:
        if len(sys.argv) > 1 and sys.argv[1] == "--continue": print(continue_render())
        else: print(render(json.load(open(sys.argv[1] if len(sys.argv) > 1 else "scenes.json"))))
