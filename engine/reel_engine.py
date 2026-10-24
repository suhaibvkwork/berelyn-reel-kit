#!/usr/bin/env python3
"""Berelyn Reel engine v2. HTML/CSS/SVG scenes with photos, transparent PNG cutouts and code-built science diagrams
-> frames (headless Chromium) -> MP4 (FFmpeg), with generated sound effects, optional voiceover and music.
Only the Inter font is fetched (npm). Images come from the user's uploads. Background removal uses rembg (optional)."""
import os, sys, json, subprocess, tarfile, glob, threading, functools, http.server, socketserver, wave, re
import numpy as np
import imageio_ffmpeg
from PIL import Image, ImageOps
from playwright.sync_api import sync_playwright

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS, SR = 1080, 1920, 30, 44100
WORK = os.environ.get("REEL_WORK", "/home/claude/reel")
UPLOADS = os.environ.get("REEL_UPLOADS", "/mnt/user-data/uploads")

# ------------------------------------------------------------------ assets
def setup_assets(work=WORK):
    a = os.path.join(work, "assets"); os.makedirs(os.path.join(a, "img"), exist_ok=True)
    if os.path.exists(os.path.join(a, "inter-latin-800-normal.woff")): return a
    subprocess.run(["npm", "pack", "@fontsource/inter", "--silent"], cwd=work, check=True, capture_output=True)
    for t in glob.glob(os.path.join(work, "fontsource-inter-*.tgz")):
        with tarfile.open(t) as tf:
            for m in tf.getmembers():
                n = m.name
                if m.isfile() and n.endswith("-normal.woff") and "inter-latin-" in n and any(f"-{w}-" in n for w in ("500", "600", "800")):
                    m.name = os.path.basename(n); tf.extract(m, a)
    return a

def _find(src):
    for base in (UPLOADS, WORK):
        p = os.path.join(base, src)
        if os.path.exists(p): return p
    for base in (UPLOADS, WORK):
        hits = glob.glob(os.path.join(base, "**", os.path.basename(src)), recursive=True)
        if hits: return hits[0]
    return None

def prepare_images(scenes, work=WORK):
    """Find each referenced image (uploads first), fix orientation, limit size, copy to assets/img.
    Returns (scenes, missing). A missing image is not an error: the engine draws a fallback graphic and continues."""
    out = os.path.join(work, "assets", "img"); os.makedirs(out, exist_ok=True); missing = []
    for sc in scenes:
        for im in sc.get("imgs", []):
            p = _find(im["src"])
            if not p: im["file"] = None; missing.append(im["src"]); continue
            img = ImageOps.exif_transpose(Image.open(p)); img.load(); img.thumbnail((2160, 2160))
            alpha = img.mode in ("RGBA", "LA") or "transparency" in img.info
            name = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(os.path.basename(p))[0]) + (".png" if alpha else ".jpg")
            (img.convert("RGBA") if alpha else img.convert("RGB")).save(os.path.join(out, name), **({} if alpha else {"quality": 92}))
            im["file"] = "assets/img/" + name
    return scenes, missing

def describe(path):
    """Size, orientation and whether a file has transparency (cutout candidate)."""
    im = Image.open(path); alpha = im.mode in ("RGBA", "LA") or "transparency" in im.info
    return {"size": im.size, "orientation": "portrait" if im.height > im.width else "landscape", "transparent": alpha}

def cutout(src, dst, model="u2net"):
    """Remove the background of a user-supplied photo -> transparent PNG. Needs: pip install rembg onnxruntime
    (the model downloads from GitHub on first use, about 12 s). Always inspect the result before using it."""
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

def default_sfx(i, n):                    # restrained defaults when a scene has no "sfx" key
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

def build_audio(scenes, out_wav, voice=None, music=None, music_db=-22, user_sfx=None):
    total = sum(s["dur"] for s in scenes); N = int(SR * (total + .6)); mix = np.zeros(N, np.float32)
    lib = {k: f() for k, f in SFX.items()}
    for k, p in (user_sfx or {}).items(): lib[k] = _decode(p)       # user files override generated ones by name
    acc = 0.0
    for i, s in enumerate(scenes):
        for e in s.get("sfx", default_sfx(i, len(scenes))):
            x = _sfx(lib, e["name"]) * e.get("gain", .6) * 10 ** (-9 / 20); st = int(SR * (acc + e.get("at", 0)))
            mix[st: st + len(x)] += x[: max(0, N - st)]
        acc += s["dur"]
    if music: m = _decode(music)[:N] * 10 ** (music_db / 20); mix[: len(m)] += m
    if voice: v = _decode(voice)[:N]; mix[: len(v)] += v
    pk = np.abs(mix).max()
    if pk > .95: mix *= .95 / pk
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.repeat(mix[:, None], 2, 1) * 32767).astype(np.int16).tobytes())
    return total

# ------------------------------------------------------------------ browser engine
ENGINE_HTML = r'''<!doctype html><html><head><meta charset="utf-8"><style>
@font-face{font-family:Inter;font-weight:500;src:url(assets/inter-latin-500-normal.woff)}
@font-face{font-family:Inter;font-weight:600;src:url(assets/inter-latin-600-normal.woff)}
@font-face{font-family:Inter;font-weight:800;src:url(assets/inter-latin-800-normal.woff)}
html,body{margin:0;width:1080px;height:1920px;overflow:hidden;font-family:Inter,sans-serif;-webkit-font-smoothing:antialiased}
#bg{position:absolute;inset:0}
.scene{position:absolute;inset:0;display:none;z-index:3}.light{color:#181614}.dark{color:#fff}
.txt{position:absolute;left:90px;width:850px;z-index:2}
h1,h2{margin:0;font-weight:800;letter-spacing:-.032em;line-height:1.05}
.w{display:inline-block;margin-right:.26em;will-change:transform,opacity}.w.pn{margin-left:-.26em}
.hl{color:#F86939}.hw{position:relative;display:inline-block}.pu{color:#181614}.dark .pu{color:#fff}.hl .ul{position:absolute;left:0;right:0;bottom:-.06em;height:.09em;background:#F86939;border-radius:99px;transform-origin:left;transform:scaleX(0)}
.body{margin:44px 0 0;font-weight:500;line-height:1.28}.light .body{color:#181614}.dark .body{color:rgba(255,255,255,.88)}
.it{display:flex;align-items:center;gap:30px;margin-top:46px}.num{flex:none;width:80px;height:80px;border-radius:22px;background:#F86939;color:#fff;font-weight:800;font-size:42px;line-height:80px;text-align:center}
.itx{font-size:54px;font-weight:600;line-height:1.14;letter-spacing:-.015em}.o{color:#F86939}
.card{border-radius:40px;padding:50px 56px}.light .card{background:rgba(24,22,20,.06)}.dark .card{background:rgba(255,255,255,.08)}
.lab{font-size:28px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#6B645E;margin-bottom:18px}.lab.or{color:#F86939}.dark .lab{color:#b9b2aa}.dark .lab.or{color:#F86939}
.ct{font-size:60px;font-weight:800;line-height:1.1;letter-spacing:-.025em}
.vsw{height:64px;position:relative}.vs{position:absolute;left:50%;top:50%;width:104px;height:104px;margin:-52px 0 0 -52px;border-radius:50%;background:#F86939;color:#fff;font-weight:800;font-size:34px;line-height:104px;text-align:center;z-index:5}
.big{font-weight:800;letter-spacing:-.045em;line-height:.95;color:#F86939}.lbl{font-size:58px;font-weight:600;line-height:1.14;margin-top:28px;letter-spacing:-.02em}
.im{position:absolute;z-index:1;will-change:transform,opacity}.im.full{z-index:0;overflow:hidden}
.im.card{border-radius:44px;overflow:hidden;box-shadow:0 30px 60px rgba(24,22,20,.18);padding:0;background:none}
.im img{width:100%;height:100%;object-fit:cover;display:block}.im.cutout img{object-fit:contain;filter:drop-shadow(0 40px 36px rgba(24,22,20,.28))}
.ov{position:absolute;inset:0;background:linear-gradient(to top,rgba(24,22,20,.9) 0%,rgba(24,22,20,.45) 55%,rgba(24,22,20,.3) 100%)}
.im.fb{background:rgba(24,22,20,.06);border-radius:44px;box-shadow:none}.ficon{position:absolute;inset:10%}
.dg{position:absolute;left:90px;width:850px;height:780px;z-index:2}.dg svg{width:850px;height:780px;overflow:visible}
.rv,.grow{transform-box:fill-box;transform-origin:center}.grow{transform-origin:left}
</style></head><body><div id="bg"></div>
<script>
const clamp=(x,a=0,b=1)=>Math.min(b,Math.max(a,x)), eo=t=>1-Math.pow(1-clamp(t),3);
const bg=document.getElementById('bg'),INK='#181614',OR='#F86939';
let S=[],els=[];
const inline=s=>s.replace(/\*([^*]+)\*/g,'<span class="o">$1</span>');
function words(str,base,step=.07){let i=0,out='';const parts=str.split(/(\*[^*]+\*)/).filter(Boolean);
  for(let k=0;k<parts.length;k++){const p=parts[k];
    if(p[0]==='*'){let pu='';const nx=parts[k+1];if(nx){const m=nx.match(/^[.,!?:;]+/);if(m){pu=m[0];parts[k+1]=nx.slice(m[0].length);}}
      out+=`<span class="w hl" data-d="${(base+i*step).toFixed(3)}"><span class="hw">${p.slice(1,-1)}<i class="ul"></i></span>${pu?`<span class="pu">${pu}</span>`:''}</span>`;i++;}
    else p.trim().split(/\s+/).filter(Boolean).forEach(wd=>{const pn=/^[.,!?:;]+$/.test(wd)?' pn':'';out+=`<span class="w${pn}" data-d="${(base+i*step).toFixed(3)}">${wd}</span>`;i++;});}
  return {html:out,end:base+i*step};}
// ---------- fallback icons (used when an image is missing)
const sv=b=>`<svg viewBox="0 0 400 400" width="100%" height="100%" fill="none" stroke-linecap="round" stroke-linejoin="round">${b}</svg>`;
const ICON={
 strands:sv([0,1,2,3,4].map(i=>`<path d="M${90+i*55} 60 C ${60+i*55} 140, ${130+i*55} 220, ${90+i*55} 340" stroke="${i==2?OR:INK}" stroke-width="10"/>`).join('')),
 drop:sv(`<path d="M200 50 C 200 50 90 190 90 260 a110 110 0 0 0 220 0 C 310 190 200 50 200 50 Z" stroke="${INK}" stroke-width="10"/><path d="M150 270 a55 55 0 0 0 40 52" stroke="${OR}" stroke-width="10"/>`),
 check:sv(`<circle cx="200" cy="200" r="130" stroke="${INK}" stroke-width="10"/><path d="M135 205 L185 255 L270 150" stroke="${OR}" stroke-width="14"/>`)};
// ---------- images
function addImgs(el,sc){(sc.imgs||[]).forEach(im=>{
  const m=(im.mode==='full'&&!im.file)?'card':(im.mode||'card');const d=document.createElement('div');d.className='im '+m+(im.file?'':' fb');
  const r=m==='full'?{x:0,y:0,w:1080,h:1920}:m==='cutout'?{x:im.x??90,y:im.y??800,w:im.w??850,h:im.h??660}:{x:im.x??90,y:im.y??880,w:im.w??850,h:im.h??580};
  d.style.cssText=`left:${r.x}px;top:${r.y}px;width:${r.w}px;height:${r.h}px`;d.dataset.d=im.at??.15;d.dataset.anim=im.anim||(m==='cutout'?'rise':'zoom');
  d.innerHTML=im.file?`<img src="${im.file}">`:`<div class="ficon">${ICON[im.fallback||'strands']}</div>`;
  if(m==='full'&&im.file)d.insertAdjacentHTML('beforeend','<div class="ov"></div>');el.insertBefore(d,el.firstChild);});}
// ---------- diagrams (labels and focus come from the scene; no numbers are drawn unless supplied)
const T=(x,y,t,o={})=>{const ls=String(t).split('\n');return `<text class="rv" data-d="${o.d??1}" x="${x}" y="${y}" font-family="Inter" font-size="${o.s||44}" font-weight="${o.w||700}" fill="${o.f||INK}" text-anchor="${o.a||'start'}">${ls.map((l,i)=>`<tspan x="${x}" dy="${i?(o.s||44)*1.15:0}">${l}</tspan>`).join('')}</text>`;};
const LN=(d,dl,c=INK,w=4)=>`<path class="draw" pathLength="1" data-d="${dl}" d="${d}" fill="none" stroke="${c}" stroke-width="${w}" stroke-linecap="round"/>`;
function dLayers(g){const L=g.labels||['Epidermis','Dermis','Subcutaneous\nfat'],f=g.focus??0,Hs=[80,190,230],Y0=170;let y=Y0,o='';
  const col=i=>i===f?OR:['rgba(24,22,20,.30)','rgba(24,22,20,.17)','rgba(24,22,20,.08)'][i];
  [100,200,300].forEach((x,i)=>o+=LN(`M${x} ${Y0} C ${x-22} ${Y0-55}, ${x+30} ${Y0-90}, ${x+8} ${Y0-130}`,(.4+i*.12).toFixed(2),INK,7));
  Hs.forEach((h,i)=>{const d=(.5+i*.5).toFixed(2),p=i===0?`M40 ${y+h} V ${y+28} Q40 ${y} 68 ${y} H392 Q420 ${y} 420 ${y+28} V ${y+h} Z`:`M40 ${y} H420 V ${y+h} H40 Z`,two=String(L[i]||'').includes('\n');
    o+=`<path class="rv" data-d="${d}" d="${p}" fill="${col(i)}"/>`+LN(`M420 ${y+h/2} H455`,d)+T(470,y+h/2+(two?-4:15),L[i]||'',{d,f:i===f?OR:INK});y+=h;});return ['0 0 790 700',o];}
function dFollicle(g){const L=g.labels||['Hair shaft','Oil gland','Follicle','Hair bulb'],f=g.focus??0,c=(i,df)=>i===f?OR:df;
  return ['0 0 770 680',`<path class="rv" data-d=".4" d="M40 215 H430 V640 H40 Z" fill="rgba(24,22,20,.07)"/><path class="rv" data-d=".4" d="M40 215 V190 Q40 170 60 170 H410 Q430 170 430 190 V215 Z" fill="rgba(24,22,20,.22)"/>`
  +`<path class="rv" data-d=".7" d="M235 215 V470" stroke="rgba(24,22,20,.22)" stroke-width="46" stroke-linecap="round" fill="none"/>`
  +`<path class="draw" pathLength="1" data-d=".9" d="M235 40 V440" stroke="${c(0,INK)}" stroke-width="16" stroke-linecap="round" fill="none"/>`
  +`<ellipse class="rv" data-d="1.2" cx="330" cy="300" rx="52" ry="34" fill="${c(1,'rgba(24,22,20,.30)')}"/><path class="draw" pathLength="1" data-d="1.2" d="M280 300 H258" stroke="rgba(24,22,20,.4)" stroke-width="8" fill="none"/>`
  +`<circle class="rv" data-d="1.6" cx="235" cy="490" r="54" fill="${c(3,'rgba(24,22,20,.18)')}"/><circle class="rv" data-d="1.6" cx="235" cy="505" r="22" fill="${c(3,OR)}"/>`
  +LN('M258 70 H500',.9)+T(516,85,L[0],{d:.9,f:c(0,INK)})+LN('M382 300 H500',1.2)+T(516,315,L[1],{d:1.2,f:c(1,INK)})
  +LN('M262 390 H500',1.4)+T(516,405,L[2],{d:1.4,f:c(2,INK)})+LN('M290 490 H500',1.6)+T(516,505,L[3],{d:1.6,f:c(3,INK)})];}
function dCycle(g){const L=g.labels||['Growth','Transition','Rest'],f=g.focus??0,cx=425,cy=290,r=200,pt=a=>[cx+r*Math.cos(a*Math.PI/180),cy+r*Math.sin(a*Math.PI/180)];
  const seg=[[-90,141],[147,189],[195,267]],cs=['rgba(24,22,20,.78)','rgba(24,22,20,.45)','rgba(24,22,20,.22)'];let o='';
  seg.forEach(([a,b],i)=>{const [x1,y1]=pt(a),[x2,y2]=pt(b),d=(.4+i*.6).toFixed(2),col=i===f?OR:cs[i];
    o+=`<path class="draw" pathLength="1" data-d="${d}" d="M${x1} ${y1} A${r} ${r} 0 ${b-a>180?1:0} 1 ${x2} ${y2}" fill="none" stroke="${col}" stroke-width="70" stroke-linecap="butt"/>`
     +`<rect class="rv" data-d="${d}" x="150" y="${580+i*70}" width="40" height="40" rx="11" fill="${col}"/>`+T(212,612+i*70,L[i]||'',{d,f:i===f?OR:INK});});return ['0 0 850 780',o+T(cx,cy+16,g.center||'Hair cycle',{d:1.8,s:46,w:800,a:'middle'})];}
function dIceberg(g){const L=g.labels||['What you see','What is\nunderneath'],wy=250;
  return ['0 0 850 700',`<polygon class="rv" data-d=".9" points="300,${wy} 380,110 450,170 520,${wy}" fill="${OR}"/>`
  +`<polygon class="rv" data-d=".5" points="190,${wy} 640,${wy} 700,420 560,620 340,650 160,470" fill="rgba(24,22,20,.13)"/>`
  +LN(`M0 ${wy} Q 60 ${wy-14} 120 ${wy} T 240 ${wy} T 360 ${wy} T 480 ${wy} T 600 ${wy} T 720 ${wy} T 850 ${wy}`,.3,INK,6)
  +T(540,130,L[0],{d:1.1,s:42,f:OR,w:800})+T(430,440,L[1],{d:1.4,s:44,w:800,a:'middle'})];}
function dBars(g){const V=g.values||[],f=g.focus??0,mx=Math.max(...V.map(v=>v.value),1);let o='';
  V.forEach((v,i)=>{const y=20+i*150,d=(.5+i*.45).toFixed(2),wd=Math.max(40,v.value/mx*640),c=i===f?OR:INK;
    o+=T(0,y+36,v.label,{d,s:42,w:600})+`<rect class="rv" data-d="${d}" x="0" y="${y+56}" width="850" height="56" rx="28" fill="rgba(24,22,20,.07)"/>`
    +`<rect class="grow" data-d="${d}" x="0" y="${y+56}" width="${wd}" height="56" rx="28" fill="${c}"/>`+T(wd+18,y+104,v.display||v.value,{d:(+d+.3).toFixed(2),s:48,w:800,f:c});});return [`0 0 850 ${40+V.length*150}`,o];}
const DIAG={layers:dLayers,follicle:dFollicle,cycle:dCycle,iceberg:dIceberg,bars:dBars};
// ---------- scenes
function build(sc){
  const el=document.createElement('div');let L=sc.layout||'headline';const full=(sc.imgs||[]).some(i=>i.mode==='full'&&i.file),photo=L==='photo'&&full;if(L==='photo')L='headline';
  el.className='scene '+((sc.bg==='dark'||photo)?'dark':'light');let h='',top=sc.y||(photo?860:300);
  if(L==='headline'||L==='cta'){const a=words(sc.headline,.05);h+=`<div class="txt" style="top:${top}px"><h1 style="font-size:${sc.size||112}px">${a.html}</h1>`;
    if(sc.body){const b=words(sc.body,a.end+.15,.04);h+=`<p class="body" style="font-size:${sc.bodySize||50}px">${b.html}</p>`;}h+='</div>';}
  else if(L==='list'){const a=words(sc.headline,.05);h+=`<div class="txt" style="top:${top}px"><h2 style="font-size:${sc.size||88}px">${a.html}</h2>`;
    sc.items.forEach((t,i)=>{const d=a.end+.25+i*.6;h+=`<div class="it" data-d="${d.toFixed(2)}" data-nx="${(d+.6).toFixed(2)}"><div class="num">${i+1}</div><div class="itx">${inline(t)}</div></div>`;});h+='</div>';}
  else if(L==='versus'){top=sc.y||520;let a={html:'',end:0};if(sc.headline){a=words(sc.headline,.05);h+=`<div class="txt" style="top:${top-250}px"><h2 style="font-size:${sc.size||84}px">${a.html}</h2></div>`;}
    h+=`<div class="txt" style="top:${top}px"><div class="card" data-d="${(a.end+.1).toFixed(2)}"><div class="lab">${sc.left.label}</div><div class="ct">${inline(sc.left.text)}</div></div><div class="vsw"><div class="vs" data-d="${(a.end+.9).toFixed(2)}">VS</div></div><div class="card" data-d="${(a.end+1.2).toFixed(2)}"><div class="lab or">${sc.right.label}</div><div class="ct">${inline(sc.right.text)}</div></div></div>`;}
  else if(L==='stat'){h+=`<div class="txt" style="top:${top}px"><div class="big" data-d=".05" style="font-size:${sc.size||280}px">${sc.value}</div><div class="lbl" data-d=".35">${inline(sc.label)}</div></div>`;}
  else if(L==='diagram'){const a=words(sc.headline,.05);h+=`<div class="txt" style="top:${top}px"><h2 style="font-size:${sc.size||88}px">${a.html}</h2></div><div class="dg" style="top:${sc.dy||700}px"><svg viewBox="${DIAG[sc.diagram.type](sc.diagram)[0]}" preserveAspectRatio="xMinYMin meet">${DIAG[sc.diagram.type](sc.diagram)[1]}</svg></div>`;}
  el.innerHTML=h;addImgs(el,sc);document.body.appendChild(el);els.push(el);}
window.loadData=async(data)=>{S=data;S.forEach(build);await document.fonts.ready;await Promise.all([500,600,800].map(w=>document.fonts.load(`${w} 40px Inter`)));
  await Promise.all([...document.images].map(i=>i.complete?1:new Promise(r=>{i.onload=i.onerror=r;})));window.ready=true;};
window.renderAt=(t)=>{
  let i=0,acc=0;for(;i<S.length-1&&t>=acc+S[i].dur;i++)acc+=S[i].dur;const sc=S[i],lt=t-acc,at=i===0?lt+1.1:lt,el=els[i];
  bg.style.background=sc.bg==='dark'?'#181614':'#FFFFFF';els.forEach((e,j)=>e.style.display=j===i?'block':'none');
  el.style.opacity=i===S.length-1?1:1-clamp((lt-(sc.dur-.16))/.16);
  el.querySelectorAll('.w').forEach(w=>{const d=parseFloat(w.dataset.d),p=eo((at-d)/.5);w.style.opacity=p;w.style.transform=`translateY(${(1-p)*40}px)`;const u=w.querySelector('.ul');if(u)u.style.transform=`scaleX(${eo((at-d-.4)/.5)})`;});
  el.querySelectorAll('.it').forEach(x=>{const d=parseFloat(x.dataset.d),p=eo((at-d)/.45),nx=parseFloat(x.dataset.nx),dim=x.nextElementSibling?clamp((at-nx)/.3)*.6:0;x.style.opacity=p*(1-dim);x.style.transform=`translateX(${(1-p)*-50}px)`;});
  el.querySelectorAll('.card,.big,.lbl,.vs').forEach(x=>{const d=parseFloat(x.dataset.d),p=eo((at-d)/.5);x.style.opacity=p;x.style.transform=`translateY(${(1-p)*50}px)`;});
  el.querySelectorAll('.rv').forEach(x=>{const p=eo((at-parseFloat(x.dataset.d))/.5);x.style.opacity=p;x.style.transform=`translateY(${(1-p)*22}px)`;});
  el.querySelectorAll('.draw').forEach(x=>{x.style.strokeDasharray=1;x.style.strokeDashoffset=1-eo((at-parseFloat(x.dataset.d))/.7);});
  el.querySelectorAll('.grow').forEach(x=>{x.style.transform=`scaleX(${eo((at-parseFloat(x.dataset.d))/.8)})`;});
  el.querySelectorAll('.im').forEach(x=>{const d=parseFloat(x.dataset.d),p=eo((at-d)/.6),a=x.dataset.anim,im=x.firstElementChild,k=clamp(lt/sc.dur);x.style.opacity=p;
    if(x.classList.contains('cutout'))x.style.transform=`translateY(${(1-p)*140+Math.sin(lt*1.6)*8}px)`;else if(!x.classList.contains('full'))x.style.transform=`translateY(${(1-p)*60}px)`;
    if(im&&im.tagName==='IMG')im.style.transform=a==='zoom'?`scale(${1+.12*k})`:a==='pan'?`scale(1.15) translateX(${(.5-k)*50}px)`:'none';});
  return true;};
</script></body></html>'''

def _serve(work):
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=work)
    h.log_message = lambda *a, **k: None
    class S(socketserver.TCPServer): allow_reuse_address = True
    srv = S(("127.0.0.1", 0), h); threading.Thread(target=srv.serve_forever, daemon=True).start(); return srv

def _open(scenes, work=WORK):
    setup_assets(work); scenes, missing = prepare_images(scenes, work); open(os.path.join(work, "engine.html"), "w").write(ENGINE_HTML)
    srv = _serve(work); p = sync_playwright().start(); b = p.chromium.launch()
    pg = b.new_page(viewport={"width": W, "height": H}); pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/engine.html")
    pg.evaluate("d => loadData(d)", scenes); pg.wait_for_function("window.ready===true", timeout=60000)
    return p, b, pg, srv, missing

def preview(scenes, times, outdir=None):
    """Still PNGs at given times (seconds) for approval or QA. Returns (paths, missing_images)."""
    outdir = outdir or os.path.join(WORK, "stills"); os.makedirs(outdir, exist_ok=True)
    p, b, pg, srv, missing = _open(scenes); paths = []
    for t in times:
        pg.evaluate(f"renderAt({t})"); f = os.path.join(outdir, f"still_{t:05.2f}s.png"); pg.screenshot(path=f); paths.append(f)
    b.close(); p.stop(); srv.shutdown(); return paths, missing

def render(scenes, out="/mnt/user-data/outputs/reel.mp4", voice=None, music=None, music_db=-22, user_sfx=None, fps=FPS):
    """Render the MP4. Returns {"path": ..., "missing": [images that were not found and replaced by fallback graphics]}."""
    os.makedirs(os.path.dirname(out), exist_ok=True); wav = os.path.join(WORK, "mix.wav")
    total = build_audio(scenes, wav, voice, music, music_db, user_sfx)
    p, b, pg, srv, missing = _open(scenes); n = int(round(total * fps))
    ff = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", str(fps), "-i", "-", "-i", wav,
                           "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                           "-movflags", "+faststart", "-shortest", out], stdin=subprocess.PIPE)
    for f in range(n):
        pg.evaluate(f"renderAt({f / fps})"); ff.stdin.write(pg.screenshot(type="jpeg", quality=93))
    ff.stdin.close(); ff.wait(); b.close(); p.stop(); srv.shutdown(); return {"path": out, "missing": missing}

def verify(path):
    """Duration, resolution and audio-stream check from FFmpeg."""
    err = subprocess.run([FFMPEG, "-i", path], capture_output=True, text=True).stderr
    return {"duration": re.search(r"Duration: (\S+),", err).group(1), "video": re.search(r"Video: [^\n]+", err).group(0)[:90],
            "audio": bool(re.search(r"Audio:", err))}

if __name__ == "__main__":
    scenes = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "scenes.json"))
    print(render(scenes))
