# Berelyn Reel Kit

Files the Berelyn Reel Studio (a Claude Project) downloads at render time.

| Folder | Contents |
|---|---|
| `engine/` | `reel_engine.py`: renders Reels from a `scenes.json` (Chromium frames + FFmpeg, Inter font fetched automatically) |
| `sfx/` | Real sound effects (CC0 licence only, see `sfx/README.md`) |
| `characters/` | Optional character images (see `characters/README.md`) |

**Version:** v1.0 (engine v2: photos, cutouts, diagrams, generated or real sound effects)

## How the Project loads it (for Claude)

```bash
mkdir -p /home/claude/kit && cd /home/claude/kit
curl -L "$KIT_URL" -o kit.zip && unzip -q -o kit.zip
pip install imageio-ffmpeg pillow numpy --break-system-packages -q
```

```python
import sys, glob
KIT_DIR = glob.glob("/home/claude/kit/*/")[0]
sys.path.insert(0, KIT_DIR + "engine")
import reel_engine as r
sfx = r.sfx_folder(KIT_DIR + "sfx")   # real effects found in the kit (may be empty)
```

Then use `r.preview(...)` and `r.render(..., user_sfx=sfx)` as described in the Project's motion graphics file.
