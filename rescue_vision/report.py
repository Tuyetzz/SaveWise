"""Self-contained HTML journey report.

Images are inlined as base64 so the file opens offline in any browser with no
server and no sibling files -- it can be handed over or emailed as-is, and it
survives the demo as a leave-behind.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from rescue_vision.palette import colour_for
from rescue_vision.types import Sighting

log = logging.getLogger(__name__)


def _inline_image(out_dir: Path, rel_path: str | None) -> str | None:
    if not rel_path:
        return None
    path = out_dir / rel_path
    try:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        log.warning("could not inline %s: %s", path, exc)
        return None
    return f"data:image/jpeg;base64,{data}"


def _card(s: Sighting, out_dir: Path) -> str:
    colour = colour_for(s.sighting_id)
    src = _inline_image(out_dir, s.best_frame_path)
    img = (
        f'<img src="{src}" alt="Person {s.sighting_id}">'
        if src
        else '<div class="noimg">no image saved</div>'
    )
    dist = (
        f"{s.closest_distance_m:.1f} m"
        if s.closest_distance_m is not None
        else "not measurable"
    )
    side = "left" if s.bearing_at_peak_deg < 0 else "right"
    return f"""
    <article class="card" style="--accent:{colour}">
      <div class="thumb">{img}</div>
      <div class="body">
        <h2><span class="swatch"></span>P{s.sighting_id}</h2>
        <dl>
          <dt>First seen</dt><dd>{s.first_seen_s:.1f}s into the sweep</dd>
          <dt>Visible for</dt><dd>{s.duration_s:.1f}s ({s.frames_seen} frames)</dd>
          <dt>Peak confidence</dt><dd><strong>{s.peak_confidence:.0%}</strong></dd>
          <dt>Mean confidence</dt><dd>{s.mean_confidence:.0%}</dd>
          <dt>Direction</dt>
          <dd>{abs(s.bearing_at_peak_deg):.0f}&deg; to the {side}</dd>
          <dt>Nearest approach</dt><dd>{dist}</dd>
        </dl>
      </div>
    </article>"""


_CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;
      --line:rgba(11,11,11,.12)}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--surface:#1a1a19;
      --ink:#fff;--muted:#c3c2b7;--line:rgba(255,255,255,.14)}}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1.25rem;background:var(--bg);color:var(--ink);
     font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:60rem;margin:0 auto}
h1{font-size:1.6rem;margin:0 0 .25rem}
.lede{color:var(--muted);margin:0}
.count{font-size:3.75rem;font-weight:650;line-height:1;margin:1.5rem 0 .1rem}
.caveat{color:var(--muted);font-size:.85rem;border-top:1px solid var(--line);
        margin-top:2.25rem;padding-top:1rem}
.cards{display:grid;gap:1rem;margin-top:1.75rem;
       grid-template-columns:repeat(auto-fill,minmax(19rem,1fr))}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;
      overflow:hidden;border-left:5px solid var(--accent)}
.thumb{background:#000;aspect-ratio:4/3;display:grid;place-items:center}
.thumb img{width:100%;height:100%;object-fit:contain;display:block}
.noimg{color:#777;font-size:.85rem}
.body{padding:.9rem 1.1rem 1.1rem}
h2{font-size:1.05rem;margin:0 0 .6rem;display:flex;align-items:center;gap:.5rem}
.swatch{width:.85rem;height:.85rem;border-radius:3px;background:var(--accent);
        flex:none}
dl{display:grid;grid-template-columns:auto 1fr;gap:.3rem .9rem;margin:0;
   font-size:.9rem}
dt{color:var(--muted)}
dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
"""


def build_report(sightings: list[Sighting], out_dir: Path, duration_s: float) -> Path:
    """Render the journey report. Returns the path written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(sightings)

    if n:
        noun = "human found" if n == 1 else "humans found"
        headline = f'<div class="count">{n}</div><p class="lede">{noun}</p>'
        cards = "".join(_card(s, out_dir) for s in sightings)
    else:
        headline = '<div class="count">0</div><p class="lede">No humans detected</p>'
        cards = ""

    mins, secs = divmod(int(duration_s), 60)
    doc = f"""<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rover journey report</title>
<style>{_CSS}</style>
<body><div class="wrap">
  <h1>Rover journey report</h1>
  <p class="lede">Autonomous sweep, {mins}m {secs:02d}s</p>
  {headline}
  <div class="cards">{cards}</div>
  <p class="caveat">
    Each person above was counted <strong>once</strong>, however many frames
    they appeared in. Times are measured from the start of the sweep and
    bearings are relative to the rover's heading at that moment &mdash; the
    rover carries no odometry, so this report records <em>when</em> someone was
    seen, never <em>where</em>. The camera is fixed forward with a 53.5&deg;
    field of view and does not scan, so anyone outside that cone as the rover
    passed was not searched.
  </p>
</div></body></html>"""

    path = out_dir / "report.html"
    path.write_text(doc, encoding="utf-8")
    return path
