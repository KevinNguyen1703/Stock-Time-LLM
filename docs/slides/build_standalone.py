"""Build docs/thesis_slides.html: a single-file, self-contained version of the slide deck in docs/slides/.

The deck's sources (deck.json + slides/<id>.html) use the Slides format: 1920x1080 sections with inline
styles, plus two app-specific elements. This script turns them into plain HTML:
  <x-shape kind="arrow-right">  -> an inline SVG arrow
  <img src="/_blob/...">         -> the image embedded as a data: URI (from figures/)
and adds a small viewer: the slides scale to the window; arrow keys / PageUp / PageDown / space / click
navigate; speaker notes (<aside>) stay hidden.
Usage: python docs/slides/build_standalone.py
"""
import base64
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BLOBS = {"/_blob/6357a4b8e716d3bcb37923fc6e655dc6": os.path.join(ROOT, "figures", "framework.png")}


def arrow_svg(style):
    w = re.search(r"width:(\d+)px", style)
    h = re.search(r"height:(\d+)px", style)
    fill = re.search(r"background:(#[0-9a-fA-F]+)", style)
    w, h = int(w.group(1)) if w else 56, int(h.group(1)) if h else 28
    fill = fill.group(1) if fill else "#5b6472"
    rest = re.sub(r"(width|height|background):[^;]+;?", "", style)
    pts = f"0,{h*0.3:.1f} {w*0.6:.1f},{h*0.3:.1f} {w*0.6:.1f},{h*0.1:.1f} {w},{h/2:.1f} {w*0.6:.1f},{h*0.9:.1f} {w*0.6:.1f},{h*0.7:.1f} 0,{h*0.7:.1f}"
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img" aria-label="arrow" '
            f'style="width:{w}px;height:{h}px;flex:none;{rest}"><polygon points="{pts}" fill="{fill}"/></svg>')


def convert(html):
    html = re.sub(r'<x-shape kind="arrow-right" style="([^"]*)"></x-shape>', lambda m: arrow_svg(m.group(1)), html)
    for blob, path in BLOBS.items():
        data = base64.b64encode(open(path, "rb").read()).decode()
        html = html.replace(f'src="{blob}"', f'src="data:image/png;base64,{data}"')
    return html


CSS = """
:root { color-scheme: light; }
html, body { margin: 0; height: 100%; background: #0e1626; overflow: hidden; }
.deck-stage { position: fixed; inset: 0; display: flex; align-items: center; justify-content: center; }
.deck-scaler { width: 1920px; height: 1080px; transform-origin: center center; flex: none; position: relative; }
.deck-scaler > section { position: absolute; inset: 0; width: 1920px; height: 1080px; box-sizing: border-box;
  overflow: hidden; display: none; }
.deck-scaler > section.active { display: flex; }
section div { display: flex; flex-direction: column; min-width: 0; min-height: 0; box-sizing: border-box; }
section h1, section h2, section h3, section p { margin: 0; overflow-wrap: break-word; }
section h1 { line-height: 1.1; } section h2 { line-height: 1.15; } section h3 { line-height: 1.2; }
section p { line-height: 1.4; }
section table { border-collapse: collapse; width: 100%; border: 2px solid rgba(20,20,19,0.22); }
section th, section td { border: 2px solid rgba(20,20,19,0.22); padding: 0.35em 0.6em; text-align: left;
  vertical-align: top; font-weight: 400; }
section th { font-weight: 600; }
section aside { display: none; }
.deck-pager { position: fixed; right: 16px; bottom: 12px; font: 14px system-ui, sans-serif; color: #9aa6b8; }
"""

JS = """
(function () {
  var slides = Array.prototype.slice.call(document.querySelectorAll('.deck-scaler > section'));
  var scaler = document.querySelector('.deck-scaler'), pager = document.querySelector('.deck-pager');
  var i = Math.max(0, Math.min(slides.length - 1, parseInt((location.hash || '#1').slice(1), 10) - 1 || 0));
  function fit() { var s = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
    scaler.style.transform = 'scale(' + s + ')'; }
  function show(n) { i = Math.max(0, Math.min(slides.length - 1, n));
    slides.forEach(function (el, k) { el.classList.toggle('active', k === i); });
    pager.textContent = (i + 1) + ' / ' + slides.length; history.replaceState(null, '', '#' + (i + 1)); }
  document.addEventListener('keydown', function (e) {
    if (['ArrowRight', 'PageDown', ' ', 'Enter'].indexOf(e.key) >= 0) { show(i + 1); e.preventDefault(); }
    if (['ArrowLeft', 'PageUp', 'Backspace'].indexOf(e.key) >= 0) { show(i - 1); e.preventDefault(); }
    if (e.key === 'Home') show(0); if (e.key === 'End') show(slides.length - 1); });
  document.addEventListener('click', function (e) { show(e.clientX > window.innerWidth / 3 ? i + 1 : i - 1); });
  window.addEventListener('resize', fit); fit(); show(i);
})();
"""


def main():
    deck = json.load(open(os.path.join(HERE, "deck.json")))
    fonts = "".join(f'<link rel="stylesheet" href="{f["href"]}">\n' for f in deck["faces"].values() if "href" in f)
    slides = [convert(open(os.path.join(HERE, "slides", f"{sid}.html")).read().strip()) for sid in deck["order"]]
    page = (f'<!doctype html>\n<html lang="vi">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>{deck["title"]}</title>\n'
            f'{fonts}<style>{CSS}</style>\n</head>\n<body>\n<div class="deck-stage"><div class="deck-scaler">\n'
            + "\n".join(slides) + '\n</div></div>\n<div class="deck-pager"></div>\n'
            f'<script>{JS}</script>\n</body>\n</html>\n')
    out = os.path.join(ROOT, "docs", "thesis_slides.html")
    open(out, "w").write(page)
    print(f"wrote {out}: {len(slides)} slides, {len(page)//1024} KB")


if __name__ == "__main__":
    main()
