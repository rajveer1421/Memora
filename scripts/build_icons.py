"""Build static/icons.svg, a sprite of the Lucide line icons the pages use (Lucide is ISC-licensed).

    python scripts/build_icons.py

Pages draw an icon with  <svg class="i"><use href="/static/icons.svg#camera"/></svg>
(the Jinja helper icon("camera") and the JS helper ic("camera") write exactly that)."""
import re
import sys
import urllib.request
from pathlib import Path

VERSION = "0.469.0"
ICONS = """activity arrow-left bell brain menu calendar-days camera check chevron-right circle-alert circle-check clock cpu
eye heart heart-handshake house image info key-round layout-dashboard lightbulb link list-checks lock log-out map
map-pin message-circle-heart mic navigation package pause phone pill play plus refresh-cw scan-face search settings
shield-check smile sparkles square trash-2 triangle-alert upload user-plus user-round users video volume-2 x""".split()
OUT = Path(__file__).resolve().parent.parent / "static" / "icons.svg"
LICENSE = OUT.parent / "LICENSE-lucide.txt"


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8")


def main():
    symbols = []
    for name in ICONS:
        svg = fetch(f"https://unpkg.com/lucide-static@{VERSION}/icons/{name}.svg")
        inner = re.search(r"<svg[^>]*>(.*)</svg>", svg, re.S).group(1).strip()
        inner = re.sub(r"\s+", " ", inner)
        symbols.append(f'<symbol id="{name}" viewBox="0 0 24 24">{inner}</symbol>')
        print("ok", name)
    OUT.write_text('<svg xmlns="http://www.w3.org/2000/svg">\n'
                   f"<!-- Lucide v{VERSION} (ISC license, see LICENSE-lucide.txt) -->\n" + "\n".join(symbols) + "\n</svg>\n",
                   encoding="utf-8")
    LICENSE.write_text(fetch(f"https://unpkg.com/lucide-static@{VERSION}/LICENSE"), encoding="utf-8")
    print(f"wrote {OUT} ({len(symbols)} icons)")


if __name__ == "__main__":
    sys.exit(main())
