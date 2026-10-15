"""Draw the original schematic; counts label the measured 2026-10-12 build.

Pillow is required only to regenerate this illustration, not to read the data.
No source data, model or network is accessed by the renderer.
"""
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 cubres
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 760
COLORS = {"background": "#0b1724", "panel": "#142638", "border": "#304c63",
          "text": "#edf4f7", "muted": "#adbfcc", "teal": "#6edcca", "gold": "#f3c780"}


def font(size, bold=False):
    options = (["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "DejaVuSans-Bold.ttf"] if bold
               else ["/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"])
    for path in options:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render(path):
    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["background"])
    draw = ImageDraw.Draw(image)
    def text(x, y, value, size=18, bold=False, color="text"):
        draw.text((x, y), value, fill=COLORS[color], font=font(size, bold))
    def panel(x, y, width, height, heading, lines, color="teal"):
        draw.rounded_rectangle((x, y, x + width, y + height), radius=18,
                               fill=COLORS["panel"], outline=COLORS["border"], width=2)
        draw.rounded_rectangle((x + 19, y + 21, x + 24, y + 49), radius=2, fill=COLORS[color])
        text(x + 35, y + 20, heading, 18, True, color)
        for index, value in enumerate(lines):
            text(x + 22, y + 61 + 27 * index, value, 17, False, "muted")
    def arrow(points, color="teal"):
        draw.line(points, fill=COLORS[color], width=3)
        x, y = points[-1]
        px, py = points[-2]
        if x > px:
            draw.polygon([(x, y), (x - 10, y - 6), (x - 10, y + 6)], fill=COLORS[color])
        else:
            draw.polygon([(x, y), (x - 6, y - 10), (x + 6, y - 10)], fill=COLORS[color])

    text(44, 38, "Reference evidence + mass ambiguity", 33, True)
    text(46, 84, "An original audit atlas for planning candidate-window experiments", 20, False, "muted")
    panel(44, 147, 294, 145, "MASSBANK REFERENCES", ["Literal CC0-labelled metadata", "Complete source attribution", "Raw experimental labels"])
    panel(44, 319, 294, 125, "SOURCE KEY + MASS", ["Full supplied InChIKey", "Supplied exact_mass"], "gold")
    panel(44, 469, 294, 132, "COCONUT MASS", ["436,389 sorted source rows", "Finite-positive float64 masses"], "gold")
    panel(390, 174, 323, 156, "ADMIT + JOIN EXACTLY", ["6,951 source occurrences", "616 full source keys", "Preserve rows and diagnostics"])
    panel(390, 417, 323, 176, "INCLUSIVE WINDOWS", ["q = supplied reference mass", "Closed 3 / 10 / 20 ppm bounds", "No candidate cap or ranking", "Neutral basis unqualified"], "gold")
    panel(765, 174, 391, 331, "FORKABLE EVIDENCE", ["616 source structure rows", "5,814 raw condition groups", "Complete record attribution", "0 unmatched source keys", "Explicit source and output pins", "Standard-library readers"])
    arrow([(338, 214), (390, 235)])
    arrow([(338, 378), (365, 378), (365, 284), (390, 284)], "gold")
    arrow([(551, 330), (551, 417)], "gold")
    arrow([(338, 533), (390, 533)], "gold")
    arrow([(713, 236), (765, 236)])
    arrow([(713, 495), (741, 495), (741, 429), (765, 429)], "gold")
    draw.rounded_rectangle((44, 645, 1156, 721), radius=15, fill="#213143", outline=COLORS["border"])
    text(64, 660, "Numerical mass counts. Preserved provenance. Chemistry overlap unmeasured.", 19, True)
    text(64, 690, "Reference neutral-mass basis and descriptor compatibility remain unqualified.", 17, False, "muted")
    with Path(path).open("xb") as stream:
        image.save(stream, format="PNG", optimize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    render(parser.parse_args().out)
