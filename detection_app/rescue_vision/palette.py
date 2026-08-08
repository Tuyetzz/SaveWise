"""Per-person colours, shared by the video overlay and the HTML report.

Colour is keyed on SIGHTING ID, never on a track's position in the current
frame: colour follows the entity, never its rank. A person leaving must not
repaint the survivors.

Slot order is a validated categorical theme, not a preference. Checked with the
dataviz validator against the light surface:

    first 3 slots, --pairs all : PASS  (CVD dE 9.2, normal-vision dE 24.0)
    first 6 slots, --pairs all : FAIL  (magenta<->orange dE 12.9, below the 15
                                        floor -- indistinguishable even to a
                                        full-colour-vision viewer)

The PRD assumes 1-3 people, so the first three slots carry the real case. Past
three, colour cannot carry identity alone -- which is why every label also
prints P<n>, and why the report repeats the number beside the swatch.
"""

from __future__ import annotations

PERSON_COLOURS_HEX: list[str] = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#008300",  # green
]

# Seen but not yet through the confirm cascade, so not yet counted as a person.
TENTATIVE_HEX = "#808080"


def colour_for(sighting_id: int) -> str:
    """Fixed-order slot for a sighting. Wraps past eight; P<n> disambiguates."""
    return PERSON_COLOURS_HEX[(sighting_id - 1) % len(PERSON_COLOURS_HEX)]


def hex_to_bgr(hex_colour: str) -> tuple[int, int, int]:
    """OpenCV wants BGR; hex is RGB."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)
