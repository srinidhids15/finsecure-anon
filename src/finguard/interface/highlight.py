"""Escaped HTML highlighting of detected entities in the original text.

``highlight_html(text, entities)`` returns markup in which every character
of ``text`` appears exactly once and in order. Plain segments and entity
segments are HTML-escaped (``html.escape``), so document text can never
become markup. Entity boundaries are the detector's own offsets; repeated
values are highlighted at each detected position and nowhere else.

Labels and colours come from the fixed ``LABEL_STYLES`` table, never from
the document. Overlapping or mismatched entities raise ``ValueError``; the
hybrid detector never produces them, and resolving them here would hide a bug.
"""

from __future__ import annotations

import html
from collections.abc import Sequence

from finguard.config import ENTITY_LABELS
from finguard.schema import Entity

# label -> (display name, short tag, colour)
LABEL_STYLES: dict[str, tuple[str, str, str]] = {
    "name": ("Name", "NAME", "#6366F1"),
    "email": ("Email", "EMAIL", "#0284C7"),
    "phone": ("Phone", "PHONE", "#0D9488"),
    "address": ("Address", "ADDR", "#D97706"),
    "ssn": ("SSN", "SSN", "#DC2626"),
    "credit_card": ("Credit card", "CARD", "#C026D3"),
    "company": ("Company", "ORG", "#16A34A"),
    "url": ("URL", "URL", "#64748B"),
}
if set(LABEL_STYLES) != set(ENTITY_LABELS):
    raise RuntimeError("LABEL_STYLES must cover exactly config.ENTITY_LABELS")

CONTAINER_CLASS = "fg-doc"


def stylesheet() -> str:
    """CSS for the highlight view (static; contains no document data)."""
    rules = [
        f".{CONTAINER_CLASS}{{white-space:pre-wrap;word-break:break-word;font-family:"
        "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:0.9rem;line-height:1.9;"
        "padding:1rem 1.1rem;border:1px solid rgba(128,128,128,.25);border-radius:10px;"
        "max-height:520px;overflow:auto}",
        f".{CONTAINER_CLASS} mark{{color:inherit;border-radius:4px;padding:1px 2px}}",
        f".{CONTAINER_CLASS} mark::after{{content:attr(data-tag);font-family:system-ui,sans-serif;"
        "font-size:0.6rem;font-weight:700;letter-spacing:.04em;margin-left:3px;padding:0 3px;"
        "border-radius:3px;vertical-align:super;color:#fff}",
        ".fg-legend{display:flex;flex-wrap:wrap;gap:.4rem .9rem;font-size:.82rem;margin:.2rem 0 .6rem}",
        ".fg-legend span{display:inline-flex;align-items:center;gap:.35rem}",
        ".fg-legend i{width:.75rem;height:.75rem;border-radius:3px;display:inline-block}",
    ]
    for label, (_, _, colour) in LABEL_STYLES.items():
        rules.append(f".{CONTAINER_CLASS} mark.fg-{label}{{background:{colour}2E;"
                     f"border-bottom:2px solid {colour}}}")
        rules.append(f".{CONTAINER_CLASS} mark.fg-{label}::after{{background:{colour}}}")
    return "<style>" + "".join(rules) + "</style>"


def legend_html() -> str:
    items = "".join(
        f'<span><i style="background:{colour}"></i>{html.escape(name)}</span>'
        for name, _, colour in LABEL_STYLES.values()
    )
    return f'<div class="fg-legend">{items}</div>'


def _checked(text: str, entities: Sequence[Entity]) -> list[Entity]:
    ordered = sorted(entities, key=lambda e: (e.start, e.end))
    for i, e in enumerate(ordered):
        if e.label not in LABEL_STYLES:
            raise ValueError(f"entity at [{e.start}, {e.end}) has unknown label {e.label!r}")
        if e.end > len(text) or text[e.start:e.end] != e.text:
            raise ValueError(f"entity at [{e.start}, {e.end}) does not match the text")
        if i and e.start < ordered[i - 1].end:
            raise ValueError(f"entities at [{ordered[i - 1].start}, {ordered[i - 1].end}) and "
                             f"[{e.start}, {e.end}) overlap")
    return ordered


def highlight_html(text: str, entities: Sequence[Entity]) -> str:
    """Escaped markup for ``text`` with each entity wrapped in a labelled ``<mark>``."""
    parts = [f'<div class="{CONTAINER_CLASS}">']
    cursor = 0
    for e in _checked(text, entities):
        name, tag, _ = LABEL_STYLES[e.label]
        parts.append(html.escape(text[cursor:e.start]))
        parts.append(
            f'<mark class="fg-{e.label}" data-tag="{tag}" '
            f'title="{html.escape(name)} [{e.start}, {e.end})">'
            f"{html.escape(text[e.start:e.end])}</mark>"
        )
        cursor = e.end
    parts.append(html.escape(text[cursor:]))
    parts.append("</div>")
    return "".join(parts)
