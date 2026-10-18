# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regenerate the figures, the example sounds and the preset gallery in docs/, with ballast itself.

    uv run python docs/build.py

Every figure is drawn from the code in this repository, as SVG that follows the reader's light or
dark theme. Every sound is rendered and exported as `ballast render` would, and docs/presets.md is
written from the shipped presets, so the gallery always shows them as they are.
"""

from __future__ import annotations

import base64
import html
import itertools
import math
import struct
import sys
import zlib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve, resample_poly, sosfreqz

from ballast_synth import audition, noise, patch, render
from ballast_synth.voice import Voice

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from numpy.typing import ArrayLike, NDArray

    from ballast_synth.curve import Curve

DOCS = Path(__file__).resolve().parent
RATE = 48000
INTERNAL = 768000  # the internal rate of the 48 kHz family
PRESETS = ("tight", "driven", "round", "long")  # shortest first
KILO = 1000
PASSBAND_HZ = 20000
STEPS = (3, 5)  # how many steps a round axis takes
LEFT = (64, 96, 380, 200)  # the two panels of a figure in halves
RIGHT = (548, 96, 380, 200)

# Chart hues and the blue ramp are validated data-visualization defaults; the neutrals lean toward the accent.
# series-* tell series apart; step-* order settings from least to most, darker on paper and lighter at night.
LIGHT = {
    "paper": "#fbfcfd",
    "panel": "#f1f4f7",
    "ink": "#0e1116",
    "ink-2": "#4a5361",
    "muted": "#626c7a",
    "rule": "#dde3ea",
    "grid": "#e9edf2",
    "accent": "#2a78d6",
    "series-2": "#eb6834",
    "series-3": "#1baf7a",
    "step-1": "#86b6ef",
    "step-2": "#3987e5",
    "step-3": "#1c5cab",
    "step-4": "#0d366b",
    "show-light": "inline",
    "show-dark": "none",
}
DARK = {
    "paper": "#0f1216",
    "panel": "#171c22",
    "ink": "#e7ecf1",
    "ink-2": "#aab4c0",
    "muted": "#8b96a4",
    "rule": "#29313b",
    "grid": "#1c2229",
    "accent": "#3987e5",
    "series-2": "#d95926",
    "series-3": "#199e70",
    "step-1": "#184f95",
    "step-2": "#2a78d6",
    "step-3": "#6da7ec",
    "step-4": "#b7d3f6",
    "show-light": "none",
    "show-dark": "inline",
}
BLUES = [  # steps 100 … 700
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]  # fmt: skip
SANS = 'system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif'
MONO = 'ui-monospace,"SF Mono","Cascadia Mono","JetBrains Mono","Roboto Mono",Menlo,Consolas,monospace'
PALETTE = "".join(f"--{k}:{v};" for k, v in LIGHT.items()), "".join(f"--{k}:{v};" for k, v in DARK.items())
LINE = "fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round"
RING = "stroke:var(--paper);stroke-width:2"
STYLE = f"""<style>
.fig{{{PALETTE[0]}}}
@media (prefers-color-scheme: dark){{.fig{{{PALETTE[1]}}}}}
.fig .bg{{fill:var(--paper)}}
.fig .grid{{fill:none;stroke:var(--grid);stroke-width:1}}
.fig .axis{{fill:none;stroke:var(--rule);stroke-width:1}}
.fig .zero{{fill:none;stroke:var(--muted);stroke-width:1;stroke-opacity:.45}}
.fig text{{font-family:{SANS};font-size:12px;fill:var(--muted);font-variant-numeric:tabular-nums}}
.fig .label{{fill:var(--ink-2)}}
.fig .title{{fill:var(--ink);font-size:15px;font-weight:600}}
.fig .name{{fill:var(--ink);font-family:{MONO};font-size:15px;font-weight:700}}
.fig .math{{fill:var(--ink-2);font-family:{MONO};font-size:12px}}
.fig .boxtitle{{fill:var(--ink);font-size:13px;font-weight:600}}
.fig .peak{{fill:var(--accent);fill-opacity:.3}}
.fig .rms{{fill:var(--accent)}}
.fig .trace{{fill:none;stroke:var(--accent);stroke-width:1.25;stroke-linejoin:round}}
.fig .s1{{{LINE};stroke:var(--accent)}}
.fig .s2{{{LINE};stroke:var(--series-2)}}
.fig .s3{{{LINE};stroke:var(--series-3)}}
.fig .o1{{{LINE};stroke:var(--step-1)}}
.fig .o2{{{LINE};stroke:var(--step-2)}}
.fig .o3{{{LINE};stroke:var(--step-3)}}
.fig .o4{{{LINE};stroke:var(--step-4)}}
.fig .ref{{fill:none;stroke:var(--muted);stroke-width:1.25}}
.fig .halo{{fill:none;stroke:var(--muted);stroke-opacity:.35;stroke-width:7}}
.fig .key1,.fig .bar{{fill:var(--accent)}}
.fig .key2{{fill:var(--series-2)}}
.fig .key3{{fill:var(--series-3)}}
.fig .ko1{{fill:var(--step-1)}}
.fig .ko2{{fill:var(--step-2)}}
.fig .ko3{{fill:var(--step-3)}}
.fig .ko4{{fill:var(--step-4)}}
.fig .kref{{fill:var(--muted)}}
.fig .khalo{{fill:var(--muted);fill-opacity:.35}}
.fig .knot,.fig .dot1{{fill:var(--accent);{RING}}}
.fig .dot2{{fill:var(--series-2);{RING}}}
.fig .dot3{{fill:var(--series-3);{RING}}}
.fig .band{{fill:var(--accent);fill-opacity:.08}}
.fig .limit{{fill:none;stroke:var(--ink-2);stroke-width:1}}
.fig .box{{fill:var(--paper);stroke:var(--rule);stroke-width:1}}
.fig .zone{{fill:var(--panel);stroke:var(--rule);stroke-width:1}}
.fig .flow{{fill:none;stroke:var(--ink-2);stroke-width:1.5}}
.fig .head{{fill:var(--ink-2)}}
.fig .light{{display:var(--show-light)}}
.fig .dark{{display:var(--show-dark)}}
</style>"""
KEY_HEIGHTS = {"khalo": 7}  # legend keys are 3 px tall lines, except a halo's


def svg(size: tuple[int, int], title: str, subtitle: str, body: str, description: str) -> str:
    """Wrap a figure's body in an SVG with its own palette, which follows the reader's light or dark theme."""
    width, height = size
    head = f'<text class="title" x="28" y="36">{title}</text><text class="label" x="28" y="57">{subtitle}</text>'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="fig" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img"><title>{html.escape(description)}</title>{STYLE}'
        f'<rect class="bg" width="{width}" height="{height}" rx="10"/>{head}{body}</svg>\n'
    )


def ramp(stops: list[str]) -> NDArray[np.uint8]:
    """Interpolate 256 colors through hex stops, as a 256 × 3 palette."""
    rgb = np.array([[int(s[i : i + 2], 16) for i in (1, 3, 5)] for s in stops], dtype=float)
    u = np.linspace(0, len(stops) - 1, 256)
    return np.round([np.interp(u, np.arange(len(stops)), rgb[:, c]) for c in range(3)]).T.astype(np.uint8)


def png(indices: NDArray[np.uint8], palette: NDArray[np.uint8]) -> str:
    """Encode an 8-bit indexed PNG, as a data URI."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    h, w = indices.shape
    raw = b"".join(b"\0" + row.tobytes() for row in indices)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 3, 0, 0, 0))
        + chunk(b"PLTE", palette.tobytes())
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(data).decode()


# Number formats for labels, with a true minus sign.


def num(v: float) -> str:
    """Format a number for a label."""
    return f"{v:g}".replace("-", "−")


def fixed5(v: float) -> str:
    """Format a number for a label with a sign and five decimals; 0 is just 0."""
    return "0" if v == 0 else f"{v:+.5f}".replace("-", "−")


def hz(v: float) -> str:
    """Format a frequency for a label, in thousands from 1 kHz."""
    return f"{v / KILO:g}k" if v >= KILO else f"{v:g}"


def note(tune: float) -> str:
    """Name the note of a frequency, with its offset in cents unless it is in tune."""
    return patch.describe_tuning(tune).removesuffix(" +0.0 cents")


def nice(hi: float) -> tuple[float, float]:
    """Find a round axis that reaches hi: the smallest end, in three to five steps of 1, 2, 2.5 or 5 × 10^k."""
    found = []
    top = math.floor(math.log10(hi))
    for exponent, mantissa in itertools.product(range(top - 2, top + 1), (1, 2, 2.5, 5)):
        step = mantissa * 10.0**exponent
        count = math.ceil(hi / step - 1e-9)
        if STEPS[0] <= count <= STEPS[1]:
            found.append((count * step, -step))
    end, step = min(found)  # the shortest axis, in the fewest steps
    return end, -step


def steps(end: float, step: float) -> tuple[float, ...]:
    """List the ticks of a round axis from 0 to its end."""
    return tuple(i * step for i in range(round(end / step) + 1))


def ticks_every(end: float, most: int) -> tuple[float, ...]:
    """List ticks from 0 to an axis's end at the smallest step of 1, 2, 2.5 or 5 × 10^k that takes `most` or fewer."""
    top = math.floor(math.log10(end))
    for exponent, mantissa in itertools.product(range(top - 2, top + 1), (1, 2, 2.5, 5)):
        step = mantissa * 10.0**exponent
        if end / step <= most and math.isclose(end / step, round(end / step)):
            return steps(end, step)
    return (0, end)


def inner(axis: Axis) -> tuple[float, ...]:
    """List round ticks inside an axis's range, three to five steps across it."""
    _, step = nice(axis.hi - axis.lo)
    first = math.ceil(axis.lo / step - 1e-9) * step
    return tuple(first + i * step for i in range(math.floor((axis.hi - first) / step + 1e-9) + 1))


class Axis(NamedTuple):
    """The data range along one side of a panel, on a linear or a log scale."""

    lo: float
    hi: float
    log: bool = False


class Ticks(NamedTuple):
    """The tick values along one axis, how to label them, and the axis's name."""

    values: Sequence[float]
    fmt: Callable[[float], str] = num
    label: str = ""


class Panel:
    """A plot area: data scaled onto the rectangle (x, y, w, h) along two axes."""

    def __init__(self, area: tuple[float, float, float, float], xs: Axis, ys: Axis) -> None:
        """Place the panel on `area`, with `xs` across it and `ys` up it."""
        self.x, self.y, self.w, self.h = area
        self.xs, self.ys = xs, ys

    @staticmethod
    def _unit(v: ArrayLike, axis: Axis) -> NDArray[np.float64]:
        a, b, log = axis
        u = np.asarray(v, dtype=float)
        if log:
            return (np.log10(u) - math.log10(a)) / (math.log10(b) - math.log10(a))
        return (u - a) / (b - a)

    def sx(self, v: ArrayLike) -> NDArray[np.float64]:
        """Place data x across the panel."""
        return self.x + self._unit(v, self.xs) * self.w

    def sy(self, v: ArrayLike) -> NDArray[np.float64]:
        """Place data y up the panel."""
        return self.y + (1 - self._unit(v, self.ys)) * self.h

    def path(self, xs: ArrayLike, ys: ArrayLike, *, close: bool = False) -> str:
        """Draw the line through the points (xs, ys), closed if asked."""
        points = "L".join(f"{x:.1f},{y:.1f}" for x, y in zip(self.sx(xs), self.sy(ys), strict=True))
        return f"M{points}{'Z' if close else ''}"

    def line(self, cls: str, xs: ArrayLike, ys: ArrayLike) -> str:
        """Draw the line through the points (xs, ys), in the style of `cls`."""
        return f'<path class="{cls}" d="{self.path(xs, ys)}"/>'

    def dots(self, cls: str, xs: ArrayLike, ys: ArrayLike, r: float = 4.5) -> str:
        """Draw a marker at each point (xs, ys)."""
        return "".join(
            f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'
            for x, y in zip(self.sx(xs), self.sy(ys), strict=True)
        )

    def vline(self, x: float, cls: str = "zero") -> str:
        """Draw a vertical line across the panel at data x."""
        return f'<path class="{cls}" d="M{self.sx(x):.1f},{self.y}V{self.y + self.h}"/>'

    def hline(self, y: float, cls: str = "zero") -> str:
        """Draw a horizontal line across the panel at data y."""
        return f'<path class="{cls}" d="M{self.x},{self.sy(y):.1f}H{self.x + self.w}"/>'

    def text(self, at: tuple[float, float], text: str, anchor: str = "start") -> str:
        """Write a label with its baseline at the data point `at`."""
        x, y = self.sx(at[0]), self.sy(at[1])
        return f'<text class="label" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{text}</text>'

    def clip(self, ident: str, content: str) -> str:
        """Clip drawings to the panel, so that a line can leave it at its edge."""
        box = f'<rect x="{self.x}" y="{self.y}" width="{self.w}" height="{self.h}"/>'
        return f'<clipPath id="{ident}">{box}</clipPath><g clip-path="url(#{ident})">{content}</g>'

    def frame(self, xticks: Ticks, yticks: Ticks, *, grid: bool = True) -> str:
        """Draw the gridlines, the baseline, the tick labels and the axis names."""
        out = []
        for v in yticks.values:
            y = self.sy(v)
            if grid:
                out.append(f'<path class="grid" d="M{self.x},{y:.1f}H{self.x + self.w}"/>')
            out.append(f'<text x="{self.x - 8}" y="{y + 4:.1f}" text-anchor="end">{yticks.fmt(v)}</text>')
        for v in xticks.values:
            x = self.sx(v)
            if grid:
                out.append(f'<path class="grid" d="M{x:.1f},{self.y}V{self.y + self.h}"/>')
            out.append(f'<text x="{x:.1f}" y="{self.y + self.h + 18}" text-anchor="middle">{xticks.fmt(v)}</text>')
        out.append(f'<path class="axis" d="M{self.x},{self.y + self.h}H{self.x + self.w}"/>')
        if xticks.label:
            right, below = self.x + self.w, self.y + self.h + 36
            out.append(f'<text class="label" x="{right}" y="{below}" text-anchor="end">{xticks.label}</text>')
        if yticks.label:
            out.append(f'<text class="label" x="{self.x}" y="{self.y - 10}">{yticks.label}</text>')
        return "".join(out)


def legend(at: tuple[float, float], keys: Iterable[tuple[str, str]]) -> str:
    """Draw a legend, a row per (class, label): a key in the series' color or marker, and the label in ink."""
    x, y = at
    out = []
    for i, (cls, text) in enumerate(keys):
        row = y + 20 * i
        if cls.startswith("dot"):
            out.append(f'<circle class="{cls}" cx="{x + 9}" cy="{row - 4}" r="5"/>')
        else:
            height = KEY_HEIGHTS.get(cls, 3)
            out.append(
                f'<rect class="{cls}" x="{x}" y="{row - 3.5 - height / 2}" width="18" height="{height}" rx="1.5"/>',
            )
        out.append(f'<text class="label" x="{x + 26}" y="{row}">{text}</text>')
    return "".join(out)


def legend_row(at: tuple[float, float], keys: Iterable[tuple[str, str]]) -> str:
    """Draw a legend on one line, for series that two panels share."""
    x, y = at
    out = []
    for cls, text in keys:
        out.append(legend((x, y), [(cls, text)]))
        x += 26 + 6.8 * len(text) + 28  # the key, the label at about 6.8 px a character, and a gap
    return "".join(out)


def below(panel: Panel) -> tuple[float, float]:
    """Return where a legend starts under a panel, clear of its axis labels."""
    return panel.x, panel.y + panel.h + 62


@dataclass
class Kick:
    """A patch, and its render at 48 kHz."""

    name: str
    text: str
    p: patch.Patch
    audio: NDArray[np.float32]
    report: render.Report


def kick(name: str, text: str) -> Kick:
    """Render a patch's text at 48 kHz, within the ceiling of 16-bit PCM."""
    p = patch.loads(text)
    return Kick(name, text, p, *render.render(p, 16))


def render_kicks() -> dict[str, Kick]:
    """Render every preset."""
    folder = resources.files("ballast_synth").joinpath("data", "presets")
    return {name: kick(name, folder.joinpath(f"{name}.toml").read_text(encoding="utf-8")) for name in PRESETS}


def design_moves(tight: Kick) -> list[tuple[str, Kick]]:
    """Render the design moves of the usage guide, each made on the tight preset: what to do, and the result."""
    p = tight.p
    moves = (
        ("tight-e1", "ballast tune kick.toml E1", patch.edit(p, "body.tune_hz", patch.parse_tune("E1"))),
        ("tight-snap", "ballast stretch kick.toml body.pitch 0.5", patch.stretch(p, "body.pitch", 0.5)),
        ("tight-long", "ballast stretch kick.toml body.amp 1.5", patch.stretch(p, "body.amp", 1.5)),
        ("tight-click", "attack.level_db = -8.0", patch.edit(p, "attack.level_db", -8.0)),
        ("tight-drive", "drive.db = 18.0", patch.edit(p, "drive.db", 18.0)),
        ("tight-cut", "output.low_cut_hz = 30.0", patch.edit(p, "output.low_cut_hz", 30.0)),
    )
    return [("ballast init tight", tight)] + [(how, kick(name, patch.dumps(q))) for name, how, q in moves]


def trace(audio: NDArray[np.float32], ms: float, columns: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reduce the first `ms` of audio to each column's lowest and highest samples, in time order."""
    x = audio[: round(ms * RATE / KILO)].astype(float)
    edges = np.linspace(0, x.size, columns + 1).astype(int)
    picks = {int(a + i) for a, b in itertools.pairwise(edges) if b > a for i in (np.argmin(x[a:b]), np.argmax(x[a:b]))}
    index = np.array(sorted(picks))
    return index / RATE * KILO, x[index]


def envelopes(
    x: NDArray[np.float32],
    columns: int,
    n: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Measure the minimum, maximum and RMS of each column of the first n samples of x, zero past its end."""
    y = np.zeros(max(n, columns))
    y[: min(n, x.size)] = x[:n]
    starts = np.arange(columns) * y.size // columns
    counts = np.diff(np.append(starts, y.size))
    rms = np.sqrt(np.add.reduceat(y * y, starts) / counts)
    return np.minimum.reduceat(y, starts), np.maximum.reduceat(y, starts), rms


def waveform(panel: Panel, x: NDArray[np.float32], seconds: float) -> str:
    """Draw the peak and RMS envelopes of x over its first `seconds`, one column per unit of width."""
    columns = round(panel.w)
    lo, hi, rms = envelopes(x, columns, round(seconds * RATE))
    t = (np.arange(columns) + 0.5) * panel.xs.hi / columns
    there_and_back = np.r_[t, t[::-1]]
    peak = panel.path(there_and_back, np.r_[hi, lo[::-1]], close=True)
    body = panel.path(there_and_back, np.r_[rms, -rms[::-1]], close=True)
    return f'<path class="peak" d="{peak}"/><path class="rms" d="{body}"/>'


def figure_presets(kicks: dict[str, Kick]) -> str:
    """Draw every preset's waveform, one row each."""
    width, left, right, top, row, gap, seconds = 960, 214, 28, 84, 62, 24, 1.5
    body = []
    for i, k in enumerate(kicks.values()):
        y = top + i * (row + gap)
        panel = Panel((left, y, width - left - right, row), Axis(0, seconds), Axis(-1, 1))
        body.append(f'<path class="grid" d="M{left},{y}H{width - right}M{left},{y + row}H{width - right}"/>')
        body.append(f'<path class="zero" d="M{left},{panel.sy(0):.1f}H{width - right}"/>')
        body.append(waveform(panel, k.audio, seconds))
        tune, length, drive = k.p.body.tune_hz, k.report.source_ms, k.p.drive.db
        body.append(f'<text class="name" x="28" y="{y + 17}">{k.name}</text>')
        body.append(f'<text class="label" x="28" y="{y + 38}">{num(tune)} Hz · {note(tune)}</text>')
        body.append(f'<text x="28" y="{y + 57}">{num(length)} ms · drive {num(drive)} dB</text>')
    end = top + 4 * (row + gap) - gap
    axis = Panel((left, end, width - left - right, 0), Axis(0, seconds), Axis(0, 1))
    body.extend(
        f'<text x="{axis.sx(v):.1f}" y="{end + 20}" text-anchor="middle">{num(1000 * v)}</text>'
        for v in np.arange(0, seconds + 1e-9, 0.25)
    )
    body.append(f'<text class="label" x="{width - right}" y="{end + 40}" text-anchor="end">time, ms</text>')
    subtitle = "Peak (light) and RMS (dark) per column, on a full-scale axis"
    return svg(
        (width, end + 60),
        "Four presets, rendered at 48 kHz",
        subtitle,
        "".join(body),
        "Waveforms of the presets",
    )


def figure_pipeline() -> str:
    """Draw the functional block diagram: from patch to file."""
    top = 76
    cy = top + 98
    body = [
        (
            '<defs><marker id="pipeline-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path class="head" d="M0,1L10,5L0,9Z"/></marker></defs>'
        ),
        f'<rect class="zone" x="16" y="{top}" width="520" height="196" rx="8"/>',
        f'<text class="label" x="32" y="{top + 22}">16 × the base rate: 705.6 or 768 kHz</text>',
        f'<text class="label" x="834" y="{top + 22}" text-anchor="middle">at the file\'s sample rate</text>',
    ]

    def box(x: float, y: float, w: float, title: str, *lines: tuple[str, str], h: float = 64) -> str:
        out = [f'<rect class="box" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>']
        out.append(f'<text class="boxtitle" x="{x + 12}" y="{y + 21}">{title}</text>')
        for i, (cls, text) in enumerate(lines):
            out.append(f'<text class="{cls}" x="{x + 12}" y="{y + 40 + 16 * i}">{text}</text>')
        return "".join(out)

    def arrow(d: str) -> str:
        return f'<path class="flow" d="{d}" marker-end="url(#pipeline-head)"/>'

    body.append(box(32, top + 36, 150, "Body", ("math", "sin φ + h₂ sin 2φ"), ("label", "pitch curve to phase")))
    body.append(box(32, top + 116, 150, "Attack", ("math", "tone + noise"), ("label", "seeded, band-limited")))
    body.append(
        f'<circle class="box" cx="214" cy="{cy}" r="14"/>'
        f'<text class="boxtitle" x="214" y="{cy + 5}" text-anchor="middle">+</text>',
    )
    body.append(arrow(f"M182,{top + 68}C192,{top + 68} 204,{cy - 30} 208,{cy - 15}"))
    body.append(arrow(f"M182,{top + 148}C192,{top + 148} 204,{cy + 30} 208,{cy + 15}"))
    body.append(box(246, cy - 32, 136, "Drive", ("math", "tanh(αx)/tanh α"), ("label", "0 dB: bypass")))
    body.append(arrow(f"M228,{cy}H244"))
    body.append(box(402, cy - 32, 118, "Low cut", ("label", "12 dB/octave"), ("label", "0 Hz: off")))
    body.append(arrow(f"M382,{cy}H400"))
    body.append(box(566, cy - 32, 128, "Decimator", ("math", "↓16, ↓8 or ↓4"), ("label", "minimum phase")))
    body.append(arrow(f"M520,{cy}H564"))
    body.append(box(720, cy - 32, 110, "Output", ("label", "gain or peak"), ("label", "polarity")))
    body.append(arrow(f"M694,{cy}H718"))
    body.append(box(850, cy - 32, 98, "WAV", ("label", "float or PCM"), ("label", "patch inside")))
    body.append(arrow(f"M830,{cy}H848"))
    subtitle = "From the patch's curves to a WAV file"
    return svg((960, top + 220), "Signal flow", subtitle, "".join(body), "Signal flow from patch to WAV")


PITCH = Axis(30, 500, log=True)  # wide enough for every preset's sweep, so that their figures compare
GAIN = Axis(0, 1.05)
LEVELS = (0, 0.5, 1)


def knotted(panel: Panel, knots: patch.Knots, curve: Curve, ticks: tuple[Ticks, Ticks], scale: float = 1) -> str:
    """Draw a curve through its knots, scaled by `scale`, with the knots that fall inside the panel."""
    t = np.linspace(0, panel.xs.hi, 600)
    kx = np.array([ms for ms, _ in knots])
    ky = scale * np.array([value for _, value in knots])
    keep = kx <= panel.xs.hi
    path = panel.line("s1", t, scale * curve(t / KILO))
    return panel.frame(*ticks) + path + panel.dots("knot", kx[keep], ky[keep])


def figure_anatomy(k: Kick) -> str:
    """Draw a preset's three curves, and the start of the sound they make."""
    v, tune, top = Voice(k.p, RATE), k.p.body.tune_hz, 96
    sweep, sweep_step = nice(1.5 * k.p.body.pitch[-1][0])
    length, length_step = nice(k.p.body.amp[-1][0])
    click, click_step = nice(k.p.attack.amp[-1][0])
    pitch = Panel((64, top, 240, 150), Axis(0, sweep), PITCH)
    shown = (Ticks(steps(sweep, sweep_step), num, "ms"), Ticks((50, 100, 200, 500), hz, "Pitch, Hz"))
    body = [knotted(pitch, k.p.body.pitch, v.pitch, shown, tune)]
    shown = (Ticks(steps(length, length_step), num, "ms"), Ticks(LEVELS, num, "Body amplitude"))
    body.append(knotted(Panel((368, top, 240, 150), Axis(0, length), GAIN), k.p.body.amp, v.body_amp, shown))
    shown = (Ticks(steps(click, click_step), num, "ms"), Ticks(LEVELS, num, "Attack amplitude"))
    body.append(knotted(Panel((672, top, 240, 150), Axis(0, click), GAIN), k.p.attack.amp, v.attack_amp, shown))
    start = tune * k.p.body.pitch[0][1]
    body.append(f'<text class="label" x="{pitch.sx(0) + 10:.1f}" y="{pitch.sy(start) + 4:.1f}">{start:.0f} Hz</text>')
    body.append(
        f'<text class="label" x="{pitch.x + pitch.w}" y="{pitch.sy(tune) - 10:.1f}" text-anchor="end">'
        f"{num(tune)} Hz, {note(tune)}</text>",
    )
    wave_top = top + 214
    sound = Panel((64, wave_top, 848, 120), Axis(0, sweep), Axis(-0.6, 0.6))
    label = f"Output, first {num(sweep)} ms, full scale ±1"
    body.append(sound.frame(Ticks(ticks_every(sweep, 10), num, "ms"), Ticks((-0.5, 0, 0.5), num, label)))
    body.append(sound.line("trace", *trace(k.audio, sweep, round(sound.w))))
    subtitle = "Three curves through the patch's knots, and the sound they make"
    description = f"The {k.name} preset's pitch, body and attack curves, and its first {num(sweep)} ms"
    return svg((960, wave_top + 172), f"Anatomy of {k.name}", subtitle, "".join(body), description)


def spectrogram(x: NDArray[np.float32], seconds: float, rows: int, columns: int) -> NDArray[np.float64]:
    """Measure the level in dB, re the loudest cell, on log-spaced rows from 20 Hz to 20 kHz.

    Each row is demodulated at its own frequency and smoothed by a Hann window eight periods or 6 ms long,
    whichever is longer.
    """
    pad = 16384
    n = round(seconds * RATE)
    y = np.zeros(n + 2 * pad)
    y[pad : pad + min(n, x.size)] = x[:n]
    centers = pad + np.linspace(0, n, columns).astype(int)
    phase = 2j * np.pi * np.arange(y.size) / RATE
    level = np.empty((rows, columns))
    for i, f in enumerate(np.geomspace(20.0, 20000.0, rows)):
        window = np.hanning(int(min(2 * pad - 1, round(max(8 / f, 0.006) * RATE))) | 1)
        level[i] = np.abs(fftconvolve(y * np.exp(-phase * f), window, mode="same")[centers]) * 2 / window.sum()
    return 20 * np.log10(np.maximum(level, 1e-12) / np.max(level))


def figure_spectrum(k: Kick) -> str:
    """Draw the spectrogram of a preset, over its whole length."""
    end, step = nice(k.p.length_ms)
    floor, top = 90, 96
    level = spectrogram(k.audio, end / KILO, rows=240, columns=500)
    index = np.round(np.clip(1 + level / floor, 0, 1) * 255).astype(np.uint8)[::-1]
    panel = Panel((64, top, 760, 300), Axis(0, end), Axis(20, 20000, log=True))
    bar = Panel((852, top + 40, 16, 220), Axis(0, 1), Axis(-floor, 0))
    scale = np.arange(255, -1, -1, dtype=np.uint8)[:, None]
    ramps = {"light": ramp([LIGHT["paper"], *BLUES]), "dark": ramp([DARK["paper"], *BLUES[::-1]])}
    body = []
    for theme, palette in ramps.items():  # quiet recedes into the paper; in the dark theme loud is light
        for image, area in ((index, panel), (scale, bar)):
            body.append(
                f'<image class="{theme}" href="{png(image, palette)}" x="{area.x}" y="{area.y}" width="{area.w}" '
                f'height="{area.h}" preserveAspectRatio="none"/>',
            )
    ticks = (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
    body.append(panel.frame(Ticks(steps(end, step), num, "time, ms"), Ticks((), hz, "frequency, Hz"), grid=False))
    body += [f'<text x="{panel.x - 8}" y="{panel.sy(f) + 4:.1f}" text-anchor="end">{hz(f)}</text>' for f in ticks]
    body.append(f'<text class="label" x="{bar.x}" y="{bar.y - 14}">level, dB</text>')
    body += [f'<text x="{bar.x + 24}" y="{bar.sy(v) + 4:.1f}">{num(v)}</text>' for v in (0, -30, -60, -90)]
    subtitle = "Level over time and frequency, to 90 dB below the peak"
    description = f"Spectrogram of the {k.name} preset over its whole length"
    return svg((960, top + 352), f"Spectrum of {k.name}", subtitle, "".join(body), description)


def sine_part(p: patch.Patch, t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the body at times t in ms without its level and amplitude shape: sin φ, for a patch with h2 = 0."""
    v = Voice(p, RATE)
    return v.body(t / KILO) / (v.body_gain * v.body_amp(t / KILO))


def figure_body(long: patch.Patch) -> str:
    """Draw how the second harmonic shapes the body, and how a tail phase meets the free-running sine."""
    shape = Panel(LEFT, Axis(0, 360), Axis(-1.4, 1.4))
    phi = np.linspace(0, 2 * math.pi, 721)
    body = [shape.frame(Ticks((0, 90, 180, 270, 360), num, "phase φ, degrees"), Ticks((-1, 0, 1), num, "One period"))]
    for cls, h2 in (("ref", 0.0), ("o2", 0.25), ("o4", 0.5)):
        body.append(shape.line(cls, np.degrees(phi), (np.sin(phi) + h2 * np.sin(2 * phi)) / math.sqrt(1 + h2 * h2)))
    body.append(legend(below(shape), [("kref", "h2 = 0: a sine"), ("ko2", "h2 = 0.25"), ("ko4", "h2 = 0.5, the most")]))
    tail = patch.edit(long, "body.h2", 0.0)
    knot = tail.body.pitch[-1][0]
    span = Panel(RIGHT, Axis(knot - 40, knot + 80), Axis(-1.4, 1.4))
    t = np.linspace(span.xs.lo, span.xs.hi, 1201)
    v = Voice(tail, RATE)
    ticks = Ticks(tuple(range(round(knot) - 40, round(knot) + 81, 30)), num, "ms")
    body.append(span.frame(ticks, Ticks((-1, 0, 1), num, "The long preset's body, around its last pitch knot")))
    body.append(span.line("halo", t, np.sin(v.phi0 + 2 * math.pi * v.f0 * t / KILO)))
    body.append(span.line("s2", t, sine_part(patch.edit(tail, "body.phase_ref", "onset"), t)))
    body.append(span.line("s1", t, sine_part(tail, t)))
    body.append(span.vline(knot) + span.text((knot + 2, 1.22), "last pitch knot"))
    keys = [
        ("khalo", "the free-running sine at tune_hz"),
        ("key1", 'phase_ref = "tail"'),
        ("key2", 'phase_ref = "onset"'),
    ]
    body.append(legend(below(span), keys))
    subtitle = "Its second harmonic keeps every zero crossing; a tail phase is exactly the free-running sine"
    description = "The body's shape for three values of h2, and the long preset with each phase reference"
    return svg((960, 96 + 200 + 128), "The body", subtitle, "".join(body), description)


def versus(panel: Panel, knots: patch.Knots, curve: Curve, spline: CubicSpline, ticks: tuple[Ticks, Ticks]) -> str:
    """Draw a curve and a cubic spline through the same knots, the spline only between knots and within the panel."""
    kx, ky = np.array([ms for ms, _ in knots]), np.array([value for _, value in knots])
    keep = kx <= panel.xs.hi
    t = np.linspace(0, panel.xs.hi, 1201)
    span = np.linspace(kx[0], min(kx[-1], panel.xs.hi), 1201)
    theirs = panel.clip(f"clip-{panel.x:.0f}", panel.line("s2", span, spline(span)))
    ours = panel.line("s1", t, curve(t / KILO))
    return panel.frame(*ticks) + theirs + ours + panel.dots("knot", kx[keep], ky[keep])


def figure_curves(p: patch.Patch) -> str:
    """Draw two of a preset's curves next to cubic splines through the same knots."""
    v = Voice(p, RATE)
    pitch = Panel(LEFT, Axis(0, 60), Axis(-2, 8))
    shape = Panel(RIGHT, Axis(0, 120), Axis(-1, 2))
    ms, ratio = zip(*p.body.pitch, strict=True)
    bent = CubicSpline(ms, ratio, bc_type=("not-a-knot", (1, 0.0)))  # the same flat end as the pitch curve
    ms, gain = zip(*p.body.amp, strict=True)
    swung = CubicSpline(ms, gain, bc_type=((1, 0.0), (1, 0.0)))  # the same flat ends as an amplitude shape
    shown = (Ticks((0, 20, 40, 60), num, "ms"), Ticks((-2, 0, 2, 4, 6, 8), num, "Pitch, ratio to tune_hz"))
    body = [versus(pitch, p.body.pitch, v.pitch, bent, shown)]
    shown = (Ticks((0, 30, 60, 90, 120), num, "ms"), Ticks((-1, 0, 1, 2), num, "Body amplitude"))
    body.append(versus(shape, p.body.amp, v.body_amp, swung, shown))
    t = np.linspace(0, p.body.pitch[-1][0], 4001)
    dip, at = float(np.min(bent(t))), float(t[np.argmin(bent(t))])
    body.append(pitch.text((at + 2, dip - 0.6), f"the spline: {num(round(dip, 2))}"))
    t = np.linspace(0, 120, 12001)
    top, bottom = float(np.max(swung(t))), float(np.min(swung(t)))
    body.append(shape.text((float(t[np.argmax(swung(t))]) + 3, 1.8), f"the spline: up to {num(round(top, 1))}"))
    body.append(shape.text((float(t[np.argmin(swung(t))]) + 3, -0.8), f"down to {num(round(bottom, 1))}"))
    keys = [("key1", "PCHIP, as Ballast draws it"), ("key2", "a cubic spline through the same knots"), ("dot1", "knot")]
    body.append(legend(below(pitch), keys))
    subtitle = "Through the tight preset's knots, Ballast's curves never overshoot; a spline swings far past them"
    description = "The tight preset's pitch curve and body amplitude, drawn by Ballast and by cubic splines"
    return svg((960, 96 + 200 + 128), "Curves", subtitle, "".join(body), description)


def measured_band(seed: int, band: tuple[float, float]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Measure the power of a seed's noise in sixth-octave bands, in dB re the designed passband."""
    lo, hi = band
    f = np.arange(1, noise.COMPONENTS + 1, dtype=float)
    power = 1.0 / ((1.0 + (lo / f) ** 4) * (1.0 + (f / hi) ** 4))
    amplitude = np.abs(np.fft.rfft(noise.table(seed, lo, hi, INTERNAL)))[1 : noise.COMPONENTS + 1] / (INTERNAL / 2)
    level = amplitude**2 * math.fsum(power)  # |H(f_k)|² ρ_k² / 2, whose expected value is |H(f_k)|²
    index = np.searchsorted(f, np.geomspace(20, 20000, 61))[:-1]
    counts = np.diff(np.append(index, f.size))
    centers = np.sqrt(f[index] * f[index + counts - 1])
    return centers, 10 * np.log10(np.add.reduceat(level, index) / counts)


def figure_attack(p: patch.Patch) -> str:
    """Draw the constant-power weights of the attack's mix, and the band of a preset's noise."""
    mix = Panel(LEFT, Axis(0, 1), Axis(0, 1.05))
    m = np.linspace(0, 1, 201)
    body = [
        mix.frame(Ticks((0, 0.25, 0.5, 0.75, 1), num, "noise_mix"), Ticks(LEVELS, num, "Weights of the mix")),
        mix.line("s1", m, np.cos(math.pi * m / 2)),
        mix.line("s2", m, np.sin(math.pi * m / 2)),
        legend(below(mix), [("key1", "tone, cos(π m / 2)"), ("key2", "noise, sin(π m / 2)")]),
    ]
    lo, hi = p.attack.noise_band_hz
    band = Panel(RIGHT, Axis(20, 20000, log=True), Axis(-60, 6))
    f = np.geomspace(20, 20000, 600)
    designed = 10 * np.log10(1.0 / ((1.0 + (lo / f) ** 4) * (1.0 + (f / hi) ** 4)))
    centers, measured = measured_band(p.seed, p.attack.noise_band_hz)
    body.append(band.frame(Ticks((100, 1000, 10000), hz, "frequency, Hz"), Ticks((-60, -40, -20, 0), num, "Level, dB")))
    body.append(band.hline(-3, "ref") + band.text((25, -8), "−3 dB at both edges"))
    body.append(band.clip("band", band.dots("dot2", centers, measured, 3.5) + band.line("s1", f, designed)))
    keys = [
        ("key1", f"the tight preset's band, {lo / KILO:g} to {hi / KILO:g} kHz"),
        ("dot2", "its seeded noise, measured"),
    ]
    body.append(legend(below(band), keys))
    subtitle = "Tone and noise mix at constant power; the noise fills its band, with half power at the edges"
    description = "The weights of tone and noise against noise_mix, and the tight preset's noise band"
    return svg((960, 96 + 200 + 128), "The attack", subtitle, "".join(body), description)


DRIVES = (("o1", 6.0), ("o2", 12.0), ("o3", 21.0), ("o4", 48.0))


def figure_saturator(p: patch.Patch) -> str:
    """Draw the saturator's transfer curve at four drives, and a full-scale sine through each."""
    transfer = Panel(LEFT, Axis(-2, 2), Axis(-1.2, 1.2))
    cycle = Panel(RIGHT, Axis(0, 360), Axis(-1.2, 1.2))
    x = np.linspace(-2, 2, 801)
    phi = np.linspace(0, 360, 721)
    sine = np.sin(np.radians(phi))
    lines, waves = [transfer.line("ref", x, x)], [cycle.line("ref", phi, sine)]
    for cls, db in DRIVES:
        v = Voice(patch.edit(p, "drive.db", db), RATE)
        lines.append(transfer.line(cls, x, v.saturate(x)))
        waves.append(cycle.line(cls, phi, v.saturate(sine)))
    body = [
        transfer.frame(Ticks((-2, -1, 0, 1, 2), num, "input"), Ticks((-1, 0, 1), num, "Output against input")),
        transfer.clip("transfer", "".join(lines)),
        cycle.frame(Ticks((0, 90, 180, 270, 360), num, "phase, degrees"), Ticks((-1, 0, 1), num, "A full-scale sine")),
        "".join(waves),
    ]
    keys = [
        ("kref", "0 dB: a bypass"),
        ("ko1", "6 dB"),
        ("ko2", "12 dB"),
        ("ko3", "21 dB, as driven"),
        ("ko4", "48 dB"),
    ]
    body.append(legend_row(below(transfer), keys))
    subtitle = "Drive raises the small-signal gain and squares the wave off, while full scale stays at full scale"
    description = "The saturator's transfer curve and a full-scale sine at drives from 0 to 48 dB"
    return svg((960, 96 + 200 + 90), "The saturator", subtitle, "".join(body), description)


CORNERS = (("o1", 10.0), ("o2", 30.0), ("o3", 100.0), ("o4", 1000.0))


def figure_low_cut() -> str:
    """Draw the low cut's magnitude and phase at four corners."""
    magnitude = Panel(LEFT, Axis(1, 20000, log=True), Axis(-60, 6))
    phase = Panel(RIGHT, Axis(1, 20000, log=True), Axis(0, 180))
    f = np.geomspace(1, 20000, 600)
    ticks = Ticks((1, 10, 100, 1000, 10000), hz, "frequency, Hz")
    gains, turns = [], []
    for cls, corner in CORNERS:
        _, h = sosfreqz(render.low_cut(corner, INTERNAL), worN=f, fs=INTERNAL)
        gains.append(magnitude.line(cls, f, 20 * np.log10(np.abs(h))))
        turns.append(phase.line(cls, f, np.degrees(np.angle(h))))
    body = [
        magnitude.frame(ticks, Ticks((-60, -40, -20, 0), num, "Magnitude, dB")),
        magnitude.hline(-3, "ref") + magnitude.text((19000, -9), "−3 dB at the corner", "end"),
        magnitude.clip("magnitude", "".join(gains)),
        phase.frame(ticks, Ticks((0, 45, 90, 135, 180), num, "Phase, degrees")),
        "".join(turns),
    ]
    keys = [("ko1", "low_cut_hz = 10"), ("ko2", "30"), ("ko3", "100"), ("ko4", "1000")]
    body.append(legend_row(below(magnitude), keys))
    subtitle = "A second-order Butterworth high-pass: half power at the corner, 12 dB per octave below it"
    description = "Magnitude and phase of the low cut at corners of 10, 30, 100 and 1000 Hz"
    return svg((960, 96 + 200 + 90), "The low cut", subtitle, "".join(body), description)


def response(base: int, n: int = 1 << 20) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute a decimator's magnitude response in dB, at n / 2 + 1 frequencies up to half the internal rate."""
    taps = render.decimator(base)
    f = np.fft.rfftfreq(n, 1 / (base * render.OVERSAMPLING))
    return f, 20 * np.log10(np.maximum(np.abs(np.fft.rfft(taps, n)), 1e-300))


def figure_decimator() -> str:
    """Draw both decimators' magnitude responses, with the passband in detail."""
    top = 96
    wide = Panel((64, top, 560, 260), Axis(1000, 384000, log=True), Axis(-170, 10))
    near = Panel((712, top, 220, 260), Axis(0, 20000), Axis(-0.00003, 0.00003))
    body = [wide.frame(Ticks((1000, 10000, 100000), hz, "frequency, Hz"), Ticks((-150, -110, -50, 0), num, "gain, dB"))]
    band = wide.sx(PASSBAND_HZ) - wide.x
    body.append(f'<rect class="band" x="{wide.x}" y="{wide.y}" width="{band:.1f}" height="{wide.h}"/>')
    body.append(f'<path class="limit" d="M{wide.sx(22050):.1f},{wide.sy(-110):.1f}H{wide.x + wide.w}"/>')
    body.append(
        f'<text class="label" x="{wide.x + wide.w - 4}" y="{wide.sy(-110) - 8:.1f}" text-anchor="end">'
        "specified: ≥ 110 dB down</text>",
    )
    ticks = (-0.00002, 0, 0.00002)
    body.append(near.frame(Ticks((0, 10000, 20000), hz, "frequency, Hz"), Ticks(ticks, fixed5, "passband detail, dB")))
    series = (("48 kHz family", 48000, "s1", "key1"), ("44.1 kHz family", 44100, "s2", "key2"))
    for name, base, cls, _ in series:
        f, gain = response(base)
        edges = np.geomspace(1000, base * render.OVERSAMPLING / 2, 561)
        envelope = np.maximum.reduceat(gain, np.searchsorted(f, edges[:-1]))  # the worst case in each column
        body.append(f'<path class="{cls}" d="{wide.path(np.sqrt(edges[:-1] * edges[1:]), envelope)}"/>')
        passband = f <= PASSBAND_HZ
        body.append(f'<path class="{cls}" d="{near.path(f[passband][::40], gain[passband][::40])}"/>')
        ripple, stopband = np.max(np.abs(gain[passband])), -np.max(gain[f >= base / 2])
        taps = render.decimator(base).size
        sys.stdout.write(f"decimator {name}: {taps} taps, ±{ripple:.5f} dB to 20 kHz, ≥ {stopband:.1f} dB\n")
    body.append(legend((wide.x + 16, float(wide.sy(-122))), [(key, name) for name, _, _, key in series]))
    subtitle = "Both frozen filters at the internal rate, as their checksummed taps say"
    description = "Magnitude response of the two frozen decimators, with the passband in detail"
    return svg((960, top + 312), "Decimator response", subtitle, "".join(body), description)


def figure_rates(p: patch.Patch) -> str:
    """Draw the 48, 96 and 192 kHz renders of one patch where its click moves fastest, sample by sample."""
    rates = ((48000, "dot1", 9.0), (96000, "dot2", 6.5), (192000, "dot3", 4.0))  # coarsest first, drawn under
    audio = {rate: render.render(patch.edit(p, "output.sample_rate", rate))[0] for rate, _, _ in rates}
    center = int(np.argmax(np.abs(np.diff(audio[RATE][: round(0.003 * RATE)]))))  # the click's steepest step
    lo, hi = (center - 3.6) * KILO / RATE, (center + 4.6) * KILO / RATE
    shown = {rate: np.arange(math.ceil(lo * rate / KILO), math.floor(hi * rate / KILO) + 1) for rate, _, _ in rates}
    values = audio[192000][shown[192000]]
    pad = 0.12 * float(np.ptp(values))
    panel = Panel((64, 96, 864, 260), Axis(lo, hi), Axis(float(values.min()) - pad, float(values.max()) + pad))
    shown_as = Ticks(inner(panel.ys), num, "Samples of the tight preset's click, where it moves fastest")
    body = [panel.frame(Ticks(inner(panel.xs), num, "time, ms"), shown_as)]
    body += [panel.dots(cls, shown[rate] * KILO / rate, audio[rate][shown[rate]], r) for rate, cls, r in rates]
    rising = audio[RATE][center + 1] > audio[RATE][center]
    lx, anchor = (panel.x + 24, "start") if rising else (panel.x + panel.w - 24, "end")
    xs, ys = panel.sx(shown[RATE] * KILO / RATE), panel.sy(audio[RATE][shown[RATE]])
    nearest = int(np.argmin(np.hypot(xs - lx, ys - panel.y - 30)))  # the 48 kHz sample closest to the label
    body.append(f'<path class="ref" d="M{lx:.1f},{panel.y + 30}L{xs[nearest]:.1f},{ys[nearest] - 12:.1f}"/>')
    label = "one sample, at every rate"
    body.append(f'<text class="label" x="{lx:.1f}" y="{panel.y + 22}" text-anchor="{anchor}">{label}</text>')
    keys = [("dot1", "48 kHz"), ("dot2", "96 kHz: the 48 kHz samples, and one between each pair"), ("dot3", "192 kHz")]
    body.append(legend(below(panel), keys))
    subtitle = "Each higher rate keeps every sample of the lower one, bit for bit, and fills in between"
    description = (
        "The tight preset's click at 48, 96 and 192 kHz, where every lower-rate sample is also a higher-rate one"
    )
    return svg((960, 96 + 260 + 128), "Rates", subtitle, "".join(body), description)


def column(x0: float, x1: float, y: float, base: float) -> str:
    """Draw a column from a baseline, with its data end rounded."""
    r = min(3.0, (x1 - x0) / 2, base - y)
    left = f"M{x0:.1f},{base:.1f}V{y + r:.1f}Q{x0:.1f},{y:.1f} {x0 + r:.1f},{y:.1f}"
    right = f"H{x1 - r:.1f}Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f}V{base:.1f}Z"
    return f'<path class="bar" d="{left}{right}"/>'


def figure_output(p: patch.Patch) -> str:
    """Draw a true peak between samples, above the highest sample, and the dither's triangular density."""
    hot = patch.edit(
        patch.edit(patch.edit(p, "attack.tone_hz", 15000.0), "attack.noise_mix", 0.0),
        "attack.level_db",
        0.0,
    )
    audio, report = render.render(hot)
    fine = resample_poly(np.concatenate((audio.astype(float), np.zeros(32))), 4, 1)  # as the report estimates it
    at = int(np.argmax(np.abs(fine))) / 4
    sign = math.copysign(1.0, fine[round(at * 4)])
    lo, hi = math.floor(at) - 4, math.ceil(at) + 4
    zoom = Panel(LEFT, Axis(lo * KILO / RATE, hi * KILO / RATE), Axis(-1, 1))
    n = np.arange(lo, hi + 1)
    m = np.arange(lo * 4, hi * 4 + 1)
    body = [
        zoom.frame(Ticks(inner(zoom.xs), num, "ms"), Ticks((-1, -0.5, 0, 0.5, 1), num, "A 15 kHz click on tight")),
        zoom.hline(sign * report.true_peak, "s2") + zoom.hline(sign * report.peak, "s1"),
        zoom.line("ref", m * KILO / (4 * RATE), fine[m]),
        zoom.dots("dot1", n * KILO / RATE, audio[n]),
    ]
    keys = [
        ("key2", f"true peak, {num(round(render.db(report.true_peak), 2))} dBTP"),
        ("key1", f"highest sample, {num(round(render.db(report.peak), 2))} dBFS"),
    ]
    body.append(legend(below(zoom), [*keys, ("kref", "the waveform between samples, 4× oversampled")]))
    d = noise.dither(p.seed, 200_000)
    counts, edges = np.histogram(d, bins=40, range=(-1, 1))
    density = counts / (d.size * (edges[1] - edges[0]))
    tpdf = Panel(RIGHT, Axis(-1, 1), Axis(0, 1.1))
    body.append(
        tpdf.frame(Ticks((-1, -0.5, 0, 0.5, 1), num, "LSB"), Ticks(LEVELS, num, "Density of 200 000 dither values")),
    )
    body += [
        column(float(tpdf.sx(a)) + 1, float(tpdf.sx(b)) - 1, float(tpdf.sy(h)), float(tpdf.sy(0)))
        for a, b, h in zip(edges[:-1], edges[1:], density, strict=True)
    ]
    body.append(tpdf.line("ref", (-1, 0, 1), (0, 1, 0)))
    body.append(legend(below(tpdf), [("key1", "the seeded dither"), ("kref", "the triangle it follows")]))
    subtitle = "A waveform can peak between its samples; PCM adds triangular dither of ±1 LSB"
    description = "A true peak above the highest sample, and the triangular density of the dither"
    return svg((960, 96 + 200 + 128), "The output stage", subtitle, "".join(body), description)


def figure_audition(k: Kick) -> str:
    """Draw hits on the beat that play to their ends, a bed that restarts on every bar, and the hits' sum."""
    bpm, beats, left, width = 128.0, 8, 128, 800
    length = k.audio.size / RATE * bpm / 60  # the kick's length in beats
    mix = audition.Hits(k.audio, RATE, bpm).fill(round(beats * 60 / bpm * RATE))[:, 0]
    grid = Panel((left, 96, width, 304), Axis(0, beats), Axis(0, 1))
    body = [grid.frame(Ticks(tuple(range(beats + 1)), num, f"beats at {num(bpm)} bpm"), Ticks(()))]
    body += [grid.vline(b, "axis") for b in (0, 4, 8)]
    for n in range(beats):
        y = 104 + 14 * n
        body.append(column(float(grid.sx(n)) + 1, float(grid.sx(min(n + length, beats))), y, y + 10))
    body.append(f'<text class="label" x="28" y="{104 + 14 * 4}">hits</text>')
    for loop in range(2):
        x0, x1 = grid.sx(4 * loop) + 1, grid.sx(4 * loop + 4) - 1
        body.append(f'<rect class="key2" x="{x0:.1f}" y="240" width="{x1 - x0:.1f}" height="10" rx="3"/>')
        body.append(f'<text class="label" x="{x0 + 4:.1f}" y="232">loop {loop + 1}</text>')
    body.append('<text class="label" x="28" y="249">bed</text>')
    out = Panel((left, 268, width, 120), Axis(0, beats), Axis(-1, 1))
    columns = round(out.w)
    lo, hi, rms = envelopes(mix, columns, mix.size)
    t = (np.arange(columns) + 0.5) * beats / columns
    body.append(f'<path class="peak" d="{out.path(np.r_[t, t[::-1]], np.r_[hi, lo[::-1]], close=True)}"/>')
    body.append(f'<path class="rms" d="{out.path(np.r_[t, t[::-1]], np.r_[rms, -rms[::-1]], close=True)}"/>')
    body.append('<text class="label" x="28" y="332">hits, summed</text>')
    subtitle = (
        f"The {k.name} preset on every beat: each hit plays to its end and the tails sum, while a bed loops on the bar"
    )
    description = "A timeline of audition: overlapping hits on the beat, their sum, and a bed looping on the bar"
    return svg((960, 96 + 304 + 60), "Audition", subtitle, "".join(body), description)


def figure_design(moves: Sequence[tuple[str, Kick]]) -> str:
    """Draw the tight preset and each design move: the first 60 ms, and the whole kick."""
    top, row, gap = 96, 56, 26
    length, length_step = nice(max(k.p.length_ms for _, k in moves))
    body = [
        '<text class="label" x="300" y="84">first 60 ms</text>',
        '<text class="label" x="630" y="84">the whole kick</text>',
    ]
    for i, (how, k) in enumerate(moves):
        y = top + i * (row + gap)
        start = Panel((300, y, 290, row), Axis(0, 60), Axis(-1, 1))
        whole = Panel((630, y, 298, row), Axis(0, length), Axis(-1, 1))
        body += [
            start.hline(0),
            whole.hline(0),
            start.line("trace", *trace(k.audio, 60, 290)),
            waveform(whole, k.audio, length / KILO),
        ]
        body.append(f'<text class="name" x="28" y="{y + 22}">{k.name}</text>')
        body.append(f'<text class="math" x="28" y="{y + 42}">{html.escape(how)}</text>')
    end = top + len(moves) * (row + gap) - gap
    start = Panel((300, end, 290, 0), Axis(0, 60), Axis(0, 1))
    whole = Panel((630, end, 298, 0), Axis(0, length), Axis(0, 1))
    body.append(
        start.frame(Ticks((0, 20, 40, 60), num, "ms"), Ticks(()))
        + whole.frame(Ticks(steps(length, length_step), num, "ms"), Ticks(())),
    )
    subtitle = "Each row is one move made on the tight preset, as the guide describes it"
    description = "Waveforms of the tight preset and of six design moves made on it"
    return svg((960, end + 56), "Designing a kick", subtitle, "".join(body), description)


CHARACTER = {
    "tight": (
        "Short and punchy, for a busy mix: a fast pitch drop, a bright click of tone and noise, and a body that "
        "is gone in a quarter of a second."
    ),
    "driven": (
        "Dense and saturated: heavy drive squares the body off and adds odd harmonics, a strong second harmonic "
        "adds even ones, and the output sits lower to leave room for them."
    ),
    "round": "Warm and rounded: a gentle pitch drop, a soft, low click, and a body that rings for longer.",
    "long": (
        "An 808-style kick with a long tail, to play as a bassline with `ballast multisample`. Its tail is locked "
        'to the note with `phase_ref = "tail"`, so editing the sweep never moves it.'
    ),
}


def preset_row(k: Kick) -> str:
    """Describe a preset in one row of the gallery's table."""
    b, r = k.p.body, k.report
    drop = f"{num(b.pitch[0][1])}× to the note in {num(b.pitch[-1][0])} ms"
    peak = num(round(render.db(r.peak), 2))
    return (
        f"| [`{k.name}`](#{k.name}) | {num(b.tune_hz)} Hz, {note(b.tune_hz)} | {num(k.p.length_ms)} ms | {drop} "
        f"| {num(k.p.drive.db)} dB | {peak} dBFS | [{k.name}.wav](audio/{k.name}.wav) |"
    )


def preset_section(k: Kick) -> str:
    """Show one preset in full: what it is, its sound, its figures and its patch."""
    sweep, _ = nice(1.5 * k.p.body.pitch[-1][0])
    return "\n".join(
        [
            f"## {k.name}",
            "",
            CHARACTER[k.name],
            "",
            f"[Listen to {k.name}.wav](audio/{k.name}.wav) · start from it with `ballast init {k.name}`",
            "",
            (
                f"![The {k.name} preset's pitch, body and attack curves through their knots, and its first "
                f"{num(sweep)} ms](figures/anatomy-{k.name}.svg)"
            ),
            "",
            f"![Spectrogram of the {k.name} preset over its whole length](figures/spectrum-{k.name}.svg)",
            "",
            "<details>",
            f"<summary>The {k.name} patch</summary>",
            "",
            "```toml",
            k.text.rstrip("\n"),
            "```",
            "",
            "</details>",
        ],
    )


def presets_page(kicks: dict[str, Kick]) -> str:
    """Write the preset gallery: a table of the presets, then each one with its sound, figures and patch."""
    head = [
        "# Presets",
        "",
        "<!-- Generated by docs/build.py from the shipped presets: edit the script, not this page. -->",
        "",
        "Ballast ships four presets: complete patches to start from, tuned with `--tune` to your track.",
        "",
        "```",
        "ballast init round --tune F1 -o kick.toml",
        "```",
        "",
        "| Preset | Tuning | Length | Pitch drop | Drive | Peak | Sound |",
        "|---|---|---|---|---|---|---|",
        *(preset_row(k) for k in kicks.values()),
        "",
        "![Waveforms of the four presets, tight, driven, round and long, rendered at 48 kHz](figures/presets.svg)",
        "",
        (
            "Every preset renders at 48 kHz in `gain` mode with no low cut, and its sound here is 16-bit PCM. Their "
            "figures share every scale but time, so the presets compare at a glance."
        ),
    ]
    return "\n\n".join(["\n".join(head), *(preset_section(k) for k in kicks.values())]) + "\n"


def main() -> None:
    """Build every figure, every example sound and the preset gallery."""
    kicks = render_kicks()
    moves = design_moves(kicks["tight"])
    figures, audio = DOCS / "figures", DOCS / "audio"
    figures.mkdir(exist_ok=True)
    audio.mkdir(exist_ok=True)
    for k in [*kicks.values(), *(k for _, k in moves[1:])]:
        render.export(audio / f"{k.name}.wav", k.p, k.text, k.audio, 16)
        summary = str(k.report).replace("\n", " | ")
        sys.stdout.write(f"{k.name}: {summary}\n")
    tight = kicks["tight"].p
    drawn = {
        "presets": figure_presets(kicks),
        "pipeline": figure_pipeline(),
        "body": figure_body(kicks["long"].p),
        "curves": figure_curves(tight),
        "attack": figure_attack(tight),
        "saturator": figure_saturator(tight),
        "low-cut": figure_low_cut(),
        "decimator": figure_decimator(),
        "rates": figure_rates(tight),
        "output-stage": figure_output(tight),
        "audition": figure_audition(kicks["long"]),
        "design": figure_design(moves),
    }
    drawn |= {f"anatomy-{name}": figure_anatomy(k) for name, k in kicks.items()}
    drawn |= {f"spectrum-{name}": figure_spectrum(k) for name, k in kicks.items()}
    for name, text in drawn.items():
        (figures / f"{name}.svg").write_text(text, encoding="utf-8")
    (DOCS / "presets.md").write_text(presets_page(kicks), encoding="utf-8")


if __name__ == "__main__":
    main()
