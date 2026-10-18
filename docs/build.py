# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regenerate the figures in docs/figures and the example sounds in docs/audio, with ballast itself.

    uv run python docs/build.py

Every figure is drawn from the code in this repository, as SVG that follows the reader's light or
dark theme, and every sound is a preset rendered and exported as `ballast render` would.
"""

from __future__ import annotations

import base64
import html
import math
import struct
import sys
import zlib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.signal import fftconvolve

from ballast_synth import patch, render
from ballast_synth.voice import Voice

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import ArrayLike, NDArray

    from ballast_synth.curve import Curve

DOCS = Path(__file__).resolve().parent
RATE = 48000
PRESETS = ("tight", "driven", "round", "long")  # shortest first
KILO = 1000
PASSBAND_HZ = 20000

# Chart hues and the blue ramp are validated data-visualization defaults; the neutrals lean toward the accent.
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
.fig .s1{{fill:none;stroke:var(--accent);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}
.fig .s2{{fill:none;stroke:var(--series-2);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}
.fig .key1{{fill:var(--accent)}}
.fig .key2{{fill:var(--series-2)}}
.fig .knot{{fill:var(--accent);stroke:var(--paper);stroke-width:2}}
.fig .band{{fill:var(--accent);fill-opacity:.08}}
.fig .limit{{fill:none;stroke:var(--ink-2);stroke-width:1}}
.fig .box{{fill:var(--paper);stroke:var(--rule);stroke-width:1}}
.fig .zone{{fill:var(--panel);stroke:var(--rule);stroke-width:1}}
.fig .flow{{fill:none;stroke:var(--ink-2);stroke-width:1.5}}
.fig .head{{fill:var(--ink-2)}}
.fig .light{{display:var(--show-light)}}
.fig .dark{{display:var(--show-dark)}}
</style>"""


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


@dataclass
class Kick:
    """A preset, and its render at 48 kHz."""

    name: str
    text: str
    p: patch.Patch
    audio: NDArray[np.float32]
    report: render.Report


def render_kicks() -> dict[str, Kick]:
    """Render every preset at 48 kHz, within the ceiling of 16-bit PCM."""
    kicks = {}
    for name in PRESETS:
        text = resources.files("ballast_synth").joinpath("data", "presets", f"{name}.toml").read_text(encoding="utf-8")
        p = patch.loads(text)
        kicks[name] = Kick(name, text, p, *render.render(p, 16))
    return kicks


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
    t = (np.arange(columns) + 0.5) * seconds / columns
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


def figure_anatomy(k: Kick) -> str:
    """Draw a preset's three curves, and the first 60 ms of the sound they make."""
    v, tune, top = Voice(k.p, RATE), k.p.body.tune_hz, 96
    body: list[str] = []

    def draw(panel: Panel, knots: patch.Knots, curve: Curve, ticks: tuple[Ticks, Ticks], scale: float = 1) -> None:
        t = np.linspace(0, panel.xs.hi, 600)
        body.append(panel.frame(*ticks))
        body.append(f'<path class="s1" d="{panel.path(t, scale * curve(t / 1000))}"/>')
        kx = np.array([ms for ms, _ in knots])
        ky = scale * np.array([value for _, value in knots])
        keep = kx <= panel.xs.hi
        body.extend(
            f'<circle class="knot" cx="{x:.1f}" cy="{y:.1f}" r="4.5"/>'
            for x, y in zip(panel.sx(kx[keep]), panel.sy(ky[keep]), strict=True)
        )

    pitch = Panel((64, top, 240, 150), Axis(0, 60), Axis(40, 500, log=True))
    pitch_ticks = Ticks((0, 20, 40, 60), num, "ms"), Ticks((50, 100, 200, 500), hz, "Pitch, Hz")
    draw(pitch, k.p.body.pitch, v.pitch, pitch_ticks, tune)
    start, end = tune * k.p.body.pitch[0][1], tune
    x, y = pitch.sx(0) + 10, pitch.sy(start) + 4
    body.append(f'<text class="label" x="{x:.1f}" y="{y:.1f}">{num(start)} Hz</text>')
    body.append(
        f'<text class="label" x="{pitch.x + pitch.w}" y="{pitch.sy(end) - 10:.1f}" text-anchor="end">'
        f"{num(end)} Hz, {note(end)}</text>",
    )
    gain, levels = Axis(0, 1.05), (0, 0.5, 1)
    body_ticks = Ticks((0, 80, 160, 240), num, "ms"), Ticks(levels, num, "Body amplitude")
    draw(Panel((368, top, 240, 150), Axis(0, 240), gain), k.p.body.amp, v.body_amp, body_ticks)
    attack_ticks = Ticks((0, 2, 4, 6), num, "ms"), Ticks(levels, num, "Attack amplitude")
    draw(Panel((672, top, 240, 150), Axis(0, 6), gain), k.p.attack.amp, v.attack_amp, attack_ticks)
    wave_top = top + 214
    n = round(0.06 * RATE)
    t = np.arange(n) / RATE * 1000
    sound = Panel((64, wave_top, 848, 120), Axis(0, 60), Axis(-0.6, 0.6))
    body.append(
        sound.frame(
            Ticks(range(0, 61, 10), num, "ms"),
            Ticks((-0.5, 0, 0.5), num, "Output, first 60 ms, full scale ±1"),
        ),
    )
    body.append(f'<path class="trace" d="{sound.path(t, k.audio[:n])}"/>')
    subtitle = "Three curves through the patch's knots, and the sound they make"
    description = f"The {k.name} preset's pitch, body and attack curves, and its first 60 ms"
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
    """Draw the spectrogram of a preset's first 250 ms."""
    seconds, floor, top = 0.25, 90, 96
    level = spectrogram(k.audio, seconds, rows=240, columns=500)
    index = np.round(np.clip(1 + level / floor, 0, 1) * 255).astype(np.uint8)[::-1]
    panel = Panel((64, top, 760, 300), Axis(0, 1000 * seconds), Axis(20, 20000, log=True))
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
    body.append(
        panel.frame(Ticks((0, 50, 100, 150, 200, 250), num, "time, ms"), Ticks((), hz, "frequency, Hz"), grid=False),
    )
    body += [f'<text x="{panel.x - 8}" y="{panel.sy(f) + 4:.1f}" text-anchor="end">{hz(f)}</text>' for f in ticks]
    body.append(f'<text class="label" x="{bar.x}" y="{bar.y - 14}">level, dB</text>')
    body += [f'<text x="{bar.x + 24}" y="{bar.sy(v) + 4:.1f}">{num(v)}</text>' for v in (0, -30, -60, -90)]
    subtitle = "Level over time and frequency, to 90 dB below the peak"
    description = f"Spectrogram of the {k.name} preset over its first 250 ms"
    return svg((960, top + 352), f"Spectrum of {k.name}", subtitle, "".join(body), description)


def response(base: int, n: int = 1 << 20) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute a decimator's magnitude response in dB, at n / 2 + 1 frequencies up to half the internal rate."""
    taps = render.decimator(base)
    f = np.fft.rfftfreq(n, 1 / (base * render.K))
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
    for i, (name, base, cls, key) in enumerate(series):
        f, gain = response(base)
        edges = np.geomspace(1000, base * render.K / 2, 561)
        envelope = np.maximum.reduceat(gain, np.searchsorted(f, edges[:-1]))  # the worst case in each column
        body.append(f'<path class="{cls}" d="{wide.path(np.sqrt(edges[:-1] * edges[1:]), envelope)}"/>')
        passband = f <= PASSBAND_HZ
        body.append(f'<path class="{cls}" d="{near.path(f[passband][::40], gain[passband][::40])}"/>')
        y = wide.sy(-122) + 22 * i  # the legend, in the plot's empty corner
        body.append(f'<rect class="{key}" x="{wide.x + 16}" y="{y - 5:.1f}" width="18" height="3" rx="1.5"/>')
        body.append(f'<text class="label" x="{wide.x + 42}" y="{y:.1f}">{name}</text>')
        ripple, stopband = np.max(np.abs(gain[passband])), -np.max(gain[f >= base / 2])
        taps = render.decimator(base).size
        sys.stdout.write(f"decimator {base}: {taps} taps, ±{ripple:.5f} dB to 20 kHz, ≥ {stopband:.1f} dB\n")
    subtitle = "Both frozen filters at the internal rate, as their checksummed taps say"
    description = "Magnitude response of the two frozen decimators, with the passband in detail"
    return svg((960, top + 312), "Decimator response", subtitle, "".join(body), description)


def main() -> None:
    """Build every figure and example sound."""
    kicks = render_kicks()
    figures, audio = DOCS / "figures", DOCS / "audio"
    figures.mkdir(exist_ok=True)
    audio.mkdir(exist_ok=True)
    for k in kicks.values():
        render.export(audio / f"{k.name}.wav", k.p, k.text, k.audio, 16)
        summary = str(k.report).replace("\n", " | ")
        sys.stdout.write(f"{k.name}: {summary}\n")
    (figures / "presets.svg").write_text(figure_presets(kicks), encoding="utf-8")
    (figures / "pipeline.svg").write_text(figure_pipeline(), encoding="utf-8")
    (figures / "anatomy.svg").write_text(figure_anatomy(kicks["tight"]), encoding="utf-8")
    (figures / "spectrum.svg").write_text(figure_spectrum(kicks["tight"]), encoding="utf-8")
    (figures / "decimator.svg").write_text(figure_decimator(), encoding="utf-8")


if __name__ == "__main__":
    main()
