"""Build the showcase in docs/: the figures, the audio and index.html, rendered by ballast itself.

    uv run python docs/build.py

Every sound, chart and number comes from the code in this repository at build time. The tests
run first, and the page and the figures load nothing from the network.
"""

from __future__ import annotations

import base64
import html
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass
from datetime import date
from importlib import resources
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import fftconvolve

from ballast import __version__, patch, render, wav
from ballast.audition import Hits, beat
from ballast.render import db
from ballast.voice import Voice

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
RATE = 48000
BPM = 128.0
LOOP_BEATS = 8  # two bars of quarter notes
PRESETS = {  # shortest first, each with what it is for
    "tight": "Short and punchy, for dense four-on-the-floor.",
    "driven": "Saturated and forward, with a strong second harmonic.",
    "round": "Warm and full, with a slower sweep.",
    "long": "An 808-style tail that holds its note.",
}

# The palette, for the page and for the standalone figures alike. Chart hues and the blue ramp are
# the validated data-visualization defaults; the neutrals lean toward the accent.
LIGHT = {
    "paper": "#fbfcfd",
    "panel": "#f1f4f7",
    "ink": "#0e1116",
    "ink-2": "#4a5361",
    "muted": "#626c7a",
    "rule": "#dde3ea",
    "grid": "#e9edf2",
    "accent": "#2a78d6",
    "accent-ink": "#1c5cab",
    "accent-fill": "#256abf",
    "on-accent": "#ffffff",
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
    "accent-ink": "#79b0f0",
    "accent-fill": "#3987e5",
    "on-accent": "#0f1216",
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
FIGURE_CSS = f"""
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
"""


def css_tokens(scope: str) -> str:
    """The palette as custom properties: light by default, dark by the OS setting or an explicit data-theme."""
    light, dark = ("".join(f"--{k}:{v};" for k, v in t.items()) for t in (LIGHT, DARK))
    if scope != ":root":  # a standalone figure follows the OS setting only
        return f"{scope}{{{light}}}@media (prefers-color-scheme: dark){{{scope}{{{dark}}}}}"
    return (
        f":root{{color-scheme:light;{light}}}"
        f'@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{color-scheme:dark;{dark}}}}}'
        f':root[data-theme="dark"]{{color-scheme:dark;{dark}}}'
    )


def ramp(stops: list[str]) -> np.ndarray:
    """256 colors interpolated through hex stops, as a 256 × 3 palette."""
    rgb = np.array([[int(s[i : i + 2], 16) for i in (1, 3, 5)] for s in stops], dtype=float)
    u = np.linspace(0, len(stops) - 1, 256)
    return np.round([np.interp(u, np.arange(len(stops)), rgb[:, c]) for c in range(3)]).T.astype(np.uint8)


def png(indices: np.ndarray, palette: np.ndarray) -> str:
    """An 8-bit indexed PNG, as a data URI."""

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
    return f"{v:g}".replace("-", "−")


def fixed5(v: float) -> str:
    return "0" if v == 0 else f"{v:+.5f}".replace("-", "−")


def hz(v: float) -> str:
    return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"


def note(tune: float) -> str:
    return patch.describe_tuning(tune).removesuffix(" +0.0 cents")


class Panel:
    """A plot area: data scaled onto the rectangle at (x, y) of size w × h, each axis linear or log."""

    def __init__(self, x, y, w, h, xs, ys, *, xlog=False, ylog=False):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.xs, self.ys, self.xlog, self.ylog = xs, ys, xlog, ylog

    @staticmethod
    def _unit(v, domain, log):
        a, b = domain
        v = np.asarray(v, dtype=float)
        if log:
            return (np.log10(v) - math.log10(a)) / (math.log10(b) - math.log10(a))
        return (v - a) / (b - a)

    def sx(self, v):
        return self.x + self._unit(v, self.xs, self.xlog) * self.w

    def sy(self, v):
        return self.y + (1 - self._unit(v, self.ys, self.ylog)) * self.h

    def path(self, xs, ys, close=False) -> str:
        points = "L".join(f"{x:.1f},{y:.1f}" for x, y in zip(self.sx(xs), self.sy(ys), strict=True))
        return f"M{points}{'Z' if close else ''}"

    def frame(self, xticks, yticks, xfmt=num, yfmt=num, xlabel="", ylabel="", grid=True) -> str:
        """Gridlines, the baseline, tick labels and axis names."""
        out = []
        for v in yticks:
            y = self.sy(v)
            if grid:
                out.append(f'<path class="grid" d="M{self.x},{y:.1f}H{self.x + self.w}"/>')
            out.append(f'<text x="{self.x - 8}" y="{y + 4:.1f}" text-anchor="end">{yfmt(v)}</text>')
        for v in xticks:
            x = self.sx(v)
            if grid:
                out.append(f'<path class="grid" d="M{x:.1f},{self.y}V{self.y + self.h}"/>')
            out.append(f'<text x="{x:.1f}" y="{self.y + self.h + 18}" text-anchor="middle">{xfmt(v)}</text>')
        out.append(f'<path class="axis" d="M{self.x},{self.y + self.h}H{self.x + self.w}"/>')
        if xlabel:
            x, y = self.x + self.w, self.y + self.h + 36
            out.append(f'<text class="label" x="{x}" y="{y}" text-anchor="end">{xlabel}</text>')
        if ylabel:
            out.append(f'<text class="label" x="{self.x}" y="{self.y - 10}">{ylabel}</text>')
        return "".join(out)

    def hover(self, xfmt, yfmt, series) -> dict:
        """What the page's hover layer needs: the geometry, the formats and each series' points."""
        return {
            **{k: getattr(self, k) for k in ("x", "y", "w", "h", "xs", "ys", "xlog", "ylog")},
            "xfmt": xfmt,
            "yfmt": yfmt,
            "series": [{"name": n, "x": np.round(x, 6).tolist(), "y": np.round(y, 6).tolist()} for n, x, y in series],
        }


def svg(name: str, width: int, height: int, body: str, title: str, standalone: bool, attrs: str = "") -> str:
    """A figure: standalone, with its own palette, for the README; or inline for the page."""
    head = f'<title id="{name}-title">{html.escape(title)}</title>'
    common = f'class="fig" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{name}-title"'
    if not standalone:
        return f'<svg {common} id="{name}" {attrs}>{head}{body}</svg>'
    style = f"<style>{css_tokens('.fig')}{FIGURE_CSS}</style>"
    background = f'<rect class="bg" width="{width}" height="{height}" rx="10"/>'
    root = f'<svg xmlns="http://www.w3.org/2000/svg" {common} width="{width}" height="{height}">'
    return f"{root}{head}{style}{background}{body}</svg>\n"


def heading(title: str, subtitle: str) -> str:
    return f'<text class="title" x="28" y="36">{title}</text><text class="label" x="28" y="57">{subtitle}</text>'


def nice_step(span: float, count: int = 6) -> float:
    raw = span / count
    magnitude = 10 ** math.floor(math.log10(raw))
    return next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)


# The kicks.


@dataclass
class Kick:
    name: str
    text: str
    p: patch.Patch
    audio: np.ndarray  # float32 at 48 kHz
    report: render.Report
    loop: np.ndarray  # two bars at BPM, mixed by audition's own Hits
    render_ms: float  # the fastest of several renders on this machine

    @property
    def seconds(self) -> float:
        return self.audio.size / RATE


def render_kicks() -> dict[str, Kick]:
    kicks = {}
    for name in PRESETS:
        text = resources.files("ballast").joinpath("data", "presets", f"{name}.toml").read_text(encoding="utf-8")
        p = patch.loads(text)
        audio, report = render.render(p, 16)
        times = []
        for _ in range(5):
            start = time.perf_counter()
            render.render(p, 16)
            times.append(time.perf_counter() - start)
        loop = Hits(audio, RATE, BPM).fill(beat(LOOP_BEATS, RATE, BPM))[:, 0]
        if np.max(np.abs(loop)) > report.ceiling:
            raise SystemExit(f"{name}: the loop clips at 16 bits")
        kicks[name] = Kick(name, text, p, audio, report, loop, 1000 * min(times))
    return kicks


def write_audio(kicks: dict[str, Kick]) -> None:
    (DOCS / "audio").mkdir(exist_ok=True)
    for k in kicks.values():
        render.export(DOCS / "audio" / f"{k.name}.wav", k.p, k.text, k.audio, 16)
        loop = render.quantize(k.loop, 16, k.p.seed)
        meta = {"patch": k.text, "loop": f"{LOOP_BEATS} unchoked hits at {BPM:g} bpm, as ballast audition plays them"}
        wav.write(DOCS / "audio" / f"{k.name}-loop.wav", loop, RATE, 16, wav.root_note(k.p.body.tune_hz), meta)


# Waveforms.


def envelopes(x: np.ndarray, columns: int, n: int):
    """Per-column minimum, maximum and RMS of the first n samples of x, zero past its end."""
    y = np.zeros(max(n, columns))
    y[: min(n, x.size)] = x[:n]
    starts = np.arange(columns) * y.size // columns
    counts = np.diff(np.append(starts, y.size))
    rms = np.sqrt(np.add.reduceat(y * y, starts) / counts)
    return np.minimum.reduceat(y, starts), np.maximum.reduceat(y, starts), rms


def waveform(panel: Panel, x: np.ndarray, seconds: float) -> tuple[str, np.ndarray]:
    """Peak and RMS envelopes of x over its first `seconds`, one column per unit of width; and each column's peak."""
    columns = round(panel.w)
    lo, hi, rms = envelopes(x, columns, round(seconds * RATE))
    t = (np.arange(columns) + 0.5) * seconds / columns
    there_and_back = np.r_[t, t[::-1]]
    peak = panel.path(there_and_back, np.r_[hi, lo[::-1]], close=True)
    body = panel.path(there_and_back, np.r_[rms, -rms[::-1]], close=True)
    return f'<path class="peak" d="{peak}"/><path class="rms" d="{body}"/>', np.maximum(hi, -lo)


def figure_presets(kicks: dict[str, Kick]) -> str:
    width, left, right, top, row, gap, seconds = 960, 214, 28, 84, 62, 24, 1.5
    body = [heading("Four presets, rendered at 48 kHz", "Peak (light) and RMS (dark) per column, on a full-scale axis")]
    for i, k in enumerate(kicks.values()):
        y = top + i * (row + gap)
        panel = Panel(left, y, width - left - right, row, (0, seconds), (-1, 1))
        body.append(f'<path class="grid" d="M{left},{y}H{width - right}M{left},{y + row}H{width - right}"/>')
        body.append(f'<path class="zero" d="M{left},{panel.sy(0):.1f}H{width - right}"/>')
        body.append(waveform(panel, k.audio, seconds)[0])
        body.append(f'<text class="name" x="28" y="{y + 17}">{k.name}</text>')
        tune, length, drive = k.p.body.tune_hz, k.report.source_ms, k.p.drive.db
        body.append(f'<text class="label" x="28" y="{y + 38}">{num(tune)} Hz · {note(tune)}</text>')
        body.append(f'<text x="28" y="{y + 57}">{num(length)} ms · drive {num(drive)} dB</text>')
    end = top + 4 * (row + gap) - gap
    axis = Panel(left, end, width - left - right, 0, (0, seconds), (0, 1))
    for v in np.arange(0, seconds + 1e-9, 0.25):
        body.append(f'<text x="{axis.sx(v):.1f}" y="{end + 20}" text-anchor="middle">{num(1000 * v)}</text>')
    body.append(f'<text class="label" x="{width - right}" y="{end + 40}" text-anchor="end">time, ms</text>')
    return svg("presets", width, end + 60, "".join(body), "Waveforms of the four presets", True)


def hero_display(k: Kick, loop: bool) -> tuple[str, dict]:
    """The sampler's big display for one preset, playing once or looped."""
    width, height = 1000, 236
    x = k.loop if loop else k.audio
    seconds = x.size / RATE
    panel = Panel(64, 16, 920, 180, (0, seconds), (-1, 1))
    body = []
    for level, text in ((1, "0 dBFS"), (0.5, "−6"), (0.25, "−12")):  # a dB ruler on a linear axis, as in an editor
        for sign in (1, -1):
            body.append(f'<path class="grid" d="M{panel.x},{panel.sy(sign * level):.1f}H{panel.x + panel.w}"/>')
        body.append(f'<text x="{panel.x - 10}" y="{panel.sy(level) + 4:.1f}" text-anchor="end">{text}</text>')
    body.append(f'<text x="{panel.x - 10}" y="{panel.sy(0) + 4:.1f}" text-anchor="end">−∞</text>')
    body.append(f'<path class="zero" d="M{panel.x},{panel.sy(0):.1f}H{panel.x + panel.w}"/>')
    if loop:  # a ruler in bars and beats, as in a sequencer
        ticks = [(beat(b, RATE, BPM) / RATE, f"{b // 4 + 1}.{b % 4 + 1}") for b in range(LOOP_BEATS)]
    else:
        step = nice_step(1000 * seconds) / 1000
        ticks = [(v, num(round(1000 * v))) for v in np.arange(0, seconds, step)]
    for v, text in ticks:
        body.append(f'<path class="grid" d="M{panel.sx(v):.1f},{panel.y}V{panel.y + panel.h}"/>')
        body.append(f'<text x="{panel.sx(v):.1f}" y="{panel.y + panel.h + 20}" text-anchor="middle">{text}</text>')
    unit = "bar.beat" if loop else "ms"
    body.append(f'<text x="{panel.x + panel.w}" y="{panel.y + panel.h + 38}" text-anchor="end">{unit}</text>')
    wave, peaks = waveform(panel, x, seconds)
    body.append(wave)
    y0, y1 = panel.y, panel.y + panel.h
    body.append(f'<path class="playhead" d="M{panel.x},{y0}V{y1}" visibility="hidden"/>')
    name = f"display-{k.name}{'-loop' if loop else ''}"
    title = f"{k.name}{', two bars at 128 bpm' if loop else ''}: waveform"
    attrs = f'data-display="{k.name}" data-loop="{str(loop).lower()}" hidden'
    meta = {"x": panel.x, "w": panel.w, "y0": y0, "y1": y1, "seconds": seconds, "peaks": np.round(peaks, 4).tolist()}
    return svg(name, width, height, "".join(body), title, False, attrs), meta


def mini_waveform(k: Kick) -> str:
    panel = Panel(0, 2, 240, 40, (0, 1.5), (-0.6, 0.6))
    return svg(f"mini-{k.name}", 240, 44, waveform(panel, k.audio, 1.5)[0], f"{k.name}: waveform", False)


# The design figures.


def figure_pipeline(standalone: bool) -> str:
    """Functional block diagram: from patch to file."""
    top = 76 if standalone else 8
    cy = top + 98
    body = [heading("Signal flow", "From the patch's curves to a WAV file")] if standalone else []
    body.append(
        '<defs><marker id="pipeline-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path class="head" d="M0,1L10,5L0,9Z"/></marker></defs>'
    )
    body.append(f'<rect class="zone" x="16" y="{top}" width="430" height="196" rx="8"/>')
    body.append(f'<text class="label" x="32" y="{top + 22}">16 × the base rate: 705.6 or 768 kHz</text>')
    body.append(f'<text class="label" x="795" y="{top + 22}" text-anchor="middle">at the file\'s sample rate</text>')

    def box(x, y, w, title, *lines, h=64):
        out = [f'<rect class="box" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>']
        out.append(f'<text class="boxtitle" x="{x + 12}" y="{y + 21}">{title}</text>')
        for i, (cls, text) in enumerate(lines):
            out.append(f'<text class="{cls}" x="{x + 12}" y="{y + 40 + 16 * i}">{text}</text>')
        return "".join(out)

    def arrow(d):
        return f'<path class="flow" d="{d}" marker-end="url(#pipeline-head)"/>'

    body.append(box(32, top + 36, 172, "Body", ("math", "sin φ + h₂ sin 2φ"), ("label", "pitch curve to phase")))
    body.append(box(32, top + 116, 172, "Attack", ("math", "tone + noise"), ("label", "seeded, band-limited")))
    body.append(f'<circle class="box" cx="246" cy="{cy}" r="14"/><text class="boxtitle" x="246" y="{cy + 5}" '
                f'text-anchor="middle">+</text>')  # fmt: skip
    body.append(arrow(f"M204,{top + 68}C222,{top + 68} 236,{cy - 30} 240,{cy - 15}"))
    body.append(arrow(f"M204,{top + 148}C222,{top + 148} 236,{cy + 30} 240,{cy + 15}"))
    body.append(box(282, cy - 32, 148, "Drive", ("math", "tanh(αx) / tanh α"), ("label", "0 dB is exact bypass")))
    body.append(arrow(f"M260,{cy}H280"))
    body.append(box(470, cy - 32, 148, "Decimator", ("math", "↓16 or ↓8"), ("label", "frozen FIR, ≥ 110 dB")))
    body.append(arrow(f"M430,{cy}H468"))
    body.append(box(646, cy - 32, 132, "Output", ("label", "gain or peak target"), ("label", "polarity")))
    body.append(arrow(f"M618,{cy}H644"))
    body.append(box(806, cy - 32, 138, "WAV", ("label", "float32, or PCM"), ("label", "patch text inside")))
    body.append(arrow(f"M778,{cy}H804"))
    caption = "minimum phase, so nothing rings before the kick"
    body.append(f'<text class="label" x="544" y="{cy + 58}" text-anchor="middle">{caption}</text>')
    height = top + 196 + (24 if standalone else 8)
    return svg("pipeline" if standalone else "fig-pipeline", 960, height, "".join(body), "Signal flow", standalone)


def figure_anatomy(k: Kick, standalone: bool) -> tuple[str, list]:
    """The tight preset's three curves, and the first 60 ms of the sound they make."""
    v, tune = Voice(k.p, RATE), k.p.body.tune_hz
    top = 96 if standalone else 30
    subtitle = "Three curves through the patch's knots, and the sound they make"
    body = [heading(f"Anatomy of {k.name}", subtitle)] if standalone else []
    panels, hover = [], []

    def curve_panel(i, knots, curve, ms_end, ys, title, scale, *, ylog=False, yticks=(), yfmt=num, xticks=()):
        panel = Panel(64 + i * 304, top, 240, 150, (0, ms_end), ys, ylog=ylog)
        t = np.linspace(0, ms_end, 600)
        y = scale * curve(t / 1000)
        out = [panel.frame(xticks, yticks, num, yfmt, "ms", title)]
        out.append(f'<path class="s1" d="{panel.path(t, y)}"/>')
        kx = np.array([ms for ms, _ in knots])
        ky = scale * np.array([value for _, value in knots])
        keep = kx <= ms_end
        out += [f'<circle class="knot" cx="{x:.1f}" cy="{y:.1f}" r="4.5"/>' for x, y in
                zip(panel.sx(kx[keep]), panel.sy(ky[keep]), strict=True)]  # fmt: skip
        hover.append(panel.hover("ms", "hz" if ylog else "gain", [(title, t, y)]))
        return out, panel

    out, pitch = curve_panel(
        0, k.p.body.pitch, v.pitch, 60, (40, 500), "Pitch, Hz", tune,
        ylog=True, yticks=(50, 100, 200, 500), yfmt=hz, xticks=(0, 20, 40, 60),
    )  # fmt: skip
    panels += out
    start, end = tune * k.p.body.pitch[0][1], tune
    x, y = pitch.sx(0) + 10, pitch.sy(start) + 4
    panels.append(f'<text class="label" x="{x:.1f}" y="{y:.1f}">{num(start)} Hz</text>')
    panels.append(
        f'<text class="label" x="{pitch.x + pitch.w}" y="{pitch.sy(end) - 10:.1f}" text-anchor="end">'
        f"{num(end)} Hz, {note(end)}</text>"
    )
    out, _ = curve_panel(
        1, k.p.body.amp, v.body_amp, 240, (0, 1.05), "Body amplitude", 1,
        yticks=(0, 0.5, 1), xticks=(0, 80, 160, 240),
    )  # fmt: skip
    panels += out
    out, _ = curve_panel(
        2, k.p.attack.amp, v.attack_amp, 6, (0, 1.05), "Attack amplitude", 1,
        yticks=(0, 0.5, 1), xticks=(0, 2, 4, 6),
    )  # fmt: skip
    panels += out
    wave_top = top + 214
    n = round(0.06 * RATE)
    t = np.arange(n) / RATE * 1000
    sound = Panel(64, wave_top, 848, 120, (0, 60), (-0.6, 0.6))
    panels.append(sound.frame(range(0, 61, 10), (-0.5, 0, 0.5), num, num, "ms", "Output, first 60 ms, full scale ±1"))
    panels.append(f'<path class="trace" d="{sound.path(t, k.audio[:n])}"/>')
    hover.append(sound.hover("ms", "fs", [("Output", t[::3], k.audio[:n:3])]))
    body += panels
    height = wave_top + 120 + 52
    title = f"Anatomy of the {k.name} preset: pitch, body and attack curves, and its first 60 ms"
    return svg("anatomy" if standalone else "fig-anatomy", 960, height, "".join(body), title, standalone), hover


def spectrogram(x: np.ndarray, seconds: float, rows: int, columns: int, fmin: float = 20.0, fmax: float = 20000.0):
    """Level in dB, re the loudest cell, on log-spaced rows: each row demodulated at its own frequency
    and smoothed by a Hann window eight periods or 6 ms long, whichever is longer."""
    pad = 16384
    n = round(seconds * RATE)
    y = np.zeros(n + 2 * pad)
    y[pad : pad + min(n, x.size)] = x[:n]
    centers = pad + np.linspace(0, n, columns).astype(int)
    freqs = np.geomspace(fmin, fmax, rows)
    phase = 2j * np.pi * np.arange(y.size) / RATE
    level = np.empty((rows, columns))
    for i, f in enumerate(freqs):
        window = np.hanning(int(min(2 * pad - 1, round(max(8 / f, 0.006) * RATE))) | 1)
        level[i] = np.abs(fftconvolve(y * np.exp(-phase * f), window, mode="same")[centers]) * 2 / window.sum()
    return freqs, 20 * np.log10(np.maximum(level, 1e-12) / level.max())


def figure_spectrum(k: Kick, standalone: bool) -> tuple[str, dict, list]:
    seconds, floor = 0.25, 90
    freqs, level = spectrogram(k.audio, seconds, rows=240, columns=500)
    index = np.round(np.clip(1 + level / floor, 0, 1) * 255).astype(np.uint8)[::-1]
    top = 96 if standalone else 30
    panel = Panel(64, top, 760, 300, (0, 1000 * seconds), (20, 20000), ylog=True)
    subtitle = "Level over time and frequency, to 90 dB below the peak"
    body = [heading(f"Spectrum of {k.name}", subtitle)] if standalone else []
    ramps = {"light": ramp([LIGHT["paper"], *BLUES]), "dark": ramp([DARK["paper"], *BLUES[::-1]])}
    for theme, palette in ramps.items():  # quiet recedes into the paper; dark mode flips the anchor
        body.append(
            f'<image class="{theme}" href="{png(index, palette)}" x="{panel.x}" y="{panel.y}" width="{panel.w}" '
            f'height="{panel.h}" preserveAspectRatio="none"/>'
        )
    ticks = (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
    body.append(panel.frame((0, 50, 100, 150, 200, 250), (), num, hz, "time, ms", "frequency, Hz", grid=False))
    body += [f'<text x="{panel.x - 8}" y="{panel.sy(f) + 4:.1f}" text-anchor="end">{hz(f)}</text>' for f in ticks]
    bar = Panel(852, top + 40, 16, 220, (0, 1), (-floor, 0))
    scale = np.arange(255, -1, -1, dtype=np.uint8)[:, None]
    for theme, palette in ramps.items():
        body.append(
            f'<image class="{theme}" href="{png(scale, palette)}" x="{bar.x}" y="{bar.y}" width="{bar.w}" '
            f'height="{bar.h}" preserveAspectRatio="none"/>'
        )
    body.append(f'<text class="label" x="{bar.x}" y="{bar.y - 14}">level, dB</text>')
    body += [f'<text x="{bar.x + 24}" y="{bar.sy(v) + 4:.1f}">{num(v)}</text>' for v in (0, -30, -60, -90)]
    cells = np.round(np.clip(1 + level[::2, ::2] / floor, 0, 1) * 255).astype(np.uint8)
    hover = {
        **{attr: getattr(panel, attr) for attr in ("x", "y", "w", "h", "xs", "ys")},
        "rows": cells.shape[0],
        "columns": cells.shape[1],
        "floor": floor,
        "cells": base64.b64encode(cells.tobytes()).decode(),
    }
    track = []  # the strongest frequency at a few moments, for the data table
    for ms in (0, 2, 4, 6, 10, 20, 30, 40, 60, 100, 150, 200):
        c = round(ms / (1000 * seconds) * (level.shape[1] - 1))
        r = int(np.argmax(level[:, c]))
        track.append((ms, freqs[r], level[r, c]))
    height = top + 300 + 52
    title = f"Spectrogram of the {k.name} preset over its first 250 ms"
    name = "spectrum" if standalone else "fig-spectrum"
    return svg(name, 960, height, "".join(body), title, standalone), hover, track


def response(base: int, n: int = 1 << 20):
    taps = render.decimator(base)
    f = np.fft.rfftfreq(n, 1 / (base * render.K))
    return f, 20 * np.log10(np.maximum(np.abs(np.fft.rfft(taps, n)), 1e-300))


def figure_decimator(standalone: bool) -> tuple[str, list, dict]:
    top = 96 if standalone else 30
    wide = Panel(64, top, 560, 260, (1000, 384000), (-170, 10), xlog=True)
    near = Panel(712, top, 220, 260, (0, 20000), (-0.00003, 0.00003))
    subtitle = "Both frozen filters at the internal rate, as their checksummed taps say"
    body = [heading("Decimator response", subtitle)] if standalone else []
    body.append(wide.frame((1000, 10000, 100000), (-150, -110, -50, 0), hz, num, "frequency, Hz", "gain, dB"))
    band = wide.sx(20000) - wide.x
    body.append(f'<rect class="band" x="{wide.x}" y="{wide.y}" width="{band:.1f}" height="{wide.h}"/>')
    body.append(f'<path class="limit" d="M{wide.sx(22050):.1f},{wide.sy(-110):.1f}H{wide.x + wide.w}"/>')
    body.append(
        f'<text class="label" x="{wide.x + wide.w - 4}" y="{wide.sy(-110) - 8:.1f}" text-anchor="end">'
        "specified: ≥ 110 dB down</text>"
    )
    ticks = (-0.00002, 0, 0.00002)
    body.append(near.frame((0, 10000, 20000), ticks, hz, fixed5, "frequency, Hz", "passband detail, dB"))
    stats = {}
    series = (("48 kHz family", 48000, "s1", "key1"), ("44.1 kHz family", 44100, "s2", "key2"))
    wide_series, near_series = [], []
    for name, base, cls, _ in series:
        f, gain = response(base)
        edges = np.geomspace(1000, base * render.K / 2, 561)
        starts = np.searchsorted(f, edges[:-1])
        envelope = np.maximum.reduceat(gain, starts)  # the worst case in each column
        centers = np.sqrt(edges[:-1] * edges[1:])
        body.append(f'<path class="{cls}" d="{wide.path(centers, envelope)}"/>')
        wide_series.append((name, centers, envelope))
        sel = f <= 20000
        fp, gp = f[sel][::40], gain[sel][::40]
        body.append(f'<path class="{cls}" d="{near.path(fp, gp)}"/>')
        near_series.append((name, fp, gp))
        stats[base] = {
            "taps": render.decimator(base).size,
            "ripple": float(np.max(np.abs(gain[sel]))),
            "stopband": float(-np.max(gain[f >= base / 2])),
            "at_20k": float(np.interp(20000, f, gain)),
            "at_half": float(np.interp(base / 2, f, gain)),
        }
    hover = [wide.hover("hz", "db", wide_series), near.hover("hz", "db5", near_series)]
    for i, (name, _, _, key) in enumerate(series):  # the legend, in the plot's empty corner
        y = wide.sy(-122) + 22 * i
        body.append(f'<rect class="{key}" x="{wide.x + 16}" y="{y - 5:.1f}" width="18" height="3" rx="1.5"/>')
        body.append(f'<text class="label" x="{wide.x + 42}" y="{y:.1f}">{name}</text>')
    height = top + 260 + 52
    title = "Magnitude response of the two frozen decimators, with the passband in detail"
    name = "decimator" if standalone else "fig-decimator"
    return svg(name, 960, height, "".join(body), title, standalone), hover, stats


# The page.


def run_tests() -> tuple[int, float]:
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True
    )
    if done.returncode:
        raise SystemExit(f"the tests fail, so the page is not built:\n{done.stdout}{done.stderr}")
    found = re.search(r"(\d+) passed in ([\d.]+)s", done.stdout)
    return int(found[1]), float(found[2])


def transcript() -> str:
    """A real session: render a preset, then get its patch back out of the WAV."""
    with tempfile.TemporaryDirectory() as tmp:

        def run(*argv):
            done = subprocess.run([sys.executable, "-m", "ballast", *argv], cwd=tmp, capture_output=True, text=True)
            return done.stdout + done.stderr

        lines = ["$ ballast init tight -o tight.toml", run("init", "tight", "-o", "tight.toml").rstrip()]
        lines += ["$ ballast render tight.toml -o tight.wav --format pcm16"]
        lines += [run("render", "tight.toml", "-o", "tight.wav", "--format", "pcm16").rstrip()]
        back = run("extract", "tight.wav")
        same = back == (Path(tmp) / "tight.toml").read_text(encoding="utf-8")
        lines += ["$ ballast extract tight.wav | diff - tight.toml && echo identical", "identical" if same else back]
    out = []
    for line in filter(None, lines):
        if line.startswith("$ "):
            out.append(f'<span class="prompt">$</span> {html.escape(line[2:])}')
        else:
            out.append(f'<span class="out">{html.escape(line)}</span>')
    return "\n".join(out)


def toml_html(text: str) -> str:
    out = []
    for line in text.splitlines():
        code, hash_, comment = line.partition("#")
        code = html.escape(code)
        code = re.sub(r"^(\[\w+\])", r'<span class="h">\1</span>', code)
        code = re.sub(r"(&quot;\w+&quot;)", r'<span class="s">\1</span>', code)
        out.append(code + (f'<span class="c">{hash_}{html.escape(comment)}</span>' if hash_ else ""))
    return "\n".join(out)


def table(head: list[str], rows: list[list], numeric: set[int] = frozenset()) -> str:
    def cell(tag, i, v):
        return f"<{tag}{' class=num' if i in numeric else ''}>{v}</{tag}>"

    header = "".join(cell("th", i, h) for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(cell("td", i, v) for i, v in enumerate(row)) + "</tr>" for row in rows)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def build_page(kicks, anatomy, spectrum, decimator, pipeline, tests) -> str:
    displays, hero = [], {}
    for k in kicks.values():
        for loop in (False, True):
            markup, meta = hero_display(k, loop)
            displays.append(markup)
            hero[f"{k.name}{'-loop' if loop else ''}"] = meta
    cards = []
    for k in kicks.values():
        cards.append(
            f'<button class="preset" type="button" data-preset="{k.name}" aria-pressed="false" id="preset-{k.name}">'
            f'<span class="preset-head"><span class="preset-name">{k.name}</span>'
            f'<span class="preset-note">{note(k.p.body.tune_hz)}</span></span>'
            f'<span class="preset-mini">{mini_waveform(k)}</span>'
            f'<span class="preset-line">{PRESETS[k.name]}</span>'
            f'<span class="preset-specs">{num(k.report.source_ms)}&nbsp;ms · drive {num(k.p.drive.db)}&nbsp;dB · '
            f"peak {num(round(db(k.report.peak), 1))}&nbsp;dBFS</span></button>"
        )
    audio = "".join(
        f'<audio id="audio-{k.name}{s}" src="audio/{k.name}{s}.wav" preload="auto"></audio>'
        for k in kicks.values()
        for s in ("", "-loop")
    )
    anatomy_svg, anatomy_hover = anatomy
    spectrum_svg, spectrum_hover, track = spectrum
    decimator_svg, decimator_hover, stats = decimator
    tight = kicks["tight"]
    n_tests, test_s = tests
    stopband = min(s["stopband"] for s in stats.values())
    ripple = max(s["ripple"] for s in stats.values())
    longest = max(kicks.values(), key=lambda k: k.seconds)
    machine = f"{platform.system()} {platform.machine()}, {os.cpu_count()} threads, Python {platform.python_version()}"
    data = {
        "hero": hero,
        "charts": {"fig-anatomy": anatomy_hover, "fig-decimator": decimator_hover},
        "spectrum": spectrum_hover,
    }
    preset_rows = [
        [
            f"<code>{k.name}</code>",
            f"{num(k.p.body.tune_hz)} Hz, {note(k.p.body.tune_hz)}",
            f"{k.report.source_ms:.0f}",
            f"{db(k.report.peak):.2f}",
            f"{db(k.report.true_peak):.2f}",
            num(k.p.drive.db),
            f"{k.render_ms:.0f}",
        ]
        for k in kicks.values()
    ]
    knot_rows = [
        [label, num(ms), num(value * scale)]
        for label, knots, scale in (
            ("pitch, Hz", tight.p.body.pitch, tight.p.body.tune_hz),
            ("body amplitude", tight.p.body.amp, 1),
            ("attack amplitude", tight.p.attack.amp, 1),
        )
        for ms, value in knots
    ]
    s48, s44 = stats[48000], stats[44100]
    decimator_rows = [
        ["Taps", s48["taps"], s44["taps"]],
        ["Largest passband deviation, 0 to 20 kHz", f"±{s48['ripple']:.5f} dB", f"±{s44['ripple']:.5f} dB"],
        ["Gain at 20 kHz", f"{s48['at_20k']:.5f} dB", f"{s44['at_20k']:.5f} dB"],
        ["Gain at half the base rate", f"{s48['at_half']:.1f} dB", f"{s44['at_half']:.1f} dB"],
        ["Least attenuation from half the base rate up", f"{s48['stopband']:.1f} dB", f"{s44['stopband']:.1f} dB"],
    ]
    fastest, slowest = min(k.render_ms for k in kicks.values()), max(k.render_ms for k in kicks.values())
    longest_body, longest_attack = num(patch.MAX_BODY_MS / 1000), num(patch.MAX_ATTACK_MS / 1000)
    spec_rows = [
        ["Output sample rates", "", "", "44.1, 48, 88.2, 96", "", "kHz"],
        ["Internal sample rate", "44.1 or 48 kHz family", "", "705.6 or 768", "", "kHz"],
        ["Passband deviation", "0 to 20 kHz", "", f"±{ripple:.5f}", "±0.001", "dB"],
        ["Stopband attenuation", "from half the base rate", "110", f"{stopband:.1f}", "", "dB"],
        ["PCM headroom", "16 and 24 bit, dither never clips", "", "2", "", "LSB"],
        ["Body tuning", "tune_hz", "20", "", "1000", "Hz"],
        ["Body frequency", "at every pitch knot", "10", "", "20 000", "Hz"],
        ["Drive", "small-signal gain", "0", "", "48", "dB"],
        ["Knot spacing", "decimal values as written", "10", "", "", "µs"],
        ["Length", "body / attack curves", "", "", f"{longest_body} / {longest_attack}", "s"],
        ["Render time", "presets at 48 kHz, this build machine", f"{fastest:.0f}", "", f"{slowest:.0f}", "ms"],
    ]  # fmt: skip
    track_rows = [[num(ms), f"{f:,.0f}", f"{level:.1f}"] for ms, f, level in track]
    values = {
        "tokens": css_tokens(":root"),
        "figure_css": FIGURE_CSS,
        "version": __version__,
        "date": date.today().isoformat(),
        "machine": html.escape(machine),
        "libraries": f"NumPy {np.__version__}, SciPy {scipy.__version__}",
        "displays": "".join(displays),
        "cards": "".join(cards),
        "audio": audio,
        "bpm": num(BPM),
        "stopband": f"{stopband:.1f}",
        "ripple": f"{ripple:.5f}",
        "render_long": f"{longest.render_ms:.0f}",
        "long_seconds": num(round(longest.seconds, 2)),
        "tests": str(n_tests),
        "test_seconds": f"{test_s:.1f}",
        "fig_pipeline": pipeline,
        "fig_anatomy": anatomy_svg,
        "fig_spectrum": spectrum_svg,
        "fig_decimator": decimator_svg,
        "table_presets": table(
            ["Preset", "Tuning", "Length, ms", "Peak, dBFS", "True peak, dBTP", "Drive, dB", "Render, ms"],
            preset_rows,
            {2, 3, 4, 5, 6},
        ),
        "table_knots": table(["Curve", "Knot, ms", "Value"], knot_rows, {1, 2}),
        "table_track": table(["Time, ms", "Strongest frequency, Hz", "Level, dB"], track_rows, {0, 1, 2}),
        "table_decimator": table(["", "48 kHz family", "44.1 kHz family"], decimator_rows, {1, 2}),
        "table_spec": table(["Parameter", "Conditions", "Min", "Typ", "Max", "Unit"], spec_rows, {2, 3, 4}),
        "patch": toml_html(tight.text),
        "transcript": transcript(),
        "data": json.dumps(data, separators=(",", ":")).replace("</", "<\\/"),
    }
    page = (DOCS / "template.html").read_text(encoding="utf-8")
    for key, value in values.items():
        page = page.replace("{{" + key + "}}", str(value))
    if missing := re.findall(r"\{\{(\w+)\}\}", page):
        raise SystemExit(f"template markers without a value: {', '.join(missing)}")
    return page


def main() -> None:
    tests = run_tests()
    kicks = render_kicks()
    write_audio(kicks)
    figures = DOCS / "figures"
    figures.mkdir(exist_ok=True)
    (figures / "presets.svg").write_text(figure_presets(kicks), encoding="utf-8")
    (figures / "pipeline.svg").write_text(figure_pipeline(True), encoding="utf-8")
    (figures / "anatomy.svg").write_text(figure_anatomy(kicks["tight"], True)[0], encoding="utf-8")
    (figures / "spectrum.svg").write_text(figure_spectrum(kicks["tight"], True)[0], encoding="utf-8")
    (figures / "decimator.svg").write_text(figure_decimator(True)[0], encoding="utf-8")
    page = build_page(
        kicks,
        figure_anatomy(kicks["tight"], False),
        figure_spectrum(kicks["tight"], False),
        figure_decimator(False),
        figure_pipeline(False),
        tests,
    )
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    print(f"built docs/: {tests[0]} tests passed, 5 figures, {2 * len(kicks)} sounds, index.html")


if __name__ == "__main__":
    main()
