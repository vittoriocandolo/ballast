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
import zlib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from ballast import patch, render
from ballast.voice import Voice

DOCS = Path(__file__).resolve().parent
RATE = 48000
PRESETS = ("tight", "driven", "round", "long")  # shortest first

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


def svg(width: int, height: int, title: str, subtitle: str, body: str, description: str) -> str:
    """A figure with its own palette, which follows the reader's light or dark theme."""
    head = f'<text class="title" x="28" y="36">{title}</text><text class="label" x="28" y="57">{subtitle}</text>'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="fig" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img"><title>{html.escape(description)}</title>{STYLE}'
        f'<rect class="bg" width="{width}" height="{height}" rx="10"/>{head}{body}</svg>\n'
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


@dataclass
class Kick:
    name: str
    text: str
    p: patch.Patch
    audio: np.ndarray  # float32 at 48 kHz
    report: render.Report


def render_kicks() -> dict[str, Kick]:
    kicks = {}
    for name in PRESETS:
        text = resources.files("ballast").joinpath("data", "presets", f"{name}.toml").read_text(encoding="utf-8")
        p = patch.loads(text)
        kicks[name] = Kick(name, text, p, *render.render(p, 16))
    return kicks


def envelopes(x: np.ndarray, columns: int, n: int):
    """Per-column minimum, maximum and RMS of the first n samples of x, zero past its end."""
    y = np.zeros(max(n, columns))
    y[: min(n, x.size)] = x[:n]
    starts = np.arange(columns) * y.size // columns
    counts = np.diff(np.append(starts, y.size))
    rms = np.sqrt(np.add.reduceat(y * y, starts) / counts)
    return np.minimum.reduceat(y, starts), np.maximum.reduceat(y, starts), rms


def waveform(panel: Panel, x: np.ndarray, seconds: float) -> str:
    """Peak and RMS envelopes of x over its first `seconds`, one column per unit of width."""
    columns = round(panel.w)
    lo, hi, rms = envelopes(x, columns, round(seconds * RATE))
    t = (np.arange(columns) + 0.5) * seconds / columns
    there_and_back = np.r_[t, t[::-1]]
    peak = panel.path(there_and_back, np.r_[hi, lo[::-1]], close=True)
    body = panel.path(there_and_back, np.r_[rms, -rms[::-1]], close=True)
    return f'<path class="peak" d="{peak}"/><path class="rms" d="{body}"/>'


def figure_presets(kicks: dict[str, Kick]) -> str:
    width, left, right, top, row, gap, seconds = 960, 214, 28, 84, 62, 24, 1.5
    body = []
    for i, k in enumerate(kicks.values()):
        y = top + i * (row + gap)
        panel = Panel(left, y, width - left - right, row, (0, seconds), (-1, 1))
        body.append(f'<path class="grid" d="M{left},{y}H{width - right}M{left},{y + row}H{width - right}"/>')
        body.append(f'<path class="zero" d="M{left},{panel.sy(0):.1f}H{width - right}"/>')
        body.append(waveform(panel, k.audio, seconds))
        tune, length, drive = k.p.body.tune_hz, k.report.source_ms, k.p.drive.db
        body.append(f'<text class="name" x="28" y="{y + 17}">{k.name}</text>')
        body.append(f'<text class="label" x="28" y="{y + 38}">{num(tune)} Hz · {note(tune)}</text>')
        body.append(f'<text x="28" y="{y + 57}">{num(length)} ms · drive {num(drive)} dB</text>')
    end = top + 4 * (row + gap) - gap
    axis = Panel(left, end, width - left - right, 0, (0, seconds), (0, 1))
    for v in np.arange(0, seconds + 1e-9, 0.25):
        body.append(f'<text x="{axis.sx(v):.1f}" y="{end + 20}" text-anchor="middle">{num(1000 * v)}</text>')
    body.append(f'<text class="label" x="{width - right}" y="{end + 40}" text-anchor="end">time, ms</text>')
    subtitle = "Peak (light) and RMS (dark) per column, on a full-scale axis"
    return svg(width, end + 60, "Four presets, rendered at 48 kHz", subtitle, "".join(body), "Waveforms of the presets")


def figure_pipeline() -> str:
    """Functional block diagram: from patch to file."""
    top = 76
    cy = top + 98
    body = [
        '<defs><marker id="pipeline-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path class="head" d="M0,1L10,5L0,9Z"/></marker></defs>',
        f'<rect class="zone" x="16" y="{top}" width="430" height="196" rx="8"/>',
        f'<text class="label" x="32" y="{top + 22}">16 × the base rate: 705.6 or 768 kHz</text>',
        f'<text class="label" x="795" y="{top + 22}" text-anchor="middle">at the file\'s sample rate</text>',
    ]

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
    subtitle = "From the patch's curves to a WAV file"
    return svg(960, top + 220, "Signal flow", subtitle, "".join(body), "Signal flow from patch to WAV")


def figure_anatomy(k: Kick) -> str:
    """A preset's three curves, and the first 60 ms of the sound they make."""
    v, tune, top = Voice(k.p, RATE), k.p.body.tune_hz, 96
    body = []

    def curve_panel(i, knots, curve, ms_end, ys, title, scale, *, ylog=False, yticks=(), yfmt=num, xticks=()):
        panel = Panel(64 + i * 304, top, 240, 150, (0, ms_end), ys, ylog=ylog)
        t = np.linspace(0, ms_end, 600)
        body.append(panel.frame(xticks, yticks, num, yfmt, "ms", title))
        body.append(f'<path class="s1" d="{panel.path(t, scale * curve(t / 1000))}"/>')
        kx = np.array([ms for ms, _ in knots])
        ky = scale * np.array([value for _, value in knots])
        keep = kx <= ms_end
        body.extend(f'<circle class="knot" cx="{x:.1f}" cy="{y:.1f}" r="4.5"/>' for x, y in
                    zip(panel.sx(kx[keep]), panel.sy(ky[keep]), strict=True))  # fmt: skip
        return panel

    pitch = curve_panel(
        0, k.p.body.pitch, v.pitch, 60, (40, 500), "Pitch, Hz", tune,
        ylog=True, yticks=(50, 100, 200, 500), yfmt=hz, xticks=(0, 20, 40, 60),
    )  # fmt: skip
    start, end = tune * k.p.body.pitch[0][1], tune
    x, y = pitch.sx(0) + 10, pitch.sy(start) + 4
    body.append(f'<text class="label" x="{x:.1f}" y="{y:.1f}">{num(start)} Hz</text>')
    body.append(
        f'<text class="label" x="{pitch.x + pitch.w}" y="{pitch.sy(end) - 10:.1f}" text-anchor="end">'
        f"{num(end)} Hz, {note(end)}</text>"
    )
    curve_panel(1, k.p.body.amp, v.body_amp, 240, (0, 1.05), "Body amplitude", 1, yticks=(0, 0.5, 1),
                xticks=(0, 80, 160, 240))  # fmt: skip
    curve_panel(2, k.p.attack.amp, v.attack_amp, 6, (0, 1.05), "Attack amplitude", 1, yticks=(0, 0.5, 1),
                xticks=(0, 2, 4, 6))  # fmt: skip
    wave_top = top + 214
    n = round(0.06 * RATE)
    t = np.arange(n) / RATE * 1000
    sound = Panel(64, wave_top, 848, 120, (0, 60), (-0.6, 0.6))
    body.append(sound.frame(range(0, 61, 10), (-0.5, 0, 0.5), num, num, "ms", "Output, first 60 ms, full scale ±1"))
    body.append(f'<path class="trace" d="{sound.path(t, k.audio[:n])}"/>')
    subtitle = "Three curves through the patch's knots, and the sound they make"
    description = f"The {k.name} preset's pitch, body and attack curves, and its first 60 ms"
    return svg(960, wave_top + 172, f"Anatomy of {k.name}", subtitle, "".join(body), description)


def spectrogram(x: np.ndarray, seconds: float, rows: int, columns: int, fmin: float = 20.0, fmax: float = 20000.0):
    """Level in dB, re the loudest cell, on log-spaced rows: each row demodulated at its own frequency
    and smoothed by a Hann window eight periods or 6 ms long, whichever is longer."""
    pad = 16384
    n = round(seconds * RATE)
    y = np.zeros(n + 2 * pad)
    y[pad : pad + min(n, x.size)] = x[:n]
    centers = pad + np.linspace(0, n, columns).astype(int)
    phase = 2j * np.pi * np.arange(y.size) / RATE
    level = np.empty((rows, columns))
    for i, f in enumerate(np.geomspace(fmin, fmax, rows)):
        window = np.hanning(int(min(2 * pad - 1, round(max(8 / f, 0.006) * RATE))) | 1)
        level[i] = np.abs(fftconvolve(y * np.exp(-phase * f), window, mode="same")[centers]) * 2 / window.sum()
    return 20 * np.log10(np.maximum(level, 1e-12) / level.max())


def figure_spectrum(k: Kick) -> str:
    seconds, floor, top = 0.25, 90, 96
    level = spectrogram(k.audio, seconds, rows=240, columns=500)
    index = np.round(np.clip(1 + level / floor, 0, 1) * 255).astype(np.uint8)[::-1]
    panel = Panel(64, top, 760, 300, (0, 1000 * seconds), (20, 20000), ylog=True)
    bar = Panel(852, top + 40, 16, 220, (0, 1), (-floor, 0))
    scale = np.arange(255, -1, -1, dtype=np.uint8)[:, None]
    ramps = {"light": ramp([LIGHT["paper"], *BLUES]), "dark": ramp([DARK["paper"], *BLUES[::-1]])}
    body = []
    for theme, palette in ramps.items():  # quiet recedes into the paper; in the dark theme loud is light
        for image, area in ((index, panel), (scale, bar)):
            body.append(
                f'<image class="{theme}" href="{png(image, palette)}" x="{area.x}" y="{area.y}" width="{area.w}" '
                f'height="{area.h}" preserveAspectRatio="none"/>'
            )
    ticks = (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
    body.append(panel.frame((0, 50, 100, 150, 200, 250), (), num, hz, "time, ms", "frequency, Hz", grid=False))
    body += [f'<text x="{panel.x - 8}" y="{panel.sy(f) + 4:.1f}" text-anchor="end">{hz(f)}</text>' for f in ticks]
    body.append(f'<text class="label" x="{bar.x}" y="{bar.y - 14}">level, dB</text>')
    body += [f'<text x="{bar.x + 24}" y="{bar.sy(v) + 4:.1f}">{num(v)}</text>' for v in (0, -30, -60, -90)]
    subtitle = "Level over time and frequency, to 90 dB below the peak"
    description = f"Spectrogram of the {k.name} preset over its first 250 ms"
    return svg(960, top + 352, f"Spectrum of {k.name}", subtitle, "".join(body), description)


def response(base: int, n: int = 1 << 20):
    taps = render.decimator(base)
    f = np.fft.rfftfreq(n, 1 / (base * render.K))
    return f, 20 * np.log10(np.maximum(np.abs(np.fft.rfft(taps, n)), 1e-300))


def figure_decimator() -> str:
    top = 96
    wide = Panel(64, top, 560, 260, (1000, 384000), (-170, 10), xlog=True)
    near = Panel(712, top, 220, 260, (0, 20000), (-0.00003, 0.00003))
    body = [wide.frame((1000, 10000, 100000), (-150, -110, -50, 0), hz, num, "frequency, Hz", "gain, dB")]
    band = wide.sx(20000) - wide.x
    body.append(f'<rect class="band" x="{wide.x}" y="{wide.y}" width="{band:.1f}" height="{wide.h}"/>')
    body.append(f'<path class="limit" d="M{wide.sx(22050):.1f},{wide.sy(-110):.1f}H{wide.x + wide.w}"/>')
    body.append(
        f'<text class="label" x="{wide.x + wide.w - 4}" y="{wide.sy(-110) - 8:.1f}" text-anchor="end">'
        "specified: ≥ 110 dB down</text>"
    )
    ticks = (-0.00002, 0, 0.00002)
    body.append(near.frame((0, 10000, 20000), ticks, hz, fixed5, "frequency, Hz", "passband detail, dB"))
    series = (("48 kHz family", 48000, "s1", "key1"), ("44.1 kHz family", 44100, "s2", "key2"))
    for i, (name, base, cls, key) in enumerate(series):
        f, gain = response(base)
        edges = np.geomspace(1000, base * render.K / 2, 561)
        envelope = np.maximum.reduceat(gain, np.searchsorted(f, edges[:-1]))  # the worst case in each column
        body.append(f'<path class="{cls}" d="{wide.path(np.sqrt(edges[:-1] * edges[1:]), envelope)}"/>')
        passband = f <= 20000
        body.append(f'<path class="{cls}" d="{near.path(f[passband][::40], gain[passband][::40])}"/>')
        y = wide.sy(-122) + 22 * i  # the legend, in the plot's empty corner
        body.append(f'<rect class="{key}" x="{wide.x + 16}" y="{y - 5:.1f}" width="18" height="3" rx="1.5"/>')
        body.append(f'<text class="label" x="{wide.x + 42}" y="{y:.1f}">{name}</text>')
        ripple, stopband = np.max(np.abs(gain[passband])), -np.max(gain[f >= base / 2])
        print(f"decimator {base}: {render.decimator(base).size} taps, ±{ripple:.5f} dB to 20 kHz, ≥ {stopband:.1f} dB")
    subtitle = "Both frozen filters at the internal rate, as their checksummed taps say"
    description = "Magnitude response of the two frozen decimators, with the passband in detail"
    return svg(960, top + 312, "Decimator response", subtitle, "".join(body), description)


def main() -> None:
    kicks = render_kicks()
    figures, audio = DOCS / "figures", DOCS / "audio"
    figures.mkdir(exist_ok=True)
    audio.mkdir(exist_ok=True)
    for k in kicks.values():
        render.export(audio / f"{k.name}.wav", k.p, k.text, k.audio, 16)
        print(f"{k.name}:", str(k.report).replace("\n", " | "))
    (figures / "presets.svg").write_text(figure_presets(kicks), encoding="utf-8")
    (figures / "pipeline.svg").write_text(figure_pipeline(), encoding="utf-8")
    (figures / "anatomy.svg").write_text(figure_anatomy(kicks["tight"]), encoding="utf-8")
    (figures / "spectrum.svg").write_text(figure_spectrum(kicks["tight"]), encoding="utf-8")
    (figures / "decimator.svg").write_text(figure_decimator(), encoding="utf-8")


if __name__ == "__main__":
    main()
