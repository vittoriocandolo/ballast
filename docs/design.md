# Design

Ballast computes one kick as a pure function of time, at 32 times the sample
rate, and band-limits it once.

    x(t) = b(t) + a(t) + v(t),    z = Q(C(E(G(W(D(x) + s))))),    out[m] = p · g_out · Σ_k h[k] · z[mK − k]

Here b is the body, a the attack, v the optional transient, D the drive, s
the optional sub, W the optional room, G the optional grit, E the EQ, C the optional low cut and Q
the optional clipper, all at the internal rate F; h is the frozen decimator of
the sample rate's family, K the decimation factor, g_out the output gain and p
the polarity.

![Signal flow: body and attack are summed, saturated and optionally high-passed at 32 times the base rate, then decimated, scaled and written to WAV](figures/pipeline.svg)

- [The body](#the-body)
- [Curves](#curves)
- [The attack](#the-attack)
- [The transient](#the-transient)
- [The saturator](#the-saturator)
- [The sub](#the-sub)
- [The room](#the-room)
- [Grit](#grit)
- [The EQ](#the-eq)
- [The low cut](#the-low-cut)
- [The clipper](#the-clipper)
- [Rates and the decimator](#rates-and-the-decimator)
- [Played kits](#played-kits)
- [The output stage](#the-output-stage)
- [Evaluation](#evaluation)
- [Reproducibility](#reproducibility)
- [Audition](#audition)
- [Code map](#code-map)

## The body

The body is a sine, or a saw that fades above `bright_hz`, with a second
harmonic, under an amplitude shape A_b. As a sine:

    b(t) = g_b / √(1 + h2²) · A_b(t) · (sin φ(t) + h2 sin 2φ(t))

The factor 1 / √(1 + h2²) keeps the body's power independent of `h2`, and
sin φ + h2 sin 2φ = sin φ (1 + 2 h2 cos φ) adds no zero crossings while
h2 ≤ ½.

The phase integrates the frequency f(t) = f₀ R(t), where R is the pitch
curve. Ballast integrates the cubic segments of R in closed form rather than
summing samples, so the phase carries no accumulated error and does not depend
on the sample rate:

    φ(t) = φ₀ + 2π f₀ τ(t),    τ(t) = ∫₀ᵗ R

With `phase_ref = "tail"`, τ is measured from the end instead: τ(t) = t from
the last pitch knot t_n on, and τ(t) = t_n − ∫ₜ^{t_n} R before it. The tail is
then exactly the free-running sine at f₀ with phase φ₀, whatever the sweep
does, and the integral from t to t_n is computed on its own so it stays exact
as t approaches t_n.

`bright_hz` turns the sine into a saw whose harmonics fade above it:

    Σ_{k≥1} r^(k−1) sin(kφ) / k = atan(r sin φ / (1 − r cos φ)) / r,    r = e^(−f(t) / bright_hz)

The fundamental stays at exactly 1, so brightness never takes weight away,
and the k-th harmonic, at k f, sits e^(−(k−1) f / bright_hz) below a saw's:
a roll-off fixed in Hz, which keeps its color as the pitch sweeps. Unlike the
drive's harmonics, these do not fade as the body decays, so the tail stays
audible where small speakers play nothing at the note itself. `h2` adds to
the saw's own second harmonic. The sum runs to infinitely many harmonics, but
they fade so fast that even the brightest body, driven by 48 dB, keeps every
alias 104 dB down.

`modes` add partials that ring from the strike, an acoustic body's own:

    m(t) = Σ_k 10^(db_k/20) · e^(−t ln 1000 / T_k) · sin(r_k · 2π f₀ ∫₀ᵗ R)

Each mode sits at a fixed ratio r_k to the body's frequency, so it follows
the pitch curve, the slight glide of a head relaxing after the hit, and
falls 60 dB in its own time T_k, under the body's shape. Its phase counts
from the trigger whatever the phase reference, so every mode starts at
zero, where the beater meets the head. An ideal membrane's first modes sit
at 1.594, 2.136, 2.296 and 2.653 times its fundamental; a real kick's heads,
tuning and air move them, and the table states whatever they are. The
modes add to the fundamental, which stays where it was.

![An acoustic body's spectrum and opening with five modes and without, over its first 150 and 60 ms](figures/modes.svg)

![One cycle of the body at four brightnesses, and the harmonics of a 1 kHz brightness on the note and an octave up](figures/brightness.svg)

![The body's shape for h2 = 0, 0.25 and 0.5, and the long preset's body around its last pitch knot with each phase reference](figures/body.svg)

A spectrogram shows the sweep over a whole kick, with the attack's short burst
at its start; the [preset gallery](presets.md) has one for every preset.

![Spectrogram of the tight preset: the body sweeps from 385 Hz down to 55 Hz while the attack adds a short broadband burst](figures/spectrum-tight.svg)

## Curves

Every curve is a monotone cubic Hermite interpolant (PCHIP) through its knots.
Interior slopes are the weighted harmonic mean of the neighboring secants
(Fritsch–Butland), or zero where the secants differ in sign or one of them is
zero; end slopes use the shape-preserving three-point rule. Every slope this
produces keeps both normalized slopes of each segment within the
Fritsch–Carlson region, so each segment is monotone and stays between its two
knot values: a curve never overshoots.

![The tight preset's pitch curve and body amplitude through their knots: Ballast's curves stay between the knots, while cubic splines through the same knots swing far past them](figures/curves.svg)

Amplitude shapes fix both end slopes to zero, so they leave and join silence
smoothly; the pitch curve fixes its last slope to zero, so it lands on the
note without a kink. Each curve also stores the integral of every segment from
both ends, which gives the exact phase above.

## The attack

The attack mixes a tone and noise at constant power, under its own amplitude
shape A_a:

    a(t) = g_a · A_a(t) · (cos(πm/2) sin(2π f_t τ_a(t)) + sin(πm/2) n(t)),    τ_a(t) = ∫₀ᵗ R_a

where m is `noise_mix`, f_t is `tone_hz` and R_a is the tone's own pitch
curve, `attack.pitch`; at m = 0 and m = 1 the weights are exactly 1 and 0.

The tone's phase integrates its curve in closed form, as the body's does. A
curve held at ratio 1 is the fixed tone, computed as τ_a(t) = t itself. A
curve that falls into ratio 1 sweeps the tone down into `tone_hz` while the
attack sounds: a zap, which spreads the click's tone over the octaves it falls
through and raises its center of brightness.

![A zap: the tone's frequency falling from 10 kHz into 2.5 kHz in 2 ms, and the tone it makes, against the fixed tone](figures/zap.svg)

The noise is a seeded continuous-time Gaussian process:

    n(t) = γ Σ_{k=1}^{20000} |H(f_k)| ρ_k cos(2π f_k t + θ_k),    f_k = k Hz

SHAKE-256, keyed by `ballast/noise/v1` and the seed, gives 16 bytes per
component. They become two 53-bit uniforms and, by Box–Muller, a Rayleigh
amplitude ρ_k and a uniform phase θ_k, so each component is a circular
Gaussian and n(t) is exactly Gaussian at every t. |H| is a pair of
second-order Butterworth skirts with their half-power points at the band
edges, and γ makes the expected power ½, that of a unit sine.

![The weights of tone and noise against noise_mix, and the tight preset's noise band, as designed and as its seeded noise measures](figures/attack.svg)

Because the noise is a function of t, it is the same signal at every sample
rate, and because the shaping only reweights components, editing the band
keeps the same noise. It repeats every second, so Ballast computes one second
of it per internal rate with a single inverse real FFT.

## The transient

A patch may layer a sampled click, or any short sound, over the attack. The
optional `[transient]` section holds its samples, at any of the six sample
rates, and plays them from t₀ = `start_ms`, ρ = `pitch` times as fast, at the
level g_v of `level_db` and with the polarity p_v:

    v(t) = p_v · g_v · U(ρ (t − t₀) F_t)

U reads u, the samples brought up to F_t, the internal rate of their own
family, by the family's frozen filter: the one that brings the kick down. Each
sample is followed by K − 1 zeros, and the result goes through K times the
filter, so the transient is flat to 20 kHz within ±0.000014 dB on the way up,
and again on the way down; its images stay at least 116 dB down; and nothing
rings before its onset, which the filter delays by 0.07 to 0.08 ms below 5
kHz.

The voice reads u at q = ρ (n − n₀) F_t / F, with n₀ = t₀ F. When the samples
share the patch's family, the pitch is 1 and the start falls on a sample, q is
whole and the voice takes u's own samples. Otherwise it reads between them, by
the Lagrange polynomial through the six nearest. At 32 times the base rate
those six are so close together that the read stays within 140 dB of exact
interpolation, below the filter's own floor: a 44.1 kHz click plays in a 48
kHz patch, and at any pitch, with no filter designed while rendering.

The samples are stored as little-endian float32 in base64 and peak at exactly
1, so `level_db` is the transient's peak level. It sums with the body and the
attack before the drive, which glues it to them, and the render runs on until
its interpolation has rung out.

![A click of three decaying modes at pitches 0.5, 1 and 2, and the tight preset's opening with it layered and without](figures/transient.svg)

## The saturator

    S_α(x) = tanh(αx) / tanh α,    α / tanh α = 10^(db/20)

The saturator keeps full scale at full scale, S_α(±1) = ±1 exactly, and its
small-signal gain is `drive.db`. α solves that equation numerically; at 0 dB,
S is the identity, an exact bypass. It is computed as sign(x) · tanh(α|x|) /
tanh α, so it is odd by construction.

![The saturator's transfer curve, and a full-scale sine through it, at drives from 0 to 48 dB](figures/saturator.svg)

A curve makes harmonics without limit, and those near a multiple of the
internal rate fold back into the audio band, where no filter can tell them
from the sound. So Ballast never reads the curve at single points. Each output
sample is the average of S along the input's path from one sample before to
one after, weighted by a triangle that peaks on the sample, with the input
taken as linear between samples:

    y[n] = R(x[n], x[n+1]) + R(x[n], x[n−1]),    R(a, b) = ∫₀¹ (1 − u) S(a + u(b − a)) du

This is second-order antiderivative anti-aliasing, centred on each sample so
that it adds no delay. The triangle's spectrum has a double zero at every
multiple of the internal rate, exactly where the aliases come from. R has a
closed form in S's first and second antiderivatives, which for tanh involve
log cosh and the dilogarithm, arranged so that rounding cancels nothing large;
a Taylor series takes over for steps much shorter than 1/α. Against numerical
integration it is exact to within 10⁻¹¹. For a small signal the average is
(x[n−1] + 4x[n] + x[n+1]) / 6, which would take 0.0097 dB off at 20 kHz; a
three-tap correction keeps the passband flat within 0.00002 dB.

![The worst alias of a full-scale tone for every tone and drive: the curve read at each sample aliases up to −38 dB, Ballast's drive no higher than −107 dB](figures/aliasing.svg)

Measured with full-scale tones, every setting the patch allows keeps its
aliases at least 104 dB down. The worst corner is a 20 kHz tone at the maximum
drive of 48 dB: 107 dB down in the 48 kHz family, 104 dB in the 44.1 kHz one,
whose internal rate is lower. Up to 24 dB of drive, every tone stays more than
130 dB down. Read at each sample instead, the same curve aliases as loud as
−38 dB.

Two controls shape the drive. `asymmetry` adds an even term to the curve:

    D(x) = S_α(x) + k S_α(x)² (1 − S_α(x)²),    k = asymmetry / 2

The term is 0 at 0 and at ±1 and flat at 0, so full scale stays exactly at
full scale and the small-signal gain stays `drive.db`; the curve stays
monotonic up to k = ½, where `asymmetry` ends. It works where S is not yet
flat, so its even harmonics rise as the body decays through the curve: at
6 dB of drive, a half-scale sine gains a second harmonic of −24 dB at the
most, while a full-scale one gains little. Even harmonics come with DC, which
the low cut removes. Its antiderivatives are elementary, so it is averaged
around each sample like S, with the same accuracy. `mix` blends the driven
signal with the undriven one; the averaging adds no delay, so the two line up
sample for sample and blend without combing.

![The drive's curve at 6 dB with asymmetry 0, 0.5 and 1, and a half-scale sine through each: the positive half grows fuller](figures/asymmetry.svg)

## The sub

The optional sub is the sine that the body's tail meets, under its own
amplitude shape A_s and level g_s:

    s(t) = g_s · A_s(t) · sin(φ₀ + 2π f₀ (t − c)),    c = t_n − ∫₀^{t_n} R

With `phase_ref = "onset"`, c makes it the very sine the body becomes from
the last pitch knot t_n on; with `"tail"`, c is 0. So the sub only ever adds
to the tail: where the body has landed, the two share every zero crossing. It
joins after the drive, so it stays a clean sine under a body driven to the
limit, and before the low cut and the clipper, whose promise of an exact true
peak covers the whole file. A sub an octave down would sit below 22 Hz for
any kick tuned under F1, so the sub stays on the note.

![The body and the sub meeting at the last pitch knot, and the tight preset with a sub under it and without](figures/sub.svg)

## The room

The optional room adds a tail to the kick: W(y) = y + g_w · (h ∗ y), delayed
by `predelay_ms`. The tail h is seeded Gaussian noise whose every frequency
fades on its own clock:

    h(t) = Σ_k ρ_k cos(2π f_k t + θ_k) · e^(−t ln 1000 / T(f_k)),    T(f) = decay / (1 + f / damping)

so it darkens as it fades, as air and walls darken a real room. Its
amplitudes and phases come from SHAKE-256 under its own key,
`ballast/room/v1`, and the patch's seed, spaced so that the tail never
repeats within itself. The fades are those of octave bands from 62.5 Hz to
16 kHz, each at the decay of its center, with every component weighted across
its two nearest bands, linearly in octaves; the weights sum to exactly 1, so
the tail starts as plain white noise. Measured in third octaves, each falls
60 dB within 10% of its designed time.

The tail lives at the family's base rate, since it holds nothing above
20 kHz, and lasts 1.5 × `decay_ms`, to −90 dB, fading out smoothly over its
last tenth. The room hears the kick as it sounds: the frozen filter brings
the kick down to the base rate, where it meets the tail, and brings the
result back up, in parallel chunks that match a single pass bit for bit, so
every rate of a family shares it and a long room costs a convolution at the
base rate rather than at 32 times that. What the kick makes of the tail is
then scaled so that its energy is exactly `level_db` against the kick's own:
at 0 dB the room holds as much energy as the kick. So the level says how
much room and the decay only how long, and the same level suits a kick of any
spectrum, where a tail of fixed gain would ring louder under a kick whose
energy sits low, where it decays slowest. The room joins after the drive,
because saturating a tail of noise only fills it with grit, and before the
EQ and the clipper, so they shape the whole sound and its exact true peak.

![The room's energy decay in three third octaves against their designed times, and the tight preset in a room and without](figures/room.svg)

## Grit

Ballast never aliases, except here, on purpose and by name. The optional
`[grit]` section models the samplers that drum machines and early house were
made on. The sampler hears the kick band-limited to 20 kHz, as an analog
source would be: the frozen filter takes it down to the base rate and back
up. It samples that signal at `rate_hz`, reading it between internal
samples by 6-point Lagrange interpolation, with no filter against aliasing,
so everything above half its rate folds back down. It rounds each sample to
`bits`, without dither, clipping at its full scale after `input_db`, and
plays the samples back through a zero-order hold: the staircase its
converter made, whose images reach up into the audio band, and whose sinc
envelope dulls the top.

Its aliasing and its staircase are the sound, so Ballast must not add its
own. Each internal sample is the staircase's exact average under the triangle
from one sample before to one after, the drive's kernel, with the same 3-tap
correction; the steps lie at least two samples apart, so a triangle meets at
most one, and the average is a closed form. Sampled at 26.04 kHz, a 15 kHz
tone comes back as its alias at 11.04 kHz and its image at 15 kHz, each
within 0.01 dB of the hold's sinc, and nothing else above the converter's own
rounding, 110 dB down. `input_db` is undone at the output, so it changes the
grit, not the level. The EQ's low-pass, after it, plays the sampler's output
filter.

![A 15 kHz tone through a 26.04 kHz sampler against the hold's sinc, and the tight preset's top end through it](figures/grit.svg)

## The EQ

`[eq]` has three bands after the drive, the sub and the room: a low shelf for
weight, a bell for knock, and a high shelf for air; and a resonant low-pass,
the filtered sound of a sampled kick. Each is one biquad from the audio EQ
cookbook, its coefficients computed in closed form like the low cut's: shelves
of slope 1, a bell whose Q sets its width, and a low-pass whose Q lifts its
corner. Designed by the bilinear transform at 32 times the base rate, they
barely warp: the bands stay within 0.004 dB of their analog prototypes all
the way to 20 kHz, where an EQ running at 48 kHz bends its top end, and the
low-pass within 0.012 dB, where its falling slope shows the last of the
warping. The low-pass is static, the same for the whole kick: a sweep belongs
to a track, across its bars. A band at 0 dB, or the low-pass at 0 Hz, is not
there at all.

![The EQ's shelves at four gains and its bell at three widths, as they respond at the internal rate](figures/eq.svg)

## The low cut

`output.low_cut_hz` adds a second-order Butterworth high-pass, 12 dB per
octave with its half-power point at the corner, after the saturator. Ballast
derives its coefficients in closed form, by the bilinear transform with the
corner prewarped, rather than calling a filter design routine, so they cannot
drift with SciPy versions. It filters the internal signal in one causal pass:
every rate of a family shares the result, and a longer render still never
changes earlier samples. At 0 the filter is not there at all.

A high-pass shifts the phase of low frequencies, so with `phase_ref = "tail"`
the tail is the free-running sine before the low cut, and the same sine
shifted by the filter's constant phase at f₀ after it.

![Magnitude and phase of the low cut at corners of 10, 30, 100 and 1000 Hz](figures/low-cut.svg)

## The clipper

The clipper makes a kick loud by shaving its peaks. It comes after the low
cut, at the internal rate, with a curve that follows the input up to the
knee's start u₀, 10^(−knee_db/20) of the threshold, then bends along a
parabola that meets the threshold with zero slope at 2 − u₀:

    Q(u) = u for |u| ≤ u₀,    then a parabola to ±1 at 2 − u₀,    then ±1

With no knee it is a hard clip. Its second derivative is constant in the knee,
or a jump at a hard corner, so the triangle's average has an exact form
without cancellation, and the clipper is anti-aliased like the drive.

`clip.db` says how far the file's true peak comes down, exactly. The
threshold follows the kick's own peak, as the output stage's peak modes do,
so the same setting clips every kick, and every key of a multisample, by the
same amount. The decimator's phase rebuilds part of a clipped peak, so the
threshold is solved for, on the file's own reconstruction (see
[the output stage](#the-output-stage)). Where the curve does not bend, the
clipper is the 5-tap linear filter its averaging makes of a straight line, so
a trial clips only the runs of samples within two of a clipped one, convolves
their difference from that filter with the decimator, and adds it to the
reconstruction of the filtered kick, made once: it costs what clips, not what
the kick lasts. The first trial clips the internal peak by `clip.db`; secant
steps on the slope the trials measure bracket the threshold, and Brent's method
finishes, in four to ten trials. The true peak lands within 0.0000001 dB of the
setting.
Everything below the knee keeps its level, so with a peak or true-peak target
in `[output]`, the decibels taken off the peaks come back as loudness.

![The clipper's curve with a hard knee and with 3 and 6 dB knees, and the tight preset's first 12 ms before and after 4 dB of clipping](figures/clipper.svg)

## Rates and the decimator

| Sample rate | Internal rate F | Factor K | Decimator |
|---|---|---|---|
| 44.1 kHz | 1411.2 kHz | 32 | 44.1 kHz family, 5231 taps |
| 88.2 kHz | 1411.2 kHz | 16 | 44.1 kHz family, 5231 taps |
| 176.4 kHz | 1411.2 kHz | 8 | 44.1 kHz family, 5231 taps |
| 48 kHz | 1536 kHz | 32 | 48 kHz family, 2919 taps |
| 96 kHz | 1536 kHz | 16 | 48 kHz family, 2919 taps |
| 192 kHz | 1536 kHz | 8 | 48 kHz family, 2919 taps |

Each family has one minimum-phase FIR, frozen and shipped as data with a
SHA-256 checksum that Ballast verifies before use, so the filter cannot drift
with SciPy versions. Both pass 0–20 kHz within ±0.000014 dB, attenuate
everything from half the base rate up by at least 116.4 dB, and have taps that
sum to exactly 1. Minimum phase puts all the ringing after the transient, none
before it. `tools/decimators.py` records how they were designed: a Kaiser
linear-phase prototype, made minimum phase by the homomorphic method.

![Magnitude response of both decimators: flat to 20 kHz, at least 110 dB down from half the base rate, with the passband ripple in detail](figures/decimator.svg)

A 96 or 192 kHz render uses the same internal signal and filter as the 48 kHz
one with a half or a quarter of the factor, so it is the 48 kHz file with the
in-between samples filled in, bit for bit; 88.2 and 176.4 kHz relate to
44.1 kHz the same way. The higher rates therefore carry the same audio band,
up to 20 kHz, with no new content above it.

![The tight preset's click at 48, 96 and 192 kHz, sample by sample: every 48 kHz sample is also a 96 and a 192 kHz sample](figures/rates.svg)

The file keeps the whole filter tail: a render is N internal samples, with N
the number of n such that n / F ≤ T, the patch's length, and out has
⌊(N + L − 2) / K⌋ + 1 samples for L taps.

## Played kits

A drum played by a person never sounds the same twice, and sounds different
hit softer, not just quieter. `[play]` describes both, and `ballast pack`
renders them. A softer hit, at x from 1, the patch itself, down to 0, the
softest, loses (1 − x) of what `[play]` says the softest loses: level before
the drive, so softer hits also come out cleaner; more of its click; and part
of its pitch drop, every pitch ratio becoming 1 + (r − 1)(1 − (1 −
`soft_punch`)(1 − x)). A repeat redraws the attack's noise from a seed of its
own and moves level, punch, length and click by small amounts, drawn from
SHAKE-256 under `ballast/play/v1`, so every repeat is the same in every
version.

The hits of a kit go through one chain. Its clipper takes `clip.db` off the
hardest hit's peak, and every other hit meets that same threshold, rather
than one solved for its own peak, which would make every repeat clip
differently: a repeat whose noise happened to peak higher would lose body.
The output gain is the smallest any hit needs, so the kit's loudest hit,
whichever it is, sits at the target, and every other keeps its level against
it.

## The output stage

In `gain` mode the output is multiplied by 10^(db/20). In `peak` mode it is
scaled so its highest sample reaches the target, capped at the format's
ceiling. In `true-peak` mode it is scaled so its true peak reaches the target,
or its highest sample reaches the ceiling, whichever comes first. Then the
polarity applies, and the result is converted once to float32.

The true peak is the peak between the file's samples, as an ideal converter
plays them back. Ballast does not estimate it from the file, as a meter does by
oversampling it 4 times, which can read a peak near 20 kHz low by tenths of a
decibel. It reads it on the file's own reconstruction: the decimator's output
at every internal sample, before it keeps one in K, which is band-limited to
116 dB, so its samples, 32 to each of the file's, are the waveform between the
file's samples. A parabola through each crest and its two neighbors finds the
peak between them to within 0.000001 dB. The reconstruction comes from one FFT
convolution, used only to measure; the file itself comes from the exact
decimation. Every rate of a family shares the true peak, and no filter is
designed to find it, so it cannot drift with SciPy.

PCM exports have a ceiling 2 LSB below full scale. They add TPDF dither of
±1 LSB, the difference of two 32-bit uniforms from SHAKE-256 keyed by
`ballast/dither/v1` and the seed, and round. A render within the ceiling stays
within range after dither, and exact zeros stay zero.

![A 15 kHz click whose true peak lies between samples, above the highest sample, on the file's own reconstruction, and the triangular density of the seeded dither](figures/output-stage.svg)

Nothing trims or fades, and only the low cut, when it is on, high-passes or
removes DC.

## Evaluation

The internal signal is computed in blocks of 2¹⁵ samples aligned to the
trigger, which bounds memory; each source is evaluated only over its own
support, and the two samples either side of a block that the drive averages
over are computed again rather than carried. A transient is read once, over
its whole support, and each block adds its share. The decimator then runs in
parallel chunks of 2048 outputs. Each output of the decimator is one fixed sum
over its own inputs, and each chunk starts on the factor's grid with every
input its outputs need, so the chunks match a single pass bit for bit. Neither
the blocks nor the threads change a sample, and because the filter is causal,
a longer render never changes earlier samples. On a 12-core laptop, the 1.45 s
`long` preset renders in about a quarter of a second.

## Reproducibility

The renderer is deterministic: no randomness, no state, and no dependence on
block sizes or thread counts. The same patch on the same platform, with the
same NumPy and SciPy versions, renders the same samples bit for bit, every
time. Each WAV records the renderer version, the decimator checksum, the
platform and the library versions next to the patch.

Across platforms or library versions, the elementary functions (sine, tanh,
logarithm, dilogarithm) and the FFT may round their last bit differently, so
samples may differ at the level of rounding. One solved value can carry that
further: the clipper's threshold settles within its tolerance, 10⁻⁸ of
itself, so a clipped kick may differ by up to that, 160 dB down, under the
smallest step of a 24-bit file. Making every platform agree to the bit would
take correctly rounded elementary functions of Ballast's own and fixed sums
throughout, slower and with nothing to hear. Instead, the tests compare
samples of every preset, at both base rates, with stored references, within a
millionth of full scale, on every platform CI runs. The inputs that define the
sound are exact and portable: the patch, the SHAKE-256 bytes behind the noise,
the room, the variations, the repeats and the dither, and the filter taps.

## Audition

Audition reuses the renderer and adds four small parts:

- **Hits** mixes the output block by block. Every sound is an unchoked
  one-shot: each hit plays the current buffer to its end, and loop k of the bed
  plays until loop k + 1 starts. Hits and loops start on the same beat
  rounding, so the bed never drifts from the kick, and sounds are summed in
  order of their start, so the block size never changes the mix.
- **Watcher** polls the patch file every 0.1 s and reads it again when its
  size or modification time changes.
- **The stream callback** fills each block from Hits and counts the device's
  underflows; the play loop reports them, since the audio thread must never
  print.
- **The play loop** renders each new version and polls once more: if the file
  changed during the render, the result is dropped and the newer version
  rendered next. Only a valid render that fits full scale replaces the buffer,
  at the next hit.

## Code map

| Module | Responsibility |
|---|---|
| `patch.py` | the schema, declared once on the dataclass fields; parsing, canonical TOML, edits, tuning, transients' samples |
| `curve.py` | monotone cubic Hermite curves with exact integrals |
| `noise.py` | the seeded continuous-time noise, and the PCM dither |
| `voice.py` | body, attack, transient, saturator and sub as pure functions of time, evaluated in blocks |
| `shaper.py` | the drive's and the clipper's curves, anti-aliased by their antiderivatives |
| `family.py` | the rate families: base and internal rates, and the frozen filters, checked by checksum |
| `render.py` | EQ, low cut, clipper, parallel decimation, output stage, loudness, report, export |
| `vary.py` | seeded variations, drawn from SHAKE-256, and grids along axes |
| `play.py` | softer hits and repeats, as `[play]` describes them |
| `pack.py` | kits: names by key, played layers and repeats with their SFZ map, the gates, the manifest |
| `grit.py` | the vintage sampler: its band-limited input, converter and exact staircase |
| `room.py` | the seeded tail of the room, and its convolution |
| `wav.py` | the RIFF writer, the reader of beds and transients, the chunk reader behind `extract`, and writing any file whole or not at all |
| `audition.py` | hits, bed, watcher and playback |
| `cli.py` | the commands |
| `data/` | the two decimators and the eleven presets |
