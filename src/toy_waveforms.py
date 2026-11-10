"""Toy VFE-board QC waveforms: five-peak staircase traces with decoys and noise.

Shape encoded here (from the operator's description of the real traces):
  * a train of many small spikes, the last of which is P0
  * P1..P4 rise monotonically in amplitude and each decays only part of the way
    back down, so the floor between consecutive peaks keeps climbing
  * small irregular structure sits on those floors and must NOT be detected
"""

from dataclasses import dataclass, field                         # - | config as a frozen-ish record

import numpy as np                                               # - | arrays and RNG
from scipy.signal import find_peaks                              # - | locate the visible apex of each peak


@dataclass
class ToyConfig:
    """Every number the generator uses; Monday's real traces only change these."""

    n_samples: int = 1024                                        # - | samples per trace
    peak_window: tuple = (0.14, 0.90)                            # - | P1..P4 live here; P0 sits ahead of P1
    p0_gap: tuple = (10.0, 60.0)                                 # - | samples from P0 to P1: the two are a pair
    n_pre_spikes: tuple = (8, 20)                                # - | count range of small spikes before P0
    pre_amp: tuple = (0.01, 0.06)                                # - | amplitude range of the small spikes
    p0_amp: tuple = (0.04, 0.09)                                 # - | P0 amplitude: top of the small-spike family
    main_amp: tuple = (0.25, 1.00)                               # - | P1..P4 amplitude span, monotone within a trace
    rise_width: tuple = (1.5, 14.0)                              # - | leading-edge sigma in samples, drawn freely
    decay_tau: tuple = (2.5, 30.0)                               # - | trailing tau, independent of the rise
    shape_bias: tuple = (0.5, 2.0)                               # - | per-trace tilt: <1 slow rise, >1 fast rise
    floor_frac: tuple = (0.25, 0.55)                             # - | fraction of a peak that stays as raised floor
    step_width: tuple = (3.0, 10.0)                              # - | sigmoid width of the floor step in samples
    n_decoys: tuple = (2, 6)                                     # - | decoy bumps per inter-peak floor
    decoy_frac: tuple = (0.04, 0.18)                             # - | decoy height as fraction of the local floor step
    noise_sigma: tuple = (0.004, 0.015)                          # - | Gaussian noise sigma
    spike_noise_rate: float = 0.002                              # - | fraction of samples hit by single-sample glitches
    spike_noise_amp: tuple = (0.02, 0.08)                        # - | glitch amplitude range
    tail_decay: tuple = (300.0, 1200.0)                          # - | slow decay of the final plateau, in samples
    jitter_frac: float = 0.50                                    # - | peak-spacing jitter as fraction of mean spacing
    labels: tuple = field(default=("P0", "P1", "P2", "P3", "P4"))  # - | channel order of the five peaks
    label_at_apex: bool = True                                   # - | label = visible apex, not the pulse centre
    apex_search: tuple = (6, 40)                                 # - | samples searched before / after the centre
    min_apex_prom: float = 0.01                                  # - | an apex must stand this far above its shoulders
    max_redraws: int = 200                                       # - | give up after this many invisible-peak draws


def spiky_pulse(t, t0, amp, rise, tau):                          # - | one asymmetric peak evaluated on a time grid
    """Fast Gaussian rise into an exponential decay: the 'spiky' look, not a sinusoid."""
    left = amp * np.exp(-0.5 * np.clip(((t - t0) / rise) ** 2, 0, 700))  # O(n) | smooth leading edge
    right = amp * np.exp(np.clip(-(t - t0) / tau, -700, 0))      # O(n) | exponential trailing edge, clamped
    return np.where(t < t0, left, right)                         # O(n) | glue the two halves at the apex
    # Total: time O(n), space O(n)


def smooth_step(t, t0, height, width):                           # - | floor that rises at a peak and stays up
    """Logistic step: the part of a peak that never comes back down."""
    return height / (1.0 + np.exp(-(t - t0) / width))            # O(n) | sigmoid centred on the peak
    # Total: time O(n), space O(n)


def draw_peak_positions(rng, cfg):                               # - | five ordered, jittered peak locations
    lo = int(cfg.peak_window[0] * cfg.n_samples)                 # - | first allowed sample
    hi = int(cfg.peak_window[1] * cfg.n_samples)                 # - | last allowed sample
    spacing = (hi - lo) / 3.0                                    # - | mean gap between P1..P4
    base = lo + spacing * np.arange(4)                           # O(1) | evenly spaced skeleton for the big four
    jitter = rng.uniform(-1, 1, 4) * cfg.jitter_frac * spacing   # O(1) | independent per-peak jitter
    main = np.sort(base + jitter)                                # O(1) | keep P1..P4 in time order
    gap = rng.uniform(*cfg.p0_gap)                               # - | P0 sits just ahead of P1, not spread out
    lo_edge = max(40.0, 0.03 * cfg.n_samples)                    # - | leave room for the pre-spike train
    p0 = max(main[0] - gap, lo_edge)                             # O(1) | never push P0 into the trace edge
    main[0] = max(main[0], p0 + 4.0)                             # O(1) | keep the pair resolvable if P1 was clipped
    pos = np.concatenate([[p0], main])                           # O(1) | P0..P4
    return np.clip(pos, lo_edge, cfg.n_samples - 5)              # O(1) | stay clear of the trace edges
    # Total: time O(1), space O(1)


def apex_labels(clean, centres, cfg):                            # - | where each peak visibly tops out
    """Label every landmark at the highest visible apex near its pulse centre.

    Each main peak is a transient pulse plus a permanent floor step centred on the
    same point, so the summed curve keeps rising after the centre and tops out
    later. A reader marks that apex, so the label goes there. The search never
    crosses the midpoint to a neighbouring landmark. Returns (labels, ok); ok is
    False when some landmark has no visible apex at all -- e.g. P0 swallowed by
    the leading edge of P1 -- and such a trace must not be used.
    """
    before, after = cfg.apex_search                              # - | search extent around the centre
    maxima, props = find_peaks(clean, prominence=cfg.min_apex_prom)  # O(n) | every visible apex
    labels = np.array(centres, dtype=np.float64)                 # - | default to the centre
    for k in range(5):                                           # O(5) | one landmark at a time
        lo = centres[k] - before                                 # - | left edge of the search
        hi = centres[k] + after                                  # - | right edge of the search
        if k > 0:                                                # - | stay clear of the previous landmark
            lo = max(lo, 0.5 * (centres[k - 1] + centres[k]))    # - | midpoint
        if k < 4:                                                # - | stay clear of the next landmark
            hi = min(hi, 0.5 * (centres[k] + centres[k + 1]))    # - | midpoint
        inside = maxima[(maxima >= lo) & (maxima <= hi)]         # O(m) | candidate apexes
        if inside.size == 0:                                     # - | nothing visible here
            return labels, False                                 # - | reject the trace
        labels[k] = float(inside[np.argmax(clean[inside])])      # O(c) | the highest of them
    ok = bool(np.all(np.diff(labels) > 0))                       # O(1) | labels must stay in time order
    return labels, ok                                            # - | visible-apex labels
    # Total: time O(n), space O(n)


def make_waveform(rng, cfg=None, return_clean=False, stats=None):  # - | one labelled toy trace
    """Return (signal, peak_positions) for a single synthetic good-board trace.

    With `cfg.label_at_apex` the positions are the visible apexes, and draws in
    which some landmark has no visible apex are thrown away and redrawn; `stats`
    (a dict) counts those redraws. With `return_clean`, also return the
    noiseless waveform.
    """
    cfg = cfg or ToyConfig()                                     # - | default numbers unless overridden
    for _ in range(cfg.max_redraws):                             # O(R) | redraw until every peak is visible
        sig, pos, clean = _draw_waveform(rng, cfg)               # O(n) | one candidate trace
        if not cfg.label_at_apex:                                # - | old behaviour: label the pulse centre
            break                                                # - | accept as drawn
        lab, ok = apex_labels(clean.astype(np.float64), pos, cfg)  # O(n) | visible-apex labels
        gap = lab[1] - lab[0]                                    # - | P0-P1 distance as a reader sees it
        ok = ok and cfg.p0_gap[0] <= gap <= cfg.p0_gap[1]        # - | the pair spec applies to the apexes
        if ok:                                                   # - | every landmark is visible
            pos = lab.astype(np.float32)                         # - | use them
            break                                                # - | accept
        if stats is not None:                                    # - | caller is counting
            stats["redraws"] = stats.get("redraws", 0) + 1       # - | one more rejected draw
    else:                                                        # - | never accepted
        raise RuntimeError("no draw with five visible peaks; check ToyConfig")
    if return_clean:                                             # - | caller wants the noiseless shape too
        return sig, pos, clean                                   # - | trace, labels, clean shape
    return sig, pos                                              # - | trace plus its five labels
    # Total: time O(R n), space O(n)


def _draw_waveform(rng, cfg):                                    # - | one unlabelled candidate trace
    """Draw signal, pulse centres and the noiseless waveform."""
    n = cfg.n_samples                                            # - | shorthand
    t = np.arange(n, dtype=np.float64)                           # O(n) | sample index as the time axis
    bias = rng.uniform(*cfg.shape_bias)                          # - | one tilt per trace: fast-rise or slow-rise
    pos = draw_peak_positions(rng, cfg)                          # O(1) | where P0..P4 sit
    sig = np.zeros(n)                                            # O(n) | start from a flat baseline

    n_pre = rng.integers(*cfg.n_pre_spikes)                      # - | how many small spikes precede P0
    pre_hi = max(4.0, pos[0] - 6.0)                              # - | never invert the window if P0 sits early
    pre_pos = rng.uniform(2.0, pre_hi, n_pre)                    # O(k) | scatter them before P0
    for p in pre_pos:                                            # O(k) | one small spike at a time
        amp = rng.uniform(*cfg.pre_amp)                          # - | small, well below P0
        r = rng.uniform(1.0, 5.0) / bias                         # - | pre-spike rise follows the trace tilt
        d = rng.uniform(2.0, 10.0) * bias                        # - | and its decay tilts the other way
        sig += spiky_pulse(t, p, amp, r, d)                      # O(n) | add the spike

    amps = np.sort(rng.uniform(*cfg.main_amp, 4))                # O(1) | P1<P2<P3<P4 by construction
    p0_amp = rng.uniform(*cfg.p0_amp)                            # - | P0 is the largest of the small family
    sig += spiky_pulse(t, pos[0], p0_amp, rng.uniform(1.0, 5.0) / bias,
                       rng.uniform(3.0, 12.0) * bias)            # O(n) | P0 itself, same tilt as the family

    floor_tops = []                                              # - | running floor height after each main peak
    for i in range(4):                                           # O(4) | P1..P4
        amp = amps[i]                                            # - | this peak's apex above the local floor
        rise = rng.uniform(*cfg.rise_width) / bias               # - | leading edge, tilted by the trace bias
        tau = rng.uniform(*cfg.decay_tau) * bias                 # - | trailing decay, tilted the opposite way
        frac = rng.uniform(*cfg.floor_frac)                      # - | how much of it stays up
        width = rng.uniform(*cfg.step_width)                     # - | how fast the floor steps up
        sig += spiky_pulse(t, pos[i + 1], amp * (1 - frac), rise, tau)       # O(n) | transient part of the peak
        sig += smooth_step(t, pos[i + 1], amp * frac, width)                 # O(n) | permanent floor step
        floor_tops.append(amp * frac)                            # - | remember the step size for decoy scaling

    for i in range(4):                                           # O(4) | decoys on each raised floor
        seg_lo = pos[i + 1] + 8.0                                # - | just after the peak
        seg_hi = pos[i + 2] - 8.0 if i < 3 else cfg.n_samples - 8.0  # - | up to the next peak (or trace end)
        if seg_hi - seg_lo < 12.0:                               # - | skip a segment too short to hold structure
            continue                                             # - | nothing to add here
        for _ in range(rng.integers(*cfg.n_decoys)):             # O(d) | a few bumps per segment
            amp = floor_tops[i] * rng.uniform(*cfg.decoy_frac)   # - | always well under the real step
            loc = rng.uniform(seg_lo, seg_hi)                    # - | anywhere inside the segment
            sig += spiky_pulse(t, loc, amp, rng.uniform(2.0, 8.0) / bias,
                               rng.uniform(4.0, 30.0) * bias)    # O(n) | add the decoy

    tail_t0 = pos[4] + 10.0                                      # - | plateau after P4 slowly sags
    sag = np.where(t > tail_t0, 1.0 - np.exp(-(t - tail_t0) / rng.uniform(*cfg.tail_decay)), 0.0)  # O(n) | sag shape
    sig *= 1.0 - 0.25 * sag                                      # O(n) | apply the slow droop
    clean = sig.copy()                                           # O(n) | the waveform before any noise

    sig += rng.normal(0.0, rng.uniform(*cfg.noise_sigma), n)     # O(n) | Gaussian readout noise
    n_glitch = int(cfg.spike_noise_rate * n)                     # - | how many single-sample glitches
    idx = rng.integers(0, n, n_glitch)                           # O(g) | where they land
    sig[idx] += rng.uniform(*cfg.spike_noise_amp, n_glitch) * rng.choice([-1, 1], n_glitch)  # O(g) | add glitches

    return sig.astype(np.float32), pos.astype(np.float32), clean.astype(np.float32)
    # Total: time O(n * (k + d)), space O(n)


def heatmap_targets(pos, n_samples, sigma=4.0):                  # - | five Gaussian target channels
    """One Gaussian bump per peak: the training target for the heatmap head."""
    t = np.arange(n_samples, dtype=np.float32)                   # O(n) | time axis
    maps = np.exp(-0.5 * ((t[None, :] - pos[:, None]) / sigma) ** 2)  # O(5n) | bump per channel
    return maps.astype(np.float32)                               # - | shape (5, n_samples)
    # Total: time O(5n), space O(5n)


def make_dataset(n_traces, seed=0, cfg=None, sigma=4.0):         # - | a batch of labelled toys
    cfg = cfg or ToyConfig()                                     # - | shared config
    rng = np.random.default_rng(seed)                            # - | reproducible stream
    X = np.zeros((n_traces, 1, cfg.n_samples), dtype=np.float32)  # O(Nn) | model input
    Y = np.zeros((n_traces, 5, cfg.n_samples), dtype=np.float32)  # O(5Nn) | heatmap targets
    P = np.zeros((n_traces, 5), dtype=np.float32)                # O(N) | true positions for scoring
    for i in range(n_traces):                                    # O(N) | one trace at a time
        sig, pos = make_waveform(rng, cfg)                       # O(n) | draw a toy
        X[i, 0] = sig                                            # - | store the trace
        Y[i] = heatmap_targets(pos, cfg.n_samples, sigma)        # O(5n) | store its targets
        P[i] = pos                                               # - | store its labels
    return X, Y, P                                               # - | dataset triple
    # Total: time O(N n), space O(N n)


if __name__ == "__main__":                                       # - | self-check on import-free run
    rng = np.random.default_rng(1)                               # - | fixed seed
    sig, pos = make_waveform(rng)                                # - | one trace
    assert sig.shape == (1024,), sig.shape                       # - | expected length
    assert np.all(np.diff(pos) > 0), pos                         # - | P0..P4 strictly ordered
    assert sig[int(pos[4])] > sig[int(pos[1])], "P4 must top P1"  # - | monotone peak amplitudes
    floor_12 = sig[int(pos[1]) + 30:int(pos[2]) - 10].mean()     # - | floor between P1 and P2
    floor_pre = sig[:int(pos[0]) - 10].mean()                    # - | level before P0
    assert floor_12 > floor_pre, (floor_12, floor_pre)           # - | floor never returns to baseline
    print("ok  peaks:", np.round(pos, 1), " apex:", np.round(sig[pos.astype(int)], 3))  # - | report the draw
