"""Read LeCroy .trc binary waveforms (WAVEDESC template, as written by HDO/WaveRunner).

Pure numpy, no third-party reader: the file is a short header block followed by the
raw ADC words. Value = VERTICAL_GAIN * raw - VERTICAL_OFFSET; time of sample i =
HORIZ_OFFSET + i * HORIZ_INTERVAL (seconds, relative to the trigger).
"""
import struct                                                    # - | fixed-layout header fields

import numpy as np                                               # - | the samples


def read_trc(path):                                              # - | one waveform file
    """Return (t_seconds, values, meta) for a LeCroy .trc file."""
    raw = open(path, "rb").read()                                # O(n) | whole file, a few hundred kB
    wd = raw.find(b"WAVEDESC")                                   # O(n) | header block start
    if wd < 0:                                                   # - | not a LeCroy waveform
        raise ValueError(f"{path}: no WAVEDESC block")           # - | refuse to guess
    order = "<" if struct.unpack_from("<h", raw, wd + 34)[0] else ">"  # - | COMM_ORDER: 1 = little-endian
    comm_type = struct.unpack_from(order + "h", raw, wd + 32)[0]  # - | 0 = int8 samples, 1 = int16
    desc, user, _rd1, trig, ris, _ra1, arr1 = struct.unpack_from(order + "7i", raw, wd + 36)  # - | block lengths, offsets 36..60
    count = struct.unpack_from(order + "i", raw, wd + 116)[0]    # - | WAVE_ARRAY_COUNT
    gain, voff = struct.unpack_from(order + "2f", raw, wd + 156)  # - | VERTICAL_GAIN, VERTICAL_OFFSET
    dt = struct.unpack_from(order + "f", raw, wd + 176)[0]       # - | HORIZ_INTERVAL of the full-rate record
    sparse = struct.unpack_from(order + "i", raw, wd + 136)[0]   # - | SPARSING_FACTOR: kept every k-th sample
    dt *= max(sparse, 1)                                         # - | spacing of the samples actually stored
    t0 = struct.unpack_from(order + "d", raw, wd + 180)[0]       # - | HORIZ_OFFSET
    start = wd + desc + user + trig + ris                        # - | first sample byte
    dtype = np.dtype(order + ("i2" if comm_type else "i1"))      # - | sample word
    n = min(count, arr1 // dtype.itemsize)                       # - | trust the smaller of the two counts
    adc = np.frombuffer(raw, dtype=dtype, count=n, offset=start)  # O(n) | raw words
    if start + n * dtype.itemsize > len(raw):                    # - | header lies about the layout
        raise ValueError(f"{path}: data block runs past end of file")
    vals = gain * adc.astype(np.float64) - voff                  # O(n) | physical units
    t = t0 + dt * np.arange(n)                                   # O(n) | seconds from trigger
    meta = {"n": n, "dt_s": dt, "sparse": sparse, "t0_s": t0, "gain": gain, "offset": voff,
            "comm_type": comm_type, "data_start": start, "file_bytes": len(raw)}
    return t, vals, meta
    # Total: time O(n), space O(n)
