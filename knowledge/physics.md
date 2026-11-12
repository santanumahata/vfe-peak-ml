# Physics — the input-current trace

- Five landmarks per shot: P0, then P1–P4 = A1, B1, A2, B2. The finder reports A1–B2; P0 is
  not one of its four peaks. Source: `src/real_data.py` docstring; `docs/RUNBOOK_vfe_peak_ml-DETAIL.md` Conventions.
- P0 in my labels = the peak just before A1: from the last bin before A1 below 0.4 × A1's
  height, walk back to the local maximum. Source: `src/real_data.py:30` (`RISE_FRAC = 0.4`), `src/real_data.py:36` (`p0_before_a1`).
- The P0 → A1 gap depends on the input voltage, so fixed-window P0 rules fail. Source:
  `docs/RUNBOOK_vfe_peak_ml-DETAIL.md` Gotchas (fixed-window P0 rule).
- Traces are averaged into 50-sample bins (10 µs at 200 ns spacing) and padded to 5120 bins.
  Source: `src/real_data.py:28-29`.
- The finder's peak time is the highest raw sample inside the peak; its start is the onset.
  Source: `docs/RUNBOOK_vfe_peak_ml-DETAIL.md` Conventions ("Real labels = the finder's output").
