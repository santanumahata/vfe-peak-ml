# VFE-board QC peak finding (ML) — Quick Reference

**Detail:** `RUNBOOK_vfe_peak_ml-DETAIL.md` — expected output, gotchas, conventions, changelog.
**Where it runs:** this machine, `$ENV` micromamba env, RTX 4070 GPU; every command from the repo root.
**Working copy:** `~/Documents/vfe_peak_ml/vfe-peak-ml` (clone + local data and outputs).

### 1. Environment
```bash
micromamba run -n $ENV pip install --force-reinstall --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
micromamba install -n $ENV -c conda-forge pyyaml -y
```

### 2. List and unpack the archive's metadata
```bash
tar -tf ~/Downloads/results.tar > <members.txt> && mkdir <yaml_dir> && tar -xf ~/Downloads/results.tar -C <yaml_dir> --no-same-owner --no-same-permissions --wildcards '*.yaml'
```

### 3. First look at real traces
```bash
micromamba run -n $ENV python -m diagnostics.real_first_look <yaml_dir> ~/Downloads/results.tar <extract_dir> diagnostics/<date>_real_first_look.png <members.txt>
```

### 4. Build the real dataset
```bash
micromamba run -n $ENV python -m src.real_data <yaml_dir> ~/Downloads/results.tar <members.txt> data
```

### 5. Train on real traces
```bash
micromamba run -n $ENV python -m src.train_real --arch unet --out outputs/real/unet
```

### 6. Compare model sizes
```bash
for cfg in "u16 16 3" "u32 32 3" "u64 64 3" "u32l4 32 4" "u32l5 32 5"; do set -- $cfg; micromamba run -n $ENV python -m src.train_real --arch unet --base $2 --levels $3 --out outputs/real/sweep/$1; done && for cfg in "t256x4 4" "t256x8 8"; do set -- $cfg; micromamba run -n $ENV python -m src.train_real --arch transformer --d 256 --tdepth $2 --out outputs/real/sweep/$1; done
for s in 1 2; do for cfg in "<tag> <base> <levels>"; do set -- $cfg; micromamba run -n $ENV python -m src.train_real --arch unet --base $2 --levels $3 --seed $s --out outputs/real/sweep_seed/$1_s$s; done; done
micromamba run -n $ENV python -m diagnostics.real_sweep_summary
```

### 7. Failure shots
```bash
micromamba run -n $ENV python -m diagnostics.real_hard_cases outputs/real/unet outputs/real/transformer diagnostics/<date>_real_hard_cases.png
```

### 8. ML vs finder PDF
```bash
micromamba run -n $ENV python -m src.report_real outputs/real/<date>_ml_vs_finder.pdf outputs/real/sweep/u32l4 data/archive_yaml
```

### 9. Full write-up (LaTeX note)
```bash
micromamba run -n $ENV python -m writeup.make_figures && cd writeup/src && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../outputs vfe_peak_ml_writeup.tex && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../outputs vfe_peak_ml_writeup.tex
```

### 10. Result YAMLs into the repo's data folder
```bash
mkdir -p data/archive_yaml && tar -xf ~/Downloads/results.tar -C data/archive_yaml --no-same-owner --no-same-permissions --wildcards 'results/*/*.yaml'
```

### 11. Today's finder vs the network on the held-out shots
```bash
micromamba run -n $ENV python -m diagnostics.finder_vs_ml data/archive_yaml ~/Documents/vfe_peak_ml/lvr-tester
```

### 12. Tester GUI on this laptop
```bash
~/Documents/vfe_peak_ml/run_lvr_tester.sh --selftest; tail -3 ~/Documents/vfe_peak_ml/lvr_tester_gui.log
```

### 13. Build a replay folder from an archived run
```bash
micromamba run -n $ENV python -m diagnostics.make_replay_run data/archive_yaml ~/Documents/vfe_peak_ml/lvr-tester --run <run>
```

### 14. Replay in the GUI, side by side, with the peak review
```bash
~/Documents/vfe_peak_ml/run_lvr_tester.sh --selftest --replay ~/Documents/vfe_peak_ml/replay/<run>; tail -2 ~/Documents/vfe_peak_ml/lvr_tester_gui.log; ~/Documents/vfe_peak_ml/run_lvr_tester.sh --replay ~/Documents/vfe_peak_ml/replay/<run>
```

### 15. Today's finder on every shot
```bash
micromamba run -n $ENV python -W ignore -m diagnostics.finder_all_shots data/archive_yaml ~/Documents/vfe_peak_ml/lvr-tester --workers 12 > outputs/real/finder_all/full_run.log 2>&1
```

### 16. Limits against the production-card distribution
```bash
mkdir -p ~/Documents/vfe_peak_ml/limits_study/data/records && tar -xf ~/Downloads/results.tar -C ~/Documents/vfe_peak_ml/limits_study/data/records --no-same-owner --no-same-permissions --wildcards 'results/*_P1_B[12]_*/*.yaml'
cd ~/Documents/vfe_peak_ml/limits_study && micromamba run -n $ENV python diagnostics/2026-11-13_limits_distribution.py data/records ../lvr-tester/configuration/test_types/incoming_A.yaml diagnostics/2026-11-13_limits_distribution
```

### 17. P0 label from the finder
```bash
micromamba run -n $ENV python -m src.relabel_p0_finder
```

### 18. Synthetic merged peaks
```bash
micromamba run -n $ENV python -m src.make_merges
```

### 19. Train on the relabelled and on the merged dataset
```bash
micromamba run -n $ENV python -m src.train_real --arch unet --base 32 --levels 4 --data data/real_shots_p0finder.npz --out outputs/real/p0finder/u32l4 > outputs/real/p0finder_u32l4.log 2>&1
micromamba run -n $ENV python -m src.train_real --arch unet --base 32 --levels 4 --data data/real_shots_merge.npz --out outputs/real/merge/u32l4 > outputs/real/merge_u32l4.log 2>&1
```

### 20. Today's finder vs a chosen model
```bash
mkdir -p outputs/real/finder_vs_ml_merge && micromamba run -n $ENV python -W ignore -m diagnostics.finder_vs_ml data/archive_yaml ~/Documents/vfe_peak_ml/lvr-tester --data data/real_shots_p0finder.npz --pred outputs/real/merge/u32l4/predictions.npz --out outputs/real/finder_vs_ml_merge --min-conf 0.5 > outputs/real/finder_vs_ml_merge/full_run.log 2>&1
```

### 21. All held-out shots of one test type in the GUI
```bash
micromamba run -n $ENV python -W ignore -m diagnostics.make_replay_run data/archive_yaml ~/Documents/vfe_peak_ml/lvr-tester --table outputs/real/finder_vs_ml_merge/shots.csv --test-type incoming_A --out ~/Documents/vfe_peak_ml/replay/all_heldout_incoming_A_merge
~/Documents/vfe_peak_ml/run_lvr_tester.sh --selftest --replay ~/Documents/vfe_peak_ml/replay/all_heldout_incoming_A_merge; tail -2 ~/Documents/vfe_peak_ml/lvr_tester_gui.log; ~/Documents/vfe_peak_ml/run_lvr_tester.sh --replay ~/Documents/vfe_peak_ml/replay/all_heldout_incoming_A_merge
```
