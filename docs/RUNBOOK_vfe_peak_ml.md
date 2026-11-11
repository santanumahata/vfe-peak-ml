# VFE-board QC peak finding (ML) — Quick Reference

**Detail:** `RUNBOOK_vfe_peak_ml-DETAIL.md` — expected output, gotchas, conventions, changelog.
**Where it runs:** this machine, `$ENV` micromamba env, RTX 4070 GPU; every command from the repo root.

### 1. Environment
```bash
micromamba run -n $ENV pip install --force-reinstall --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
micromamba install -n $ENV -c conda-forge pyyaml -y
```

### 2. Generator self-check (toys)
```bash
micromamba run -n $ENV python -m src.toy_waveforms
```

### 3. Train the toy CNN
```bash
micromamba run -n $ENV python -m src.train
```

### 4. Hybrid stage 2 (toys, not the default)
```bash
micromamba run -n $ENV python -m src.hybrid
```

### 5. Decoder ablation (toys)
```bash
micromamba run -n $ENV python -m diagnostics.hybrid_ablation
```

### 6. Apex snap tuning (toys)
```bash
micromamba run -n $ENV python -m diagnostics.refine_sweep && micromamba run -n $ENV python -m diagnostics.refine_gated
```

### 7. Apex precision checks (toys)
```bash
micromamba run -n $ENV python -m diagnostics.apex_ambiguity && micromamba run -n $ENV python -m diagnostics.decode_compare outputs
```

### 8. Toy results PDF
```bash
micromamba run -n $ENV python -m src.report
```

### 9. Slide figures
```bash
micromamba run -n $ENV python talk/make_plots.py
```

### 10. Lab deck
```bash
cd talk/src && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../outputs vfe_peak_ml.tex && pdflatex -interaction=nonstopmode -output-directory=../outputs vfe_peak_ml.tex
```

### 11. Rebuild order after a generator change
```bash
micromamba run -n $ENV python -m src.train && micromamba run -n $ENV python -m src.hybrid && micromamba run -n $ENV python -m diagnostics.hybrid_ablation && micromamba run -n $ENV python -m diagnostics.refine_sweep && micromamba run -n $ENV python -m diagnostics.refine_gated && micromamba run -n $ENV python -m diagnostics.apex_ambiguity && micromamba run -n $ENV python talk/make_plots.py
```

### 12. List and unpack the archive's metadata (real)
```bash
tar -tf ~/Downloads/results.tar > <members.txt> && mkdir <yaml_dir> && tar -xf ~/Downloads/results.tar -C <yaml_dir> --no-same-owner --no-same-permissions --wildcards '*.yaml'
```

### 13. First look at real traces
```bash
micromamba run -n $ENV python -m diagnostics.real_first_look <yaml_dir> ~/Downloads/results.tar <extract_dir> diagnostics/<date>_real_first_look.png <members.txt>
```

### 14. Build the real dataset
```bash
micromamba run -n $ENV python -m src.real_data <yaml_dir> ~/Downloads/results.tar <members.txt> data
```

### 15. Train on real traces
```bash
micromamba run -n $ENV python -m src.train_real --arch unet --out outputs/real/unet
```

### 16. Compare model sizes
```bash
for cfg in "u16 16 3" "u32 32 3" "u64 64 3" "u32l4 32 4" "u32l5 32 5"; do set -- $cfg; micromamba run -n $ENV python -m src.train_real --arch unet --base $2 --levels $3 --out outputs/real/sweep/$1; done && for cfg in "t256x4 4" "t256x8 8"; do set -- $cfg; micromamba run -n $ENV python -m src.train_real --arch transformer --d 256 --tdepth $2 --out outputs/real/sweep/$1; done
for s in 1 2; do for cfg in "<tag> <base> <levels>"; do set -- $cfg; micromamba run -n $ENV python -m src.train_real --arch unet --base $2 --levels $3 --seed $s --out outputs/real/sweep_seed/$1_s$s; done; done
micromamba run -n $ENV python -m diagnostics.real_sweep_summary
```

### 17. Failure shots
```bash
micromamba run -n $ENV python -m diagnostics.real_hard_cases outputs/real/unet outputs/real/transformer diagnostics/<date>_real_hard_cases.png
```

### 18. ML vs finder PDF
```bash
micromamba run -n $ENV python -m src.report_real outputs/real/<date>_ml_vs_finder.pdf
```
