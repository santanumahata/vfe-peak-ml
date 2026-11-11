# VFE-board QC peak finding (ML) — Quick Reference

**Detail:** `RUNBOOK_vfe_peak_ml-DETAIL.md` — expected output, gotchas, conventions, changelog.
**Where it runs:** this machine, `$ENV` micromamba env, RTX 4070 GPU; every command from the repo root.

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
micromamba run -n $ENV python -m src.report_real outputs/real/<date>_ml_vs_finder.pdf outputs/real/sweep/u32l4
```

### 9. Full write-up (LaTeX note)
```bash
micromamba run -n $ENV python -m writeup.make_figures && cd writeup/src && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../outputs vfe_peak_ml_writeup.tex && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../outputs vfe_peak_ml_writeup.tex
```
