# VFE-board QC peak finding (ML) — Runbook (detail), steps 25–31: review answers, retraining, review note

**Main detail:** `RUNBOOK_vfe_peak_ml-DETAIL.md` — overview, conventions, gotchas, changelog.
**Card:** `RUNBOOK_vfe_peak_ml.md` — commands only.

All steps work on the review folder of step 23, `~/Documents/vfe_peak_ml/replay/crossfit_disputed`.

### 25. Which disputed peaks hold a decision
**Known-good command:**
```bash
micromamba run -n $ENV python -m diagnostics.same_candidate ~/Documents/vfe_peak_ml/replay/crossfit_disputed ~/Documents/vfe_peak_ml/lvr-tester
```
**Expect:** a few seconds; read-only on the replay folder and on `review.csv`. Each disputed peak
gets a class: `same` (finder and network time snap to one candidate peak under the GUI's rule,
so the pick cannot change the label), `different` (two candidates: a real decision), `refused`
(one side has no candidate within 0.1 ms). The counts are printed per difference band and per
landmark, then `outputs/real/crossfit_vs_finder/same_candidate.csv` and
`same_candidate_summary.json`, each with a `.meta`.

### 26. Answer the disputes that need no eye
**Known-good command:**
```bash
micromamba run -n $ENV python -m src.answer_same_candidate ~/Documents/vfe_peak_ml/replay/crossfit_disputed ~/Documents/vfe_peak_ml/lvr-tester
micromamba run -n $ENV python -m src.answer_same_candidate ~/Documents/vfe_peak_ml/replay/crossfit_disputed ~/Documents/vfe_peak_ml/lvr-tester --write
micromamba run -n $ENV python -m src.answer_by_rule ~/Documents/vfe_peak_ml/replay/crossfit_disputed ~/Documents/vfe_peak_ml/lvr-tester
micromamba run -n $ENV python -m src.answer_by_rule ~/Documents/vfe_peak_ml/replay/crossfit_disputed ~/Documents/vfe_peak_ml/lvr-tester --write
```
Run in this order: the rule only answers peaks still open after the first script.
**Expect:** without `--write` each script prints its report and writes nothing. With `--write`,
`review.csv` is first copied to `review.csv.before_auto_<date>` / `review.csv.before_rule_<date>`
(the script stops if that copy exists), then the new rows go in with operator
`auto_same_candidate` / `auto_rule_tallest`. `answer_by_rule` also writes `rule_answers.csv`
(one line per open peak: candidates, heights above baseline, outcome) and prints counts of
kept answers, replaced raw clicks, rule → finder, rule → network, and no sizable P0. A shot
with no sizable P0 gets no row, so step 27 drops it. Rows of `agree` are untouched; the GUI
self-test of step 23 still passes on the folder.

### 27. Review answers into labels
**Known-good command:**
```bash
micromamba run -n $ENV python -m src.relabel_review ~/Documents/vfe_peak_ml/replay/crossfit_disputed ~/Documents/vfe_peak_ml/lvr-tester
micromamba run -n $ENV python -m src.relabel_review ~/Documents/vfe_peak_ml/replay/crossfit_disputed ~/Documents/vfe_peak_ml/lvr-tester --write
```
**Expect:** the report (answers by choice, operator and landmark; open peaks; shots dropped;
labels moved by more than 0.02 and 0.1 ms; split by train / val / hard) is printed, and with
`--write` written to `data/real_shots_review.npz` with a `.meta`. The script does not save the
report; the copy in `outputs/real/relabel_review_report.txt` was saved separately. Input `data/real_shots_p0finder.npz` is never
changed; old labels are kept as `P_finder`. The script stops if the output exists, if an answer
is not on a candidate peak, if the review does not belong to the dataset, or if a relabelled
shot's five peaks are out of time order.

### 28. Retrain on the review labels
**Known-good command:**
```bash
mkdir -p outputs/real/review && micromamba run -n $ENV python -m src.train_real --arch unet --base 32 --levels 4 --data data/real_shots_review.npz --out outputs/real/review/u32l4 > outputs/real/review_u32l4.log 2>&1
```
**Expect:** as step 19, about 6 min on the laptop GPU; the log ends with the metrics block
(`"n_train_synthetic": 0`, `"val_synthetic": null`). Same model and seed as
`outputs/real/p0finder/u32l4`, so the only change is the labels. The split is the same runs;
dropped shots leave training, validation and failure counts lower by those shots.

### 29. Finder, old and retrained model against the review reference
**Known-good command:**
```bash
micromamba run -n $ENV python -m diagnostics.score_vs_review
```
**Expect:** a few seconds; per landmark and for all five, the share within 0.02 ms and within
0.1 ms for the finder (`P_finder`), the old model and the retrained model, on validation and
failure runs only, on all peaks and on the peaks the review changed. Written to
`outputs/real/review/score_vs_review.json` with a `.meta`. Network times are argmax bins, so
they lie on the 10 µs grid; the 0.02 ms column is therefore two bins wide.

### 30. Only the shots that hold a decision, in the GUI
**Known-good command:**
```bash
micromamba run -n $ENV python -W ignore -m diagnostics.make_replay_run data/archive_yaml ~/Documents/vfe_peak_ml/lvr-tester --table outputs/real/crossfit_vs_finder/review_queue_decisions.csv --all-runs --keep-order --out ~/Documents/vfe_peak_ml/replay/crossfit_decisions > outputs/real/crossfit_vs_finder/replay_decisions_build.log 2>&1
~/Documents/vfe_peak_ml/run_lvr_tester.sh --selftest --replay ~/Documents/vfe_peak_ml/replay/crossfit_decisions; tail -2 ~/Documents/vfe_peak_ml/lvr_tester_gui.log; ~/Documents/vfe_peak_ml/run_lvr_tester.sh --replay ~/Documents/vfe_peak_ml/replay/crossfit_decisions
```
`review_queue_decisions.csv` = the rows of `review_queue.csv` whose `idx` has at least one peak
not classed `same` in step 25, in queue order. It and the copy of the existing answers into the
new folder's `review.csv` (rows mapped by `idx` through both `replay_index.csv` files, written
with `review_strip.write_review`) were made by one-off snippets, not kept as scripts.
**Expect:** as step 23: one `shots/*.npz` per queue row, `replay OK` and `replay table OK` from
the self-test. The window opens with the answers of steps 14 and 26 already in place; any change
made there lands in `crossfit_decisions/campaign_replay/review.csv`, not in the step-23 folder.

### 31. Review note for my guide
**Known-good command:**
```bash
micromamba run -n $ENV python -m writeup.make_review_figures
micromamba run -n $ENV python -m writeup.make_training_figures
cd writeup/src && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../outputs review_labels_note.tex && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../outputs review_labels_note.tex
```
Needs steps 25–29. Source `writeup/src/review_labels_note.tex`; diagrams `writeup/src/diagrams/
{training_pipeline,unet,split}.tex` (TikZ, `\input`); code listings read the scripts directly
(`\lstinputlisting`), including the tester's `review_strip.py`.
**Expect:** the first script writes `writeup/plots/review/{examples.png, p0_heights.png,
numbers.tex}`, the second `heatmaps.png` and `training_curves.png` (first validation shot of the
retrained model; loss and validation curves of old and retrained model). The PDF is
`writeup/outputs/review_labels_note.pdf`, 22 pages, with no `Overfull` or `undefined` lines in
the log; every page's footer names the unsigned algorithm steps.
