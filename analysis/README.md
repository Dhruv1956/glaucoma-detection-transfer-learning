# analysis/

- `analyze_final.py` : recomputes the final numbers from the per-fold held-out predictions of the five arms (pooled ROC-AUC and PR-AUC, group bootstrap, paired differences, per-source results). Set `RES_DIR` (folder with the five unzipped arm folders) and `OUT_DIR`.
- `make_figures.py` : draws `figures/fig_roc` and `figures/fig_source` from `analysis_out/` produced by the script above.
- `final_results_10x5.md`, `final_numbers.json` : its output, i.e. the numbers reported in the README and the paper.

The per-image predictions that these scripts read are **not included** (dataset licences), so the two scripts can only be re-run after reproducing the cross-validation runs.
