# Experiment Log

One entry per meaningful run. Format:

```
## YYYY-MM-DD — short title
- Commit: <git sha>
- Config: <relevant config/*.yaml values or diff>
- Result: <what happened, key numbers>
- Next: <what this implies for the next step>
```

Entries start once there's a result worth recording (Phase 2 onward).

## 2026-08-13 — full-panel snapshot differencing
- Commit: c4308e4
- Config: R207 panel (502 genomes, phylum-stratified), dark-at-T0 = no Pfam-35 GA hit, characterised-T1-proxy = Pfam-37 hit in a family new since 35.0
- Result: 1,341,100 proteins scanned; 297,798 dark-at-T0 (22.21%); 51,287 characterised-T1-proxy (3.82%); 4,138 positive-proxy (0.309%) -- 41-83x the 50-100 go/no-go floor
- Next: Report to Track 1 for formal go/no-go sign-off; proxy uses Pfam-37 net-new-family, not true InterPro-latest per P1-D3 -- real number should only be larger

## 2026-08-13 — esm2 embedding separation check (genos-m blocked on this gpu)
- Commit: c4308e4
- Config: esm2_t33_650M, mean-pooled; unambiguous single-Pfam-35-hit proteins, 5/family; pilot=1 genome/6 families/24 seqs, full=30 phylum-diverse genomes/20 families/100 seqs
- Result: pilot: within-family cosine 0.972 vs across-family 0.932, gap 0.040. full: within 0.963 vs across 0.928, gap 0.036 -- consistent, modest but real separation
- Next: Genos-m side still pending Angshuman on his M1 Pro; re-run this comparison once his result lands to check separation is comparable across models

## 2026-08-13 — stage 3 embedding sample (esm2, R207 panel)
- Commit: 2ea3410
- Config: esm2, 64000 sequences: {'dark_negative': 30000, 'characterised_at_t0': 29862, 'positive': 4138}
- Result: embedded in 3592s, shape (64000, 1280)
- Next: feed into the Layer 1 novelty scorer once Track 1 fixes EVT vs density-based

## 2026-08-15 — full-panel nucleotide extraction (genos-m input)
- Commit: 9a3e937
- Config: R207 panel (502 genomes), gtdb_proteins_nt_reps streamed directly from GTDB public mirror (61GB archive), no AWS credentials needed
- Result: 502/502 genomes extracted, 1.4GB total, gene counts verified matching the protein version exactly (e.g. 219/219) -- same genes, DNA modality
- Next: Angshuman can now run Genos-m against real, correctly-matched panel data on his M1 Pro

## 2026-08-20 — layer 1 evt novelty scorer (esm2, R207 panel)
- Commit: c235131
- Config: k=10, gpd_threshold_quantile=0.9 (PROVISIONAL, not locked by Track 1)
- Result: 29862 reference, 34138 queries scored (4138 true positives); gpd u=0.0297 shape=0.0966; mean novelty positive=0.1854 vs still-dark=0.1631
- Next: Precision@K, held-out-family AUROC, and calibration check still need Track 1 to fix K and the held-out family before real evaluation numbers can be reported

## 2026-08-20 — layer 1 evaluation (esm2, R207 panel) -- Precision@K, held-out-family AUROC, calibration
- Commit: c235131
- Config: k=10, gpd_threshold_quantile=0.9, held_out_family=PF00528.25 (145 members), precision_k=[50, 100, 500, 1000, 5000] (ALL PROVISIONAL, not locked by Track 1)
- Result: Precision@K: {50: 0.0, 100: 0.01, 500: 0.028, 1000: 0.056, 5000: 0.1312} (baseline rate 0.1212); held-out-family AUROC=0.8980; figures in E:\dark_matter\data\processed\gtdb_R207\figures
- Next: waiting on Track 1 to lock k, GPD threshold, K, and the held-out family choice before these numbers are reportable as final

## 2026-08-20 — scorer evaluation -- precision@k (esm2)
- Commit: 216df0b
- Config: k=5, threshold_quantile=0.9, K=[50, 100, 500, 1000] (frozen, config/scorer.yaml)
- Result: P@K={50: 0.0, 100: 0.0, 500: 0.028, 1000: 0.06}, lift={50: 0.0, 100: 0.0, 500: 0.2309966167230546, 1000: 0.49499275012083127}; baseline P@K={50: 0.0, 100: 0.0, 500: 0.028, 1000: 0.06}, baseline lift={50: 0.0, 100: 0.0, 500: 0.2309966167230546, 1000: 0.49499275012083127}; set_positive_rate=0.1212; gpd ks_pvalue=0.9837
- Next: held-out-family AUROC + calibration via scripts/heldout_family_eval.py

## 2026-08-20 — held-out-family evaluation (esm2) -- AUROC distribution + calibration
- Commit: 216df0b
- Config: min_members=5, max_families=5, seed=42 (frozen, config/scorer.yaml)
- Result: 5 families: median AUROC=0.9792, mean=0.9553; calibration in E:\dark_matter\results\heldout_esm2_calibration.csv
- Next: report to Track 1 for interpretation against the acceptance sanity checks in docs/Track2_Phase2_scoring_handoff.md

## 2026-08-20 — held-out-family evaluation (esm2) -- AUROC distribution + calibration
- Commit: 8dabd13
- Config: min_members=5, max_families=500, seed=42 (frozen, config/scorer.yaml)
- Result: 500 families: median AUROC=0.9057, mean=0.8712; calibration in E:\dark_matter\results\heldout_esm2_calibration.csv
- Next: report to Track 1 for interpretation against the acceptance sanity checks in docs/Track2_Phase2_scoring_handoff.md

## 2026-08-22 — genos-m embedding separation (30 genomes, 20 families)
- Commit: c61f267
- Config: genos-m, mean-pooled, unambiguous single-Pfam-35-hit proteins, 5/family, seed=42
- Result: 100 sequences, 20 families: within-family cosine 0.997 vs across-family 0.993, gap 0.004
- Next: compare against the other model's separation once both are available

## 2026-08-23 — scorer evaluation -- precision@k (esm2)
- Commit: c42ef10
- Config: k=5, threshold_quantile=0.9, K=[50, 100, 500, 1000] (frozen, config/scorer.yaml)
- Result: P@K={50: 0.0, 100: 0.0, 500: 0.028, 1000: 0.06}, lift={50: 0.0, 100: 0.0, 500: 0.2309966167230546, 1000: 0.49499275012083127}; baseline P@K={50: 0.0, 100: 0.0, 500: 0.028, 1000: 0.06}, baseline lift={50: 0.0, 100: 0.0, 500: 0.2309966167230546, 1000: 0.49499275012083127}; set_positive_rate=0.1212; gpd ks_pvalue=0.9837
- Next: held-out-family AUROC + calibration via scripts/heldout_family_eval.py

## 2026-09-07 — immune layer-3 sweep + full-scale dark-query flagging (esm2, layer 22)
- Commit: 1710a5a
- Config: n_detectors sweep [1000, 5000, 10000, 20000], margin sweep [0.5, 1.0, 1.5, 2.0], pca sweep [0, 50, 100], alpha sweep [np.float64(0.01), np.float64(0.05), np.float64(0.1)], base config from E:\dark_matter\config\immune.yaml
- Result: detector-count AUROC [0.5919805228876249, 0.7365232906269299, 0.8322199805025746, 0.862368882894188]; margin AUROC [0.6269443604042476, 0.628536708964332, 0.6273491091682547, 0.6284035696557208]; pca AUROC [0.5325393100617686, 0.628536708964332, 0.6169832900230413]; full-scale (14138 queries) flag rate dark=0.1195 positive=0.2008 dark_negative=0.0858; convergence with Layer-1 rho=0.485, top-K jaccard={'50': 0.0, '100': 0.005025125628140704, '500': 0.059322033898305086, '1000': 0.11982082866741321}
- Next: report to Track 1 against P3-D6's acceptance sanity + fold into manuscript (Phase 4)

## 2026-09-10 — immune layer-3 detector-count saturation sweep (esm2, layer 22)
- Commit: 887e9c6
- Config: n_detectors [20000, 30000, 40000, 50000] (extends P3-D7's [1000,5000,10000,20000]), same 30 held-out families, seed, PCA-50, box sampling; config/immune.yaml frozen default (5000) unchanged
- Result: AUROC mean 0.862(20k, matches P3-D7 exactly) -> 0.889(30k) -> 0.890(40k) -> 0.899(50k); marginal gain collapses from +0.030 (10k->20k) to +0.027 (20k->30k) to +0.001 (30k->40k) -- plateaus around 30,000 detectors, ~0.89 AUROC, still short of Layer 1's 0.962
- Next: report saturation point to Track 1 for a reviewed decision on whether to update the frozen n_detectors default (P2-D5/P3-D5 anti-fishing rule -- not changed unilaterally here)

## 2026-09-14 -- full 903-family genos-m eval + genos-m as 4th convergence axis (cloud, g5.xlarge)
- Commit: pending
- Config: layers 12,9; O(N^2)-fixed reembed_eval.py; embed_genos_m_dark_queries.py on all 34,138 dark queries; run_convergence.py --extra-axis composition_novelty.csv --extra-axis genos-m_novelty_scores.csv
- Result: full-903-family AUROC layer 9 raw=0.665 mean/0.703 median (vs 300-family-capped 0.739/0.795); 4-axis convergence on 31,712 queries -- genos-m~evt rho=0.15, genos-m~context rho=-0.05, genos-m~composition rho=-0.32; genos-m-top-10% positive-rate lift=1.30x (the only axis that enriches, likely leakage-driven per P1-D7/P2-D7); 4-way convergent set n=35, lift=0.94x
- Next: report to Track 1 -- the leakage caveat on the genos-m lift needs to land in the manuscript alongside the number, not just the number; fold corrected full-scale genos-m AUROC into any place the 300-family estimate was cited

## 2026-09-14 -- true-intergenic layer 5 controls + standalone benchmark release
- Commit: pending
- Config: 80-genome/12GB/30min bounded stream of GTDB R207 genome-assembly archive (65GB), intergenic segments >=60bp between Prodigal gene coordinates; benchmark_release/ packaged from genome_panel.csv + panel_protein_labels.csv
- Result: 44 genomes (12GB byte cap bound first), 81286 intergenic segments; real CPBB median 0.0721 (matches P4-D5) vs true-intergenic 0.0308 (vs shuffle/markov1 0.011); coding-vs-true-intergenic AUROC=0.7816 (vs 0.94 against artificial nulls -- a harder, more honest control); one mid-run network interruption (IncompleteRead at 2.26GB) recovered via retry-wrapper on attempt 1
- Next: fold into manuscript Layer 5 section + limitations; benchmark_release/ needs no further action, packaging complete

## 2026-09-14 -- prostt5 full 300-family scale + 5th convergence axis (local, 8GB gpu, checkpointed)
- Commit: pending
- Config: run_layer2_prostt5.py --eval-only --score-queries --batch-size 2, 300 families / 6907 reference proteins (unchanged from P4-D6), 34138 dark queries, length-sorted chunking (CH=256) with per-chunk checkpointing + OOM backoff (halves batch_size on CUDA OOM, up to 4 attempts)
- Result: full-300-family AUROC raw=0.9332 mean/0.9454 median, centered=0.9437/0.9599 (300 families) -- confirms P4-D6's 60-family estimate direction; ProstT5-as-5th-axis convergence on 31712 queries -- ProstT5~composition rho=0.64 (highest pairwise value, i.e. NOT independent), ProstT5~genos-m rho=-0.43, ProstT5-top-10% lift=0.28x (depletes, same direction as composition); 5-way convergent set n=20, lift=1.24x (vs 4-way n=35, lift=0.94x from P4-D8)
- Next: report to Track 1 -- ProstT5 axis is measured but not promoted over the 3-axis headline (P4-D1/D2/D4) since it is not independent of composition; a genuinely independent structural axis (contact-map or secondary-structure-derived, not mean-pooled kNN) is future work

## 2026-09-14 — full-panel 36->37 boundary labeling (leakage-tightened, Genos-m axis follow-up)
- Commit: f6f4204
- Config: pfam-36 GA dark-at-36 / pfam-37 net-new-family characterised-since-36, batch-size=25, full 502-genome panel
- Result: 1341100 proteins; dark-at-36 292995 (21.85%); characterised-37-since-36 31308 (2.33%); positive-36-37 2448 (0.183%)
- Next: cross-reference against genos-m_novelty_scores.csv to recompute the convergence-axis lift on this leakage-tightened positive set

## 2026-09-14 -- genos-m axis lift re-test under leakage-tightened boundary (P4-D11)
- Commit: pending
- Config: scripts/genosm_leakage_relift.py, top-decile (quantile=0.90) on genos_m_novelty, same 34,138-query set as P4-D8, positive label swapped from panel_protein_labels.csv (35->37) to panel_protein_labels_36_37.csv (36->37)
- Result: 2385/34138 queries positive under 36-37 (vs 4138 under 35-37); main-boundary lift recomputed 1.32x (matches P4-D8's reported 1.30x); leakage-tightened lift 1.26x -- only a ~5% relative drop despite the positive set shrinking 43%; all 2385 leakage-tightened positives are a subset of the main-boundary positives (clean consistency check)
- Next: the enrichment surviving this tightening weakens (does not confirm) the pure-leakage explanation for the genos-m axis inversion -- update manuscript's Layer 4/Discussion framing from "plausibly leakage" to "leading, not confirmed, hypothesis"

## 2026-10-14 -- Review II re-analysis (docs/reviewer_response.md)
- Commit: see git log (scripts/reviewer_*.py, label_audit.py, layer_sweep_esm2.py, layer_selection_validation.py, embed_full_query_reference.py)
- Config: ESM-2 650M fp32 weights with TF32 matmuls on the RTX 4060; all 34 layers for the 11,527 held-out evaluation proteins; layers 22 and 33 for the remaining 34,502 sample proteins; Pfam 35 and 37 rescans at E <= 1e-3
- Result: layers 22 and 33 match the saved embeddings (cosine >= 0.99994); layer 22 = 0.962, plateau 18-28 (0.959-0.964), layer picked on development families scores 0.9638 on test families; on the common 34,138-query set the dark-query AUROC is 0.470 (layer 22) and 0.568 (layer 33), protein length alone 0.618; 3-way convergent set is 1.4x its correlated null (not 3.4x) and not enriched; length-matched Layer 5 AUROC vs intergenic 0.91; 15.1% of still-dark and 23.3% of positives have a sub-threshold Pfam 35 hit
- Next: nothing blocking; the Review II report (v3) and the paper's wording on Precision@K may cite these numbers

## 2026-10-26 -- baselines, independent test set and structure case studies (TCBB revision)
- Commit: see git log (scripts/baseline_*.py, build_independent_set.py, independent_eval.py, case_studies.py, top_ranked_composition.py)
- Config: frozen scorer (ESM-2 layer 22, k=5, 90th-percentile GPD threshold); phmmer and Pfam 35.0 hmmsearch via pyhmmer; 60 GTDB R232 genomes of species absent from R207 (CheckM2 >= 90/<= 5, <= 2 per phylum, seed 42), genes called with pyrodigal; ESMFold v1 from the safetensors conversion, Foldseek web server (afdb50, pdb100, 3Di+AA)
- Result: withheld-family AUROC knn5 0.962, nearest neighbour 0.977, phmmer 0.969, Mahalanobis 0.565, Pfam profile 1.000 (by construction); conformal and GPD p-values equally calibrated at alpha 0.001-0.2; dark-query AUROC phmmer 0.381, Pfam margin 0.458, length 0.618. Independent set: 178,954 proteins, 574 positives; calibration at alpha 0.05 is 0.054 overall but 0.005 / 0.100 / 0.324 for families with >=5 / 1-4 / 0 reference members; withheld-family AUROC 0.867 over 740 families; retrospective AUROC 0.439 (layer 22), 0.567 (layer 33), length 0.590. Top three still-dark proteins fold with pLDDT 26-40 and have no Foldseek hit; the control (PF02545) folds at pLDDT 96.9 with Maf-family hits; the top 1% of the ranking has lower composition entropy (3.78 vs 3.96 bits) and more disorder-promoting residues
- Next: compile the TCBB draft on Overleaf, check the 12-page limit, Zenodo DOI, biographies

## 2026-10-27 -- reviewer-driven revision analyses (track-2-research-extension)
- Commit: see git log on track-2-research-extension (scripts/revision_scoring.py, revision_stats.py, revision_normalised_dark.py, panel_representativeness.py, fetch_external_hmms.py, cross_database_scan.py, top_ranked_checks.py, case_studies.py)
- Config: frozen layer-22 embeddings; density-normalised score nu(x) = d5(x) / mean LOO d5 of the 5 nearest reference proteins, same GPD tail; random / genome-disjoint / phylum-disjoint calibration splits (20 each); BH over 78 tests; convergence nulls permutation, Gaussian copula and t4 copula (Sigma = sin(pi tau / 2)); pyhmmer scans of all 34,138 dark queries against NCBIfam (TC), AntiFam (GA) and NMPFamsDB (E <= 1e-5); filter thresholds from 3,000 characterised proteins; ESMFold + Foldseek on the filtered top six and a mid-ranked comparison set (ranks 15,000-20,000, seed 42)
- Result: distance score at alpha 0.05 flags 0.027 / 0.21 of families with >=5 / 1-4 reference members under every split; the normalised score flags 0.05 / 0.01 and on the independent set 0.038 / 0.016 / 0.109 (>=5 / 1-4 / none) against 0.005 / 0.100 / 0.324, at withheld-family AUROC 0.942 vs 0.962; phmmer 0.969, rank average of phmmer and kNN 0.984; 46.9% of dark queries have no phmmer hit; no convergent intersection enriched after BH; NCBIfam matches 5.4% of dark queries (22.1% positives vs 3.1% still-dark), NMPFamsDB 4.0%, AntiFam 0.13%; the filter keeps 47.7% of dark queries and 28.4% of the top 1%; filtered top six fold at pLDDT 28.5-37.7, five with no AFDB hit, one (DLYX01000357.1_5) with weak transglutaminase-like AFDB hits (E 0.02, 12% identity); six random still-dark proteins from ranks 15,000-20,000 fold at pLDDT 41-96 (median 58 vs 33) and all have AFDB hits (five at E < 1e-3), exact Mann-Whitney p = 0.002
- Next: compile on Overleaf and check the length
