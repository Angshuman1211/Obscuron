# Response to the Review II comments

An external evaluator read the Interim Report (Review II) and sent eight comments. This file says what each comment found, what was run, and what the report should now say. Every number below comes from a script in `scripts/` and a file in `results/reviewer/`; nothing was tuned to improve a result.

## What went wrong in the report

Two things caused most of the confusion, and the first one was our mistake.

1. **The report mixed two analysis setups.** The AUROC of 0.962 is layer 22, on the 15,360 single-hit references and 903 held-out families. The GPD fit (xi = 0.0998, KS p = 0.984) and the Precision@K lifts (0.23x at K=500, 0.50x at K=1000) are from the earlier last-layer run: layer 33, 29,862 references, 34,138 queries. Sentences in the report read as if they described one system. The Layer 3 flags use a third setup (layer 22, 14,138 queries).
2. **The held-out-family AUROC and Precision@K measure different things**, and the report did not say how far apart they are. The new analysis (comment 3) puts a number on it.

All of this is now fixed by scoring layers 22 and 33 on one query set and one reference (`scripts/reviewer_common_set.py`).

## Comment 1: dataset and sample counts

| Set | Proteins | Positives | Positive share | Used for |
|---|---|---|---|---|
| Panel (502 GTDB R207 genomes) | 1,341,100 | 4,138 | 0.31% | labelling |
| Dark at T0 (no Pfam 35 hit at GA) | 297,798 (22.21%) | 4,138 | 1.39% | the population the method is aimed at |
| Characterised at T0 | 1,043,302 | 0 | 0 | source of the reference |
| Embedding sample | 64,000 | 4,138 | 6.5% | all embedded proteins |
| Dark-query set | 34,138 | 4,138 | 12.12% | last-layer scorer, Layer 2/4/5 signals, Precision@K |
| Layer-22 query subset | 14,138 | 4,138 | 29.27% | Layer 3 flags, earlier layer-22 Precision@K |
| Reference, last-layer scorer | 29,862 | 0 | 0 | kNN reference and GPD fit |
| Reference, held-out-family test | 15,360 (single Pfam hit) in 3,109 families | 0 | 0 | AUROC, calibration |
| Held-out-family evaluation | 11,527 in 903 families of 5 or more members | 0 | 0 | AUROC 0.962 |
| Layer 4 | 31,712 | 3,843 | 12.12% | queries with all five signals |
| Layer 5 | 34,138 genes, 81,286 intergenic segments from 44 genomes | 4,138 | 12.12% | coding-structure tests |

The dark-query set is 30,000 dark proteins drawn at random (seed 42) from the 293,660 that never gain a family, plus all 4,138 positives. It is 11.5% of dark proteins, and its positive share (12.12%) is about nine times the natural share (1.39%). The 14,138 subset keeps every positive and 10,000 of the 30,000 negatives, so its share is higher still (29.27%). Precision@K lifts are only comparable within a set, which is why the report's 0.50x (34,138 set) and the paper's 0.98x (14,138 set) look different.

The "34,138 versus 14,138" in the comment is this: 34,138 is the dark-query set, 14,138 is the subset that was re-embedded at layer 22 first.

## Comment 2: how retrospective labels are made, and how label errors are handled

Rules (`src/darkmatter/data/hmmscan.py`, `pfam_diff.py`):
- Dark at T0: no hit against Pfam 35.0 at each family's own gathering (GA) threshold, using HMMER (pyhmmer).
- Positive: dark at T0 and a GA hit to a Pfam 37.0 family whose accession does not exist in Pfam 35.0 (2,383 such families). Characterised at T0: at least one Pfam 35 GA hit.
- Still dark (negative): dark at T0 and no such hit.

A hard threshold produces borderline cases, so `scripts/label_audit.py` rescans all 34,138 queries and 3,000 characterised controls against both libraries without the GA cut-off (E <= 1e-3) and records each protein's best bit-score margin over GA.

| Check | Result |
|---|---|
| Control: characterised proteins with a Pfam 35 hit at or above GA | 100% (3,000 of 3,000) |
| Labelled dark proteins with a Pfam 35 hit at or above GA | 0% (positives and still-dark) |
| Still-dark proteins with a sub-threshold Pfam 35 hit (E <= 1e-3, below GA) | 15.1% |
| Positives with a sub-threshold Pfam 35 hit | 23.3% (21.6% within 10 bits of GA) |
| Positives: Pfam 37 new-family score above GA | median 48.8 bits; 11.0% under 5 bits, 18.9% under 10 bits |
| Still-dark proteins with a sub-threshold hit to a new Pfam 37 family (unlabelled-positive risk) | 0.9% (0.7% within 10 bits) |
| Positives in a clan that already existed in Pfam 35 | 71.3% (543 distinct new families; 25.8% are DUF or UPF) |

How errors are handled, in plain terms:
- **Dark-side errors** (a "dark" protein that has a weak known relative): about 15% to 23%. Positives are more often borderline than still-dark proteins. This is the near-known bias in numbers, and it is not removed by the labels, only measured.
- **Missed positives** (a still-dark protein that is really characterised later): under 1% by this measure. The label is a Pfam proxy, so proteins characterised through other InterPro databases are also in the negative class. The positive count of 4,138 is a floor.
- **Weak positives** (labelled by a hit just above GA): 19% are within 10 bits.
- **Robustness**: `scripts/reviewer_label_robustness.py` redoes the dark-query results on a stringent label set (2,691 positives at least 10 bits above GA and with no Pfam 35 hit within 10 bits; 25,814 negatives with nothing within 10 bits). Layer 22 AUROC goes from 0.495 to 0.501 and layer 33 from 0.593 to 0.600, so label noise does not change any conclusion.
- Pfam 35 to 37 is also checked on a shorter boundary (Pfam 36 to 37, 2,448 positives; P1-D14 and P4-D11).

## Comment 3: AUROC 0.962 against Precision@K lift 0.50

These are two different tests.

- **Held-out-family AUROC (0.962)**: hide one Pfam family from the reference, then ask whether its members score higher than ordinary known proteins. The positives are real homologues at moderate distance, the negatives are ordinary known proteins.
- **Precision@K and the query-task AUROC**: among dark proteins only, do the ones that later gain a family rank above the ones that stay dark? Both groups are far from the reference. The median kNN distance of a dark query is 0.026 at layer 22, seven times the median reference distance (0.0035), and 33% of dark queries already have p <= 0.05.

Same 34,138 queries and 29,862 references, both layers (`results/reviewer/common_set.json`):

| | Layer 22 | Layer 33 (last) |
|---|---|---|
| Held-out-family AUROC, 903 families (mean / median) | 0.962 / 0.985 | 0.786 / 0.815 |
| Dark-query AUROC, positives vs still dark | 0.470 (0.460 to 0.479) | 0.568 (0.558 to 0.577) |
| Precision@K lift at K = 500 / 1000 / 2000 | 0.16 / 0.55 / 0.92 | 0.23 / 0.48 / 0.83 |
| Protein length alone, AUROC | 0.618 | 0.618 |

On the 14,138 subset the same layer-22 score gives a query-task AUROC of 0.495 and a lift of 0.98 at K = 1000. The decile table shows no trend: the positive rate in the most novel decile (29.4%) equals the base rate (29.3%).

What this means for candidate prioritisation:
- The score separates a hidden family from known proteins very well, and its false-flag rate is calibrated. It does **not** rank which dark proteins will be characterised next. Protein length does that better (AUROC 0.618), because later-characterised proteins are longer (median 209 against 149 amino acids on the 34,138 set; `results/reviewer/length_baseline.json`, AUROC 0.618, interval 0.610 to 0.626, 0.050 above the last-layer score with interval 0.037 to 0.062) and sit near known families.
- Precision of a flagged set depends on prevalence. At alpha = 0.05 the layer-22 test has a true-positive rate of 0.69 and a false-flag rate of 0.048. If 1.4% of the proteins screened are truly novel (the natural share among dark proteins), a flag is correct 17% of the time; at 12% prevalence 67%; at 29% prevalence 86%. At alpha = 0.01 these become 39%, 86% and 95%.
- So the score is for ruling in "far from everything characterised" with a known error rate. It is not a predictor of annotatability, and the report should say so.

## Comment 4: why layer 22, and validation on independent data

How layer 22 was chosen: from a 100-sequence, 20-family diagnostic (P2-D7, P2-D8), using the embedding separation gap (the AUROC on the same small set was computed too and agreed); the full benchmark came later. The report did not say that, and the family set of the diagnostic is a subset of the 903.

New check (`scripts/layer_sweep_esm2.py`, `scripts/layer_selection_validation.py`): all 34 layers embedded for the 11,527 evaluation proteins (layers 22 and 33 match the saved embeddings to cosine 0.99994), then scored with the same held-out-family AUROC.

- Layers 18 to 28 form a plateau between 0.959 and 0.964; layer 22 is 0.962. The last layer (33) is 0.786, layer 32 is 0.908, layer 12 is 0.936.
- Without the 20 diagnostic families (883 left): layer 22 is 0.962, layer 33 is 0.785.
- Development/test protocol, 50 random splits (300 development families, 603 test): the layer picked on the development families is mostly 23 to 25, and its test AUROC is 0.9638, against 0.9617 for fixed layer 22. A clan-blocked split (no Pfam clan on both sides) gives the same: 0.9639 against 0.9620. Fixing layer 22 costs 0.002 AUROC.
- Layer 22 beats layer 33 in 96% of families (mean gain 0.176, bootstrap interval 0.167 to 0.185).

The reason to prefer a middle layer is the usual one: the last layers of a masked language model specialise towards the pretraining objective and lose family-level structure. Note that no test set is claimed here beyond held-out families; the benchmark has no second, independent dataset.

## Comment 5: GPD fitting, threshold, k, goodness of fit, calibration

(`scripts/reviewer_scorer_analysis.py`, layer 22, 11,527 evaluation proteins.)

- **Threshold**: u is the 90th percentile of leave-one-out kNN distances (frozen before results). Thresholds 0.80, 0.85, 0.90, 0.95 give xi = -0.008, 0.002, -0.014, 0.010 and beta = 0.0122, 0.0119, 0.0123, 0.0118. The fit does not move. Mean residual life is flat (0.0118 to 0.0122) from the 80th to the 98th percentile, which is what a GPD predicts.
- **Number of exceedances**: 1,153 at the 90th percentile (2,306 to 577 across the thresholds above). The 95% bootstrap interval of xi is -0.066 to 0.038, so an exponential tail (xi = 0) cannot be ruled out.
- **Goodness of fit**: the KS p-value in the report (0.44, and 0.984 for the last-layer fit) is optimistic because the parameters were estimated from the same data. A parametric bootstrap (500 refits) gives KS p = 0.12 and Anderson-Darling p = 0.14: the fit is not rejected, and not strongly supported either. The report should quote these.
- **k (neighbours)**: k = 5 was fixed before results. Held-out AUROC is 0.977 at k = 1, 0.962 at k = 5, 0.927 at k = 10 and 0.861 at k = 20. Smaller k is better here, so k = 5 is not the optimum.
- **Calibration**: over 20 random calibration/test splits the false-flag rate at alpha = 0.01, 0.02, 0.05, 0.10, 0.20 is 0.0093, 0.0197, 0.0486, 0.0983, 0.1989 (largest gap 0.0017; spread across splits 0.003 to 0.009). A family-clustered bootstrap interval at alpha = 0.05 is 0.036 to 0.058, wider than the naive interval (0.039 to 0.056) because proteins of one family are not independent. Tail p-values below 0.2 are uniform to within KS distance 0.012.
- **Limit of what calibration means**: it is checked on held-out known proteins from the same families as the reference. It does not claim the rate for proteins from unseen phyla.

## Comment 6: strengthening the corroborating layers

(`scripts/reviewer_corroboration_stats.py`; real dark queries, positives against still dark, orientation fixed in advance as higher = more novel; bootstrap intervals; paired bootstrap against the primary score.)

| Signal | Set | AUROC (95% interval) | Reading |
|---|---|---|---|
| ESM-2 last layer (primary on 34,138) | 34,138 | 0.568 (0.559 to 0.577) | weak |
| Genos-m novelty | 34,138 | 0.543 (0.533 to 0.551), 0.025 below primary | weak |
| Genomic context novelty | 34,138 | 0.439 (0.430 to 0.448) | below chance |
| Composition novelty | 34,138 | 0.334 (0.326 to 0.342) | strongly below chance |
| ProstT5 structural novelty | 34,138 | 0.351 (0.343 to 0.360) | strongly below chance |
| ESM-2 layer 22 (primary on 14,138), raw kNN distance | 14,138 | 0.495 (0.484 to 0.504); the tied tail score gives 0.543 | chance |
| Layer 3 immune score | 14,138 | 0.575 (0.565 to 0.585); +0.033 over the layer-22 tail score (paired interval 0.023 to 0.043) | weak, positive |
| Codon-position bias | 34,138 | 0.512 (0.502 to 0.522) | chance |

- **Layer 3**: 20.1% of positives are flagged and 8.6% of still-dark proteins. Fisher exact test: odds ratio 2.68, p = 7e-76. Controlling for the Layer 1 score in deciles (Cochran-Mantel-Haenszel): odds ratio 2.41, p = 7e-54. So the flag carries information beyond Layer 1, but the AUROC is still small, and Layer 3 is a weaker detector of held-out families (0.74 with the frozen 5,000 detectors, 0.89 near 30,000).
- **Layer 2 (ProstT5)** against ESM-2 on the same 300 held-out families (6,907 proteins): 0.944 against 0.988. ESM-2 is better in 92% of families; Wilcoxon p = 3e-43. The structure-aware model is a clearly weaker family separator.
- **Layer 4**: a null that keeps the measured correlations between axes (Gaussian copula, 400 draws, the true size of each top-decile set) replaces "chance". The three-way set (EVT, context, composition) has 108 genes. The report's "3.4x chance" compared it with 31.7, which assumes independent axes with exactly 10% each. With the real set sizes the independence value is 42.3, and with the correlations kept it is 75.0 (95th percentile 90). So the set is 1.4x its null, p = 0.0025, not 3.4x. Its positive rate is 6.5% against 12.1%: the set has fewer, not more, later-characterised genes. No intersection of axes is significantly enriched for positives (best lift, EVT+context+Genos-m+ProstT5, 5 of 29 against 12.1% expected 3.5; hypergeometric p = 0.27). All five axes together give 20 genes, 3 positives (lift 1.24x, p = 0.44).
- **Protein length** is a stronger single predictor (AUROC 0.618) than any novelty signal. Report it as the baseline every claim has to beat.

What changes in the report: the corroborating layers are described as measured checks with intervals, not as evidence that "independent lines of evidence agree"; Key Finding 3 is withdrawn in its old form.

## Comment 7: Layer 5 coding-structure evaluation

(`scripts/reviewer_coding_structure.py`.)

- **What it is**: codon-position base bias (CPBB) has no fitted parameter, so there is no training or test split. "Real against shuffle, 0.94" is a paired control comparison on the same genes, not a prediction on independent data. The report should say that.
- **Null construction**: the shuffle null is a random permutation of the gene's nucleotides (keeps base composition, destroys order). The Markov-1 null regenerates the sequence from the gene's own order-1 transition counts (keeps dinucleotide frequencies, destroys reading frame). Both are drawn once per gene. Repeated with five seeds on 4,000 genes: real against shuffle 0.942, against Markov-1 0.944 (bootstrap intervals about 0.937 to 0.947).
- **True intergenic controls**: the earlier AUROC of 0.78 (34,138 genes against 81,286 segments, interval 0.779 to 0.784) was not length matched. Segments are short (median 168 nt against 471 nt for genes) and CPBB is a chi-square-like statistic that depends on length. Matched in ten length bins, the weighted AUROC is 0.906, from 0.736 in the shortest bin (under 168 nt) to 0.966. The 0.78 understated the signal, and the shortest segments carry most of the difference.
- **Within each genome**: the 44 genomes give a median AUROC of 0.766 (quartiles 0.704 to 0.823; range 0.540 to 0.942), all above 0.5. The AUROC rises with genome GC (Spearman 0.48), which is expected because codon-position bias is stronger in high-GC genomes.
- **Representativeness of the 44 genomes**: they cover 21 of the 189 phyla in the panel and hold 10.3% of the dark queries. The phylum mix differs from the panel (chi-square p = 0.035; for example, no Patescibacteria). Their median coding GC is 0.433 against 0.484 for the panel (KS p = 0.065). So the figure is for a lower-GC, less diverse subset, and will probably be higher in the full panel.
- **Do dark genes look like real genes?** Characterised genes score 0.992 against shuffle (3,000 genes), and dark genes 0.942, partly a length effect. Dark against characterised CPBB is 0.503, essentially the same distribution (medians 0.0721 and 0.0704).
- **Does it say anything about which dark genes get characterised?** No. CPBB separates positives from still-dark proteins at AUROC 0.512; amino-acid 3-mer entropy gives 0.621, but that statistic rises with protein length.

## Comment 8: citations, related work, bibliography

The report named tools in the text without numbered citations, and gave no related-work section. The new section (in the v3 report) groups the work as follows; every entry was checked against Crossref or the arXiv page.

- Dark matter and large-scale annotation: Rinke et al. 2013, Pavlopoulos et al. 2023, Vanni et al. 2022, HiFi-NN (Ayres 2025), FUGAsseM (Zhang 2025). What they do: map or assign function to unknown sequences. What we add: a calibrated novelty score and a leakage-controlled retrospective test; we do not assign function.
- Sequence and profile search: BLAST, HMMER, Pfam. They need a characterised neighbour.
- Protein language models: ESM-2 (Lin 2023, Rives 2021), ProtTrans (Elnaggar 2022), ProstT5 (Heinzinger 2024), Genos-m for DNA. We use them as frozen embedders and compare layers; none of them provides novelty calibration.
- Structure analysis: AlphaFold (Jumper 2021), Foldseek, and the structure-clustering studies of Barrio-Hernandez 2023 and Durairaj 2023. These find novel folds from predicted structures. Our structural layer uses a sequence-to-3Di model and is not a folding pipeline.
- Open-set recognition and out-of-distribution detection: Scheirer 2013, Geng 2021, OpenMax, the Extreme Value Machine, the softmax baseline of Hendrycks and Gimpel 2017, and deep nearest neighbours (Sun 2022). Our kNN-distance score belongs to the last family; the EVT tail is the same device as in the Extreme Value Machine. Statistical references for the tail: Pickands 1975, Balkema and de Haan 1974, Davison and Smith 1990, Choulakian and Stephens 2001 (goodness of fit with estimated parameters).
- Conformal methods for protein function (Boger 2025, Ding 2024) give coverage guarantees for labelled classes; they do not address sequences with no class.
- Positive-unlabelled learning (Elkan and Noto 2008) for the label-noise discussion; negative selection (Forrest 1994, Ji and Dasgupta 2004) for Layer 3; Prodigal (Hyatt 2010) for the gene calls behind Layer 5.

Checks done on the bibliography: every DOI above resolves to the expected title, authors and year. Corrections to entries already in the bibliography are listed in the v3 report.

## Files

`scripts/embed_full_query_reference.py`, `scripts/layer_sweep_esm2.py`, `scripts/reviewer_common_set.py`, `scripts/reviewer_scorer_analysis.py`, `scripts/layer_selection_validation.py`, `scripts/label_audit.py`, `scripts/reviewer_label_robustness.py`, `scripts/reviewer_corroboration_stats.py`, `scripts/reviewer_coding_structure.py`; outputs in `results/reviewer/`.

Embedding precision: the new layer embeddings use TF32 matrix products on the RTX 4060. They match the earlier fp32 embeddings of layers 22 and 33 to cosine 0.99994 or better, and the last-layer GPD fit and Precision@K reproduce the report (xi 0.1005 against 0.0998; lift 0.23 at K = 500).
