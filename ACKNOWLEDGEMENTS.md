# Acknowledgements

Obscuron audits pretrained models and public reference databases; it trains no new
foundation model. The resources below are used under their respective licenses and
are cited here in place of vendoring them into the repository.

## Foundation models (embeddings)

**Genos-m** — "Genos-m: a foundation model for human-associated microbial genomes"
Genos Team (BGI-Research & Zhejiang Lab), bioRxiv preprint, 2026,
doi:10.64898/2026.05.21.726868.
Primary genome-context embedding arm. **Not yet peer-reviewed** — all reported
specifications and performance figures are treated as preliminary and benchmarked
independently rather than cited from the developers, and its GTDB R220 pretraining
window is treated as a leakage source to control for, not assume away
(see `docs/problems_and_decisions.md`, D3, P1-D7, P2-D7).

**ESM-2** — protein language model
Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a
language model," *Science* 379, 1123–1130 (2023). `github.com/facebookresearch/esm`
Leakage-clean headline embedding arm (UniRef50 2021_04 cutoff predates the T0
snapshot) and the protein-level baseline for the genome-vs-protein comparison.

## Structure prediction (Layer 2 — built via ProstT5)

**ProstT5** — structure-aware protein language model (bilingual amino-acid / Foldseek-3Di).
The Layer-2 arm actually built: used encoder-only as the lightweight substitute for full
3-D folding, giving a structure-informed embedding on the same held-out-family benchmark
(P4-D6). Heinzinger, Weissenow, Sanchez, Henkel, Mirdita, Steinegger & Rost, "Bilingual
language model for protein sequence and structure," *NAR Genomics and Bioinformatics* 6(4),
lqae150 (2024).

**ESMFold** — end-to-end structure prediction from sequence
Lin et al. 2023 (as above). The heavier full-folding alternative — not used; ProstT5's
encoder embedding was the memory-feasible substitute on the available hardware (§9).

**Foldseek** — fast structural search; its 3Di structural alphabet is the target ProstT5
was trained to translate into. van Kempen et al. 2024, *Nature Biotechnology* 42, 243–246.

## Methods referenced (no source code used)

**Peaks-over-threshold / Generalized Pareto tail theorem** — the theoretical basis
for Layer 1's EVT calibration (a threshold exceedance's limiting distribution is
Generalized Pareto):
Pickands, "Statistical Inference Using Extreme Order Statistics," *The Annals of
Statistics* 3(1), 119–131 (1975); Balkema & de Haan, "Residual Life Time at Great
Age," *The Annals of Probability* 2(5), 792–804 (1974).

**OpenMax** — extreme-value-theory calibration for open-set recognition
Bendale & Boult, "Towards Open Set Deep Networks," CVPR 2016.

**EVM (Extreme Value Machine)** — nonparametric, EVT-calibrated open-set
classification
Rudd, Jain, Scheirer & Boult, *IEEE Transactions on Pattern Analysis and Machine
Intelligence* 40(3), 762–768 (2018). Alongside OpenMax, part of the general-ML EVT
open-set neighbourhood Layer 1's calibrated-distance design draws on (P1-D11).

**Self / non-self discrimination** — immune-inspired anomaly detection
Forrest, Perelson, Allen & Cherukuri, IEEE S&P 1994. Basis for the Layer 3
supporting narrative.

**V-detector** — variable-radius negative selection, the specific algorithm
`src/darkmatter/immune/negative_selection.py` implements to make negative
selection tractable in 1280-dimensional embedding space:
Ji & Dasgupta, "Real-Valued Negative Selection Algorithm with Variable-Sized
Detectors," in Deb, K. & Tari, Z. (eds.) *GECCO 2004*, Part I, LNCS vol. 3102,
pp. 287–298, Springer (2004).

**Coding-structure statistics** — the statistical-regularity basis for Layer 5
(`src/darkmatter/statistical/`): codon-position composition bias distinguishes
protein-coding DNA from non-coding sequence and shuffled/Markov nulls (P4-D5):
Fickett, "Recognition of protein coding regions in DNA sequences," *Nucleic Acids
Research* 10(17), 5303–5318 (1982). The blueprint §4 frames this as the
computational-linguistics parallel — statistical regularity as evidence of authentic
coding structure.

## Software tools

**HMMER** — profile hidden Markov model search, the actual mechanism behind every
dark-at-T0 / characterised-at-T0 label in the benchmark (Pfam-35 GA-threshold
hmmscan/hmmsearch, P1-D3):
Eddy, "Accelerated Profile HMM Searches," *PLOS Computational Biology* 7(10),
e1002195 (2011).

**pyhmmer** — Python bindings to HMMER, used throughout `darkmatter/data/hmmscan.py`
and the reference-family scans in `scripts/reembed_eval.py`:
Larralde & Zeller, "PyHMMER: a Python library binding to HMMER for efficient
sequence analysis," *Bioinformatics* 39(5), btad214 (2023).

## Reference databases

**GTDB** — Genome Taxonomy Database, the historical (R207) / current (R232)
snapshot pair for retrospective validation:
Parks et al., "A standardized bacterial taxonomy based on genome phylogeny
substantially revises the tree of life," *Nature Biotechnology* 36, 996–1004
(2018); Parks et al., "GTDB: an ongoing census of bacterial and archaeal diversity
through a phylogenetically consistent, rank normalized and complete genome-based
taxonomy," *Nucleic Acids Research* 50(D1), D785–D794 (2022).

**Pfam** — protein family database, the family-assignment signal on both sides of
the T0/T1 boundary (Pfam 35.0 at T0, a Pfam 37.0 net-new-family proxy at T1,
P1-D8):
Mistry et al., "Pfam: The protein families database in 2021," *Nucleic Acids
Research* 49(D1), D412–D419 (2021).

**UniProt** — supplementary sequence reference, held in reserve for broadening the
characterised set at the go/no-go gate (not drawn on so far — the Pfam-based proxy
cleared the gate by 41–83×, P1-D5).

## Key literature

- Pavlopoulos et al., "Unraveling the functional dark matter through global metagenomics," *Nature* 622, 2023 — scale of the problem.
- Chothia, "One thousand families for the molecular biologist," *Nature* 357, 1992 — structural conservation.
- "Detecting Anomalous Proteins Using Deep Representations," *NAR Genomics and Bioinformatics* 6(1), 2024 — embedding-based protein anomaly detection.
- Zhou et al., "The CAFA challenge reports improved protein function prediction and new functional annotations for hundreds of genes through experimental screens," *Genome Biology* 20, 244 (2019) — the community precedent that temporal-holdout function-prediction benchmarking is not itself novel; Obscuron's retrospective contribution is the leakage control (P1-D7), not the holdout design.

## Closest prior art (literature search, P1-D11 + pre-submission re-run P4-D3)

- Ayres, Munsamy et al., "Annotating the microbial dark matter with HiFi-NN," *iScience* 28(6):112480, 2025 (PMC12148589) — closest by target; annotates dark matter to EC numbers over ESM-2 650M embeddings with a heuristic kNN confidence. Obscuron differs: open-set **novelty** (not annotation) with formal EVT calibration and a leakage-controlled retrospective benchmark.
- "DeepVirus" / "Illuminating the Virosphere's Dark Matter using Hierarchical Deep Learning," *bioRxiv* 2025 — open-set recognition + protein FM + genome context for novel viral groups. Obscuron differs: microbial **functional** dark matter (not viral lineage), EVT calibration (not hypothesis testing).
- Zhang et al., "Predicting functions of uncharacterized gene products from microbial communities" (FUGAsseM), *Nature Biotechnology* 2025 — genomic-context function prediction with temporal-holdout validation. Establishes retrospective validation is *not itself novel*; Obscuron's retrospective contribution is the leakage control (P1-D7).

**Added by the pre-submission re-run (P4-D3):**
- Boger et al., "Functional protein mining with conformal guarantees," *Nature Communications* 16:85, 2025, and Ding, Luo & Luo, "Leveraging conformal prediction to annotate enzyme function space with limited false positives" (CPEC), *PLOS Computational Biology* 20(5):e1012135, 2024 — establish conformal-calibrated protein **annotation**. Obscuron differs by calibrating open-set **novelty** (distance beyond known space), not the annotation decision — so the "not annotation" distinction is load-bearing.
- LAFA, "A Framework for Reproducible Longitudinal Assessment of Protein Function Annotation Models," 2026 — a temporal/longitudinal function-annotation benchmark; reinforces that temporal validation is not itself novel, the leakage control is.
- "Deciphering enzymatic potential in metagenomic reads through DNA language models" (REMME/REBEAN), *Nucleic Acids Research* 2025 — a genomic-FM metagenomic annotator (reference-free EC annotation, not calibrated novelty).

Full source list: blueprint §12 (`Microbial_Dark_Matter_Blueprint_Updated.pdf`, kept locally and not part of the public repository).
Every non-obvious design decision referencing these sources — snapshot boundary,
scorer family, immune-layer design, and every result above — is logged with its
reasoning in `docs/problems_and_decisions.md`.
