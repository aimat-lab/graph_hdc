# Code Ocean submission — working notes & handoff

Handoff/state doc for the **Code Ocean reproducibility capsule** that accompanies the
paper *"Hyper-Dimensional Fingerprints as Molecular Representations."* Read this to
pick the work back up. (Companion: the `## Paper submission artifact` section in
`CLAUDE.md`.)

Last updated: 2026-07 (session that built + validated the capsule).

---

## 1. Coordinates (the important IDs)

| Thing | Value |
|---|---|
| Paper | *Hyper-Dimensional Fingerprints as Molecular Representations* |
| Preprint | **arXiv:2604.27810** |
| Target journal | **Nature Computational Science** (NCOMPS) — Springer Nature |
| Method package (library) | Zenodo **`10.5281/zenodo.19373621`** (already deposited) |
| Experiment capsule repo | **github.com/aimat-lab/hdf-experiments** (PRIVATE), branch `main` |
| Capsule local path | `/media/ssd2/Programming/hdf-experiments` (separate git repo) |
| Local Docker image | `hdf-artifact` (python:3.11, CPU-only) |
| Capsule contact | Jonas Teufel `<jonas.teufel@kit.edu>` |
| Authors | Teufel, Torresi, Eberhard, Friederich (all KIT INT + IAR) |

The capsule is the **experiment code** artifact (reproduces the paper's results). The
Zenodo package is the **method/library** (`graph_hdc`). The paper's Code Availability
Statement cites both.

---

## 2. What the capsule is

A [Code Ocean](https://codeocean.com) **compute capsule** = a curated, self-contained
copy of this project's paper-relevant experiments + a pinned environment + a single
headless entry point, laid out in Code Ocean's convention:

```
hdf-experiments/
├── environment/Dockerfile        # pinned CPU stack (torch, rdkit, pycomex,
│                                  #   chem_mat_database + vgd_counterfactuals git deps,
│                                  #   Cairo/Pango for cairosvg)
├── environment/postInstall        # warms chem-mat-database cache for small demo datasets
├── code/
│   ├── graph_hdc/                 # VENDORED library (copied, not installed)
│   ├── experiments/fingerprints/  # only the 4 in-scope families + YAML configs +
│   │                              #   mixin_clogp.py + figure_style.py + make_figure_*.py
│   ├── tests/
│   ├── run                        # THE reproducible-run entry point (tiers below)
│   └── collect_outputs.py         # gathers metrics summary + the 3 curated figures
├── data/                          # datasets auto-fetched by chem-mat-database (nothing committed)
├── metadata/metadata.yml          # authors, affiliations, ORCIDs, MIT, arXiv link
├── requirements.txt · pyproject.toml · README.md · LICENSE (MIT) · .gitignore
```

**In-scope experiment families** (what the paper reports): `predict_molecules`,
`molecule_similarity`, `optimize_molecule_bo`, `predict_bioactivity`.
**Dropped** (not in paper): normalizing-flow / generation / reconstruction experiments,
GNN baseline, all `_slurm_*`, analysis notebooks, exploratory datasets.

---

## 3. How to build & run it locally (Docker = simulates a Code Ocean Reproducible Run)

```bash
cd /media/ssd2/Programming/hdf-experiments
docker build -t hdf-artifact -f environment/Dockerfile .
mkdir -p out
docker run --rm -v "$PWD/code":/code -v "$PWD/out":/results \
    -e TIER=demo -e PYTHONPATH=/code -e MPLBACKEND=Agg -w /code hdf-artifact bash run
# browse out/summary.md and out/figures/{figure_ged,figure_bo,figure_prediction}.pdf
```

`run` tiers (env var `TIER`):
- `smoke` — minutes, 1 seed (CI-style sanity).
- `demo` — **default**, ~25 min on CPU, 3 seeds, small datasets. Reproduces the *shape*
  of every figure. This is what Code Ocean runs.
- `full` — the paper protocol (all datasets, 5 seeds, virtual screening). Needs a
  cluster; not executed here (but reuses the same validated code paths).

Datasets download on first use via `chem-mat-database`; the Docker container has network.

---

## 4. Status — DONE ✅

- Capsule scaffolded, curated, Code Ocean layout. Git repo, `main`, ~7 commits.
- Dockerfile builds clean; **validated end-to-end** in Docker (demo run: 39 experiments,
  0 failures, ~25 min).
- **Three curated paper-quality figures** produced by `make_figure_{ged,bo,prediction}.py`
  (shared style in `figure_style.py`; HDF green `#2FB877` / Morgan blue `#4C64EB` /
  random grey). Experiments' own exploratory plots are NOT collected.
  - `figure_ged` — |Pearson corr with GED| vs embedding size (32/128/512/2048), HDF ≫ Morgan at low dim.
  - `figure_bo` — BO convergence, HDF reaches optimum by ~round 8 vs Morgan≈random flat.
  - `figure_prediction` — test R² across BACE / Lipophilicity / AqSolDB / ClogP (ClogP: HDF 0.95 vs Morgan 0.64).
- `metadata.yml` complete (4 authors, affiliations, ORCIDs [André has none], MIT, arXiv link, contact = Jonas @kit).
- Pushed to **github.com/aimat-lab/hdf-experiments** (private).
- **Paper Code Availability Statement updated** in `../latex_hyperdimensional_fingerprints`
  `main.tex` (p.17 of the recompiled `main.pdf`) and `supplementary.tex`. Both PDFs
  recompiled clean. A `% TODO` marks where the capsule DOI goes at acceptance.
  **NOTE:** the paper edits are NOT git-committed (paper repo is the user's to commit).
- Note added to this repo's `CLAUDE.md`; project memory in
  `~/.claude/projects/.../memory/paper-hdf-submission.md`.

## 5. Status — REMAINING ⬜ (all manual, deferred on purpose)

**Strategy decided:** do NOT trigger the Code Ocean *submission* until the paper is past
the editor/desk-reject stage — no point in the CO-staff verification if it might be
desk-rejected. Building the capsule (the effort) is already done and parked.

1. **Create the Code Ocean capsule** — codeocean.com → New Capsule → *Clone from Git* →
   the `aimat-lab/hdf-experiments` URL (authorize CO's GitHub access for the private repo,
   or `gh repo edit aimat-lab/hdf-experiments --visibility public`). Click **Reproducible
   Run** once to confirm. (Optional cheap insurance; can be done anytime.)
2. **When invited to peer review:** email `support@codeocean.com` with the journal +
   handling-editor details → press **Submit for publication** (this does NOT publish; it
   sends for a brief CO-staff reproducibility check) → CO gives the editor a private,
   frozen link → reviewers run it anonymously.
3. **On acceptance:** publish → DOI `10.24433/CO.<slug>.v1` → paste into the CAS `% TODO`
   in `main.tex` + add to the reference list. Set associated-publication DOI + funding in
   the CO UI (funding text: bwHPC/HoreKa, BMBF FLAIM 01DM21002A, DFG SPP 2363, Helmholtz
   Core-Informatics).
4. **Commit + recompile the paper** once the CAS is finalized (paper repo currently has
   uncommitted `main.tex`/`supplementary.tex`).
5. **`metadata/metadata.yml`** — no blockers; André Eberhard's ORCID intentionally omitted.

### How Code Ocean / OSL works (quick reference)
- Free for authors + reviewers (Springer Nature partnership). **Nature authors get NO
  compute-hour cap** (others get ~10 h). Still, full runs are cluster-only → ship the demo.
- DOIs via DataCite, versioned: `10.24433/CO.<slug>.v<n>`.
- **Once submitted the capsule is frozen/immutable** — get it right first (that's why we
  Docker-validated).

---

## 6. Gotchas already fixed (don't regress these when editing the capsule)

- **Run experiments from their own dir + `PYTHONPATH=/code`** — pycomex resolves YAML
  `extend:`/`include:` relative to CWD, and the vendored `graph_hdc` must stay importable.
  `run` handles both.
- **Small datasets use full data + fractional splits** in demo — several fp configs
  hard-code `NUM_TEST: 1000`, which breaks under `NUM_DATA` subsampling.
- **`bace` config → `bace_reg`** (regression, IC50) to match the paper (was classification).
- **Dockerfile needs** Cairo/Pango libs (cairosvg, pulled by chem-mat-database) **and**
  the `vgd_counterfactuals` git dep (n-hop neighborhoods for the GED experiment).
- **Bioactivity is `full`-only** (riniker ~168k molecules too heavy for CPU/demo).
- GED figure uses the paper's canonical sizes `[32,128,512,2048]` (64/256 were extra).

## 7. Open finding — the "dim-64 anomaly" (INVESTIGATED, no action taken yet)

HDF's GED correlation looked like it collapsed at 64 dims (0.38 for `seed=0`). A 4-seed ×
3-size sweep showed it is **NOT special to 64 and NOT a bug** — it's **low-dimension
encoder-seed variance**: at D=64 seeds 1 & 3 give 0.78/0.86 (fine); seed-0 was an unlucky
random dictionary. Spread across seeds is ±0.13–0.22 at 32/64/128.

**Implication for the paper (worth doing, deferred by user):** low-dimensional GED /
dimensionality-ablation points from a *single* encoder seed are unreliable — should be
**averaged over ~3–5 encoder seeds** (or report the spread). This is an encoder property,
independent of dataset size, so full QM9 won't wash it out. HDF still clearly beats Morgan
regardless. Could be wired into `run` + `make_figure_ged.py` (loop seeds, average) if
desired — ~3–5× the GED runtime.

---

## 8. Caveat: the capsule is a VENDORED copy

`../hdf-experiments/code/graph_hdc/` and the experiment scripts are **copies** of this
repo's files (assembled once). Edits *here* do **not** propagate to the capsule. If a
paper-relevant experiment or the library changes, re-sync the affected files into
`../hdf-experiments` and re-run the demo + figures to re-validate.
