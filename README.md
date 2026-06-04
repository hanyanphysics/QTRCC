# Curated QtRCC Check CSVs, L=10..25

This folder contains only the local stabilizer-check CSV data for the curated
QtRCC publication examples with `L=10..25`.

For each model, the files are:

- `z_checks.csv`: the Z-type stabilizer check parameters.
- `x_checks.csv`: the X-type stabilizer check parameters.

The original model folders also contain provenance, solution, and helper files;
those are intentionally not copied here. Use `manifest.csv` to locate every
exported model and to see its basic code parameters.

## Folder Layout

```text
results_publish/curated_qtrcc_check_csv_L10_L25/
  README.md
  manifest.csv
  L10/
    L10_model_01/
      z_checks.csv
      x_checks.csv
    ...
  L11/
    L11_model_01/
      z_checks.csv
      x_checks.csv
    ...
  ...
  L25/
    L25_model_01/
      z_checks.csv
      x_checks.csv
```

## Field and Index Conventions

All arithmetic is over `F_3`, so entries `0,1,2` are interpreted modulo 3.
The CSV coordinates `i,j,k` are zero-based cube-anchor coordinates in
`{0,...,L-1}^3`, with periodic boundary conditions.

Each lattice site has two qutrits:

- qutrit 1: acted on by `XI` or `ZI`
- qutrit 2: acted on by `IX` or `IZ`

Each stabilizer check is attached to one cube anchor `(i,j,k)` and acts on the
eight cube corners. The corner offsets from the anchor are:

| corner | offset `(dx,dy,dz)` |
|---|---|
| `O` | `(0,0,0)` |
| `A` | `(1,0,0)` |
| `B` | `(0,1,0)` |
| `C` | `(0,0,1)` |
| `D` | `(0,1,1)` |
| `E` | `(1,0,1)` |
| `F` | `(1,1,0)` |
| `G` | `(1,1,1)` |

The physical site for corner `c` of check anchor `(i,j,k)` is
`(i+dx, j+dy, k+dz) mod L`.

## `z_checks.csv`

Header:

```csv
i,j,k,w1,w2,uD,uE,uF,vA,vB,vC
```

This row defines one Z-type stabilizer check. Its nonzero local factors are:

| corner | qutrit 1 exponent | qutrit 2 exponent |
|---|---:|---:|
| `G` | `w1` | `w2` |
| `D` | `uD` | `0` |
| `E` | `uE` | `0` |
| `F` | `uF` | `0` |
| `A` | `0` | `vA` |
| `B` | `0` | `vB` |
| `C` | `0` | `vC` |
| `O` | `0` | `0` |

For example, `uD=2` means the Z check contains `ZI^2` on qutrit 1 at corner
`D`, i.e. at site `(i,j+1,k+1) mod L`.

## `x_checks.csv`

Header:

```csv
i,j,k,t1,t2,rA,rB,rC,sD,sE,sF
```

This row defines one X-type stabilizer check. Its nonzero local factors are:

| corner | qutrit 1 exponent | qutrit 2 exponent |
|---|---:|---:|
| `O` | `t1` | `-t2 mod 3` |
| `A` | `0` | `rA` |
| `B` | `0` | `rB` |
| `C` | `0` | `rC` |
| `D` | `sD` | `0` |
| `E` | `sE` | `0` |
| `F` | `sF` | `0` |
| `G` | `0` | `0` |

The minus sign on the qutrit-2 exponent at `O` is part of the model convention.
For instance, if `t2=1`, the local factor on qutrit 2 at corner `O` is
`IX^2`, since `-1 = 2 mod 3`.

## Matrix Reconstruction

To build the CSS check matrices:

1. Let `m = L^3` and `n = 2L^3`.
2. Number check rows by cube anchor:
   `row = ((i * L) + j) * L + k`, using zero-based `i,j,k`.
3. Number qutrit columns by site and qutrit:
   `site = ((x * L) + y) * L + z`,
   `col(qutrit 1) = 2 * site`,
   `col(qutrit 2) = 2 * site + 1`.
4. For each row in `z_checks.csv`, add the listed Z exponents into the Z-check
   matrix `HZ[row, col]` at the corresponding corner sites.
5. For each row in `x_checks.csv`, add the listed X exponents into the X-check
   matrix `HX[row, col]` at the corresponding corner sites.
6. Reduce all entries modulo 3.

With this convention, the CSS commutation condition is:

```text
HZ * HX^T = 0 mod 3
```

The code dimension is:

```text
k = 2L^3 - rank(HZ) - rank(HX)
```

The manifest records the verified `rank_HZ`, `rank_HX`, and `k` for each model
from the source curated collection.

## Minimal Python Loader Sketch

```python
import csv
import numpy as np

P = 3
CORNERS = {
    "O": (0, 0, 0),
    "A": (1, 0, 0),
    "B": (0, 1, 0),
    "C": (0, 0, 1),
    "D": (0, 1, 1),
    "E": (1, 0, 1),
    "F": (1, 1, 0),
    "G": (1, 1, 1),
}

def row_index(i, j, k, L):
    return (i * L + j) * L + k

def col_index(x, y, z, q, L):
    site = (x * L + y) * L + z
    return 2 * site + q  # q=0 for qutrit 1, q=1 for qutrit 2

def add_term(M, row, i, j, k, L, corner, q1_exp, q2_exp):
    dx, dy, dz = CORNERS[corner]
    x, y, z = (i + dx) % L, (j + dy) % L, (k + dz) % L
    if q1_exp % P:
        M[row, col_index(x, y, z, 0, L)] = (M[row, col_index(x, y, z, 0, L)] + q1_exp) % P
    if q2_exp % P:
        M[row, col_index(x, y, z, 1, L)] = (M[row, col_index(x, y, z, 1, L)] + q2_exp) % P

def load_checks(model_dir, L):
    HZ = np.zeros((L**3, 2 * L**3), dtype=np.int8)
    HX = np.zeros((L**3, 2 * L**3), dtype=np.int8)

    with open(model_dir / "z_checks.csv") as f:
        for r in csv.DictReader(f):
            i, j, k = int(r["i"]), int(r["j"]), int(r["k"])
            row = row_index(i, j, k, L)
            add_term(HZ, row, i, j, k, L, "G", int(r["w1"]), int(r["w2"]))
            add_term(HZ, row, i, j, k, L, "D", int(r["uD"]), 0)
            add_term(HZ, row, i, j, k, L, "E", int(r["uE"]), 0)
            add_term(HZ, row, i, j, k, L, "F", int(r["uF"]), 0)
            add_term(HZ, row, i, j, k, L, "A", 0, int(r["vA"]))
            add_term(HZ, row, i, j, k, L, "B", 0, int(r["vB"]))
            add_term(HZ, row, i, j, k, L, "C", 0, int(r["vC"]))

    with open(model_dir / "x_checks.csv") as f:
        for r in csv.DictReader(f):
            i, j, k = int(r["i"]), int(r["j"]), int(r["k"])
            row = row_index(i, j, k, L)
            add_term(HX, row, i, j, k, L, "O", int(r["t1"]), -int(r["t2"]))
            add_term(HX, row, i, j, k, L, "A", 0, int(r["rA"]))
            add_term(HX, row, i, j, k, L, "B", 0, int(r["rB"]))
            add_term(HX, row, i, j, k, L, "C", 0, int(r["rC"]))
            add_term(HX, row, i, j, k, L, "D", int(r["sD"]), 0)
            add_term(HX, row, i, j, k, L, "E", int(r["sE"]), 0)
            add_term(HX, row, i, j, k, L, "F", int(r["sF"]), 0)

    return HZ % P, HX % P
```

## Contents

This export contains `374` curated models. Each model directory contains only
the two check CSVs needed to reconstruct the CSS stabilizer matrices.

## Random Audit Script

The folder also includes:

```text
audit_random_checks.py
audit_random_checks_results.csv
```

The audit script randomly samples exported models, reconstructs the local checks
from `z_checks.csv` and `x_checks.csv`, and verifies:

1. the local topology constraints from the Z-check parameters,
2. the nonzero topology sums,
3. direct CSS commutation for every overlapping local X/Z check pair.

The included result CSV was generated with:

```bash
python3 results_publish/curated_qtrcc_check_csv_L10_L25/audit_random_checks.py \
  --samples 12 \
  --seed 20260604 \
  --include L10_model_01,L18_model_01,L25_model_01
```

This audited 15 models total: the three explicitly included models plus 12
seeded random models. The result was:

```text
topology failures: 0
commutation failures: 0
```
