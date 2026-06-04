from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

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


def modp(x: int) -> int:
    return x % P


def add3(a: tuple[int, int, int], b: tuple[int, int, int], L: int) -> tuple[int, int, int]:
    return ((a[0] + b[0]) % L, (a[1] + b[1]) % L, (a[2] + b[2]) % L)


def sub3(a: tuple[int, int, int], b: tuple[int, int, int], L: int) -> tuple[int, int, int]:
    return ((a[0] - b[0]) % L, (a[1] - b[1]) % L, (a[2] - b[2]) % L)


def read_csv_dicts(path: Path) -> dict[tuple[int, int, int], dict[str, int]]:
    out: dict[tuple[int, int, int], dict[str, int]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["i"]), int(row["j"]), int(row["k"]))
            out[key] = {k: int(v) % P for k, v in row.items() if k not in ("i", "j", "k")}
    return out


def z_local(row: dict[str, int]) -> dict[str, tuple[int, int]]:
    return {
        "O": (0, 0),
        "A": (0, row["vA"]),
        "B": (0, row["vB"]),
        "C": (0, row["vC"]),
        "D": (row["uD"], 0),
        "E": (row["uE"], 0),
        "F": (row["uF"], 0),
        "G": (row["w1"], row["w2"]),
    }


def x_local(row: dict[str, int]) -> dict[str, tuple[int, int]]:
    return {
        "O": (row["t1"], -row["t2"] % P),
        "A": (0, row["rA"]),
        "B": (0, row["rB"]),
        "C": (0, row["rC"]),
        "D": (row["sD"], 0),
        "E": (row["sE"], 0),
        "F": (row["sF"], 0),
        "G": (0, 0),
    }


def topology_q1(A: dict[tuple[int, int, int], dict[str, int]], s: tuple[int, int, int], L: int) -> int:
    x, y, z = s
    return modp(
        A[((x - 1) % L, (y - 1) % L, (z - 1) % L)]["w1"]
        + A[(x % L, (y - 1) % L, (z - 1) % L)]["uD"]
        + A[((x - 1) % L, y % L, (z - 1) % L)]["uE"]
        + A[((x - 1) % L, (y - 1) % L, z % L)]["uF"]
    )


def topology_q2(A: dict[tuple[int, int, int], dict[str, int]], s: tuple[int, int, int], L: int) -> int:
    x, y, z = s
    return modp(
        A[((x - 1) % L, (y - 1) % L, (z - 1) % L)]["w2"]
        + A[((x - 1) % L, y % L, z % L)]["vA"]
        + A[(x % L, (y - 1) % L, z % L)]["vB"]
        + A[(x % L, y % L, (z - 1) % L)]["vC"]
    )


def topology_sums(A: dict[tuple[int, int, int], dict[str, int]], r: tuple[int, int, int], L: int) -> tuple[int, int]:
    x, y, z = r
    su = modp(
        A[((x + 1) % L, y, z)]["uD"]
        + A[(x, (y + 1) % L, z)]["uE"]
        + A[(x, y, (z + 1) % L)]["uF"]
    )
    sv = modp(
        A[(x, (y + 1) % L, (z + 1) % L)]["vA"]
        + A[((x + 1) % L, y, (z + 1) % L)]["vB"]
        + A[((x + 1) % L, (y + 1) % L, z)]["vC"]
    )
    return su, sv


def check_topology(A: dict[tuple[int, int, int], dict[str, int]], L: int) -> tuple[bool, str]:
    for x in range(2):
        for y in range(2):
            for z in range(2):
                s = (x, y, z)
                q1 = topology_q1(A, s, L)
                q2 = topology_q2(A, s, L)
                if q1 != 0 or q2 != 0:
                    return False, f"16-constraint topology failed at site={s}: q1={q1}, q2={q2}"

    for r, row in A.items():
        su, sv = topology_sums(A, r, L)
        if modp(row["w1"] + su) != 0:
            return False, f"topology w1+Su failed at anchor={r}: w1={row['w1']}, Su={su}"
        if modp(row["w2"] + sv) != 0:
            return False, f"topology w2+Sv failed at anchor={r}: w2={row['w2']}, Sv={sv}"
        if su == 0 or sv == 0:
            return False, f"topology nonzero sum failed at anchor={r}: Su={su}, Sv={sv}"

    return True, "ok"


def local_phase(
    zrow: dict[str, int],
    xrow: dict[str, int],
    delta: tuple[int, int, int],
) -> int:
    z = z_local(zrow)
    x = x_local(xrow)
    phase = 0
    for cz, zoff in CORNERS.items():
        for cx, xoff in CORNERS.items():
            if (xoff[0] + delta[0], xoff[1] + delta[1], xoff[2] + delta[2]) != zoff:
                continue
            z1, z2 = z[cz]
            x1, x2 = x[cx]
            phase -= z1 * x1 + z2 * x2
    return modp(phase)


def check_commutation(
    A: dict[tuple[int, int, int], dict[str, int]],
    B: dict[tuple[int, int, int], dict[str, int]],
    L: int,
) -> tuple[bool, str]:
    displacements = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    for z_anchor, zrow in A.items():
        for delta in displacements:
            x_anchor = add3(z_anchor, delta, L)
            phase = local_phase(zrow, B[x_anchor], delta)
            if phase != 0:
                return (
                    False,
                    f"commutation failed z_anchor={z_anchor}, x_anchor={x_anchor}, delta={delta}, phase={phase}",
                )
    return True, "ok"


def audit_model(root: Path, manifest_row: dict[str, str]) -> dict[str, str]:
    L = int(manifest_row["L"])
    A = read_csv_dicts(root / manifest_row["z_checks_csv"])
    B = read_csv_dicts(root / manifest_row["x_checks_csv"])
    expected = L**3
    if len(A) != expected:
        return {**manifest_row, "topology_ok": "0", "commutation_ok": "0", "status": f"bad z row count {len(A)}"}
    if len(B) != expected:
        return {**manifest_row, "topology_ok": "0", "commutation_ok": "0", "status": f"bad x row count {len(B)}"}

    topology_ok, topology_msg = check_topology(A, L)
    comm_ok, comm_msg = check_commutation(A, B, L)
    status = "ok" if topology_ok and comm_ok else f"{topology_msg}; {comm_msg}"
    return {
        **manifest_row,
        "topology_ok": "1" if topology_ok else "0",
        "commutation_ok": "1" if comm_ok else "0",
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomly audit exported QtRCC check CSVs.")
    parser.add_argument("--samples", type=int, default=12, help="number of random models to audit")
    parser.add_argument("--seed", type=int, default=20260604, help="random seed")
    parser.add_argument("--include", default="", help="optional comma-separated model ids to always include")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    rows = list(csv.DictReader((root / "manifest.csv").open()))
    by_id = {row["model_id"]: row for row in rows}
    rng = random.Random(args.seed)
    picked: list[dict[str, str]] = []
    for model_id in [s.strip() for s in args.include.split(",") if s.strip()]:
        if model_id not in by_id:
            raise SystemExit(f"unknown model id: {model_id}")
        picked.append(by_id[model_id])

    remaining = [row for row in rows if row["model_id"] not in {p["model_id"] for p in picked}]
    picked.extend(rng.sample(remaining, min(args.samples, len(remaining))))

    out_rows = [audit_model(root, row) for row in picked]
    out_path = root / "audit_random_checks_results.csv"
    fieldnames = list(out_rows[0].keys()) if out_rows else [
        "model_id", "L", "topology_ok", "commutation_ok", "status"
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    bad = [row for row in out_rows if row["status"] != "ok"]
    print(f"audited {len(out_rows)} models")
    print(f"topology failures: {sum(row['topology_ok'] != '1' for row in out_rows)}")
    print(f"commutation failures: {sum(row['commutation_ok'] != '1' for row in out_rows)}")
    print(f"wrote {out_path}")
    if bad:
        for row in bad:
            print(f"FAILED {row['model_id']}: {row['status']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
