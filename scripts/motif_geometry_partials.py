"""Separate the two motif-geometry terms from each other and from data volume.

PURPOSE
    Section 4.5.1 reports that per-allele AUROC is predicted by anchor information
    content (+0.533) and by motif isolation (-0.291 / -0.372 depending on panel),
    and not by the number of training peptides (-0.118, ns). Those are three
    pairwise correlations. They do not establish that anchor IC and motif
    isolation are independent of each other, and the marker's feedback raised
    exactly the right objection: motif isolation may be a restatement of data
    availability, since a learner with motif-similar neighbours to borrow from
    should generalise better.

    This script answers both questions with partial correlations:

      Q1  Is anchor IC a data-volume effect in disguise?
          -> partial(anchor_ic, auroc | log10 peptide count)

      Q2  Is motif isolation a data-volume effect in disguise?
          -> partial(isolation, auroc | log10 peptide count)

      Q3  Are anchor IC and motif isolation independent of each other, or is
          one a shadow of the other?
          -> partial(isolation, auroc | anchor_ic)
             partial(anchor_ic, auroc | isolation)

    Q3 is the one that matters most. On the 74-allele subset for which nn_dist
    was already available, isolation did NOT survive controlling for anchor IC
    (rho -0.158, p 0.18) while anchor IC survived controlling for isolation
    (rho +0.504, p 5e-6). If that holds on the full panel it means the thesis
    has one geometry mechanism, not two, and the isolation result is a
    correlate of motif diffuseness rather than an independent effect. That is a
    revision to Section 4.5.1, the abstract and Section 1.2, so it must be run
    on all alleles rather than the subset before anything is rewritten.

METHOD
    Spearman throughout, to match Section 4.5. Partial correlations are computed
    on the rank-transformed variables (i.e. partial Spearman), with significance
    from the standard t statistic on n - 2 - k degrees of freedom, k = 1
    controlling variable. Confidence intervals are bootstrap percentile
    intervals over alleles (default 10,000 resamples, seeded).

    Motif isolation is recomputed here rather than read from a previous run, so
    that it is available for every allele in the panel and not only the subset
    carrying expression annotation. Definition follows Section 3.4.1: for each
    allele, build a 9 x 20 position weight matrix over its 9mers, then take the
    minimum over all other alleles of the position-averaged Jensen-Shannon
    divergence between the two PWMs. PWMs use a pseudocount so that positions
    with unobserved residues do not produce infinite divergence.

    Anchor information content is the mean of the per-position IC values at the
    allele's anchor positions, both read from anchors.json. Negative anchor
    indices are treated as offsets from the C-terminus, matching the convention
    in that file.

INPUTS
    --anchors        data/processed/anchors.json
                     per allele: "ic" (9 floats), "anchors" (indices, may be
                     negative), "n_peptides"
    --auroc          results/per_allele_auroc_v3.csv
                     columns: allele, auroc, peptide_count
    --peptides       training table with one row per peptide, used to rebuild
                     PWMs for the isolation term.
                     columns: allele, peptide  (extra columns ignored)
    --isolation      OPTIONAL. Pre-computed isolation, columns: allele, nn_dist.
                     If given, --peptides is not needed and these values are
                     used as-is. Use this to reproduce the existing numbers;
                     omit it to extend isolation to the full panel.

OUTPUTS
    results/motif_geometry_partials.csv    one row per test, with rho, p, n, CI
    results/motif_geometry_per_allele.csv  the assembled per-allele table, so
                                           the inputs to every test above are
                                           inspectable and committable

SANITY CHECK
    The script recomputes the pairwise anchor_ic vs auroc correlation and warns
    loudly if it does not land on +0.533 to two decimals. If that check fails,
    the anchor IC being computed here is not the quantity Section 4.5.1 reports
    and nothing else in the output should be trusted.

DESIGN DECISIONS AND REJECTED ALTERNATIVES
    Pearson partials were rejected: Section 4.5 is Spearman throughout, AUROC is
    bounded, and peptide counts span orders of magnitude.

    Controlling for both terms at once via multiple regression was considered
    and rejected as the headline test. With anchor IC and isolation correlated
    at about -0.44, a two-predictor model is interpretable but invites reading
    coefficient signs as mechanism. Pairwise partials answer the specific
    question asked, which is whether either term survives the other.

    Isolation is defined against the panel it is computed on. Recomputing it for
    123 alleles rather than 74 changes each allele's nearest neighbour, so the
    full-panel numbers are not expected to match the subset exactly and should
    not be forced to.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.stats import t as tdist

AA = "ACDEFGHIKLMNPQRSTVWY"
PEPTIDE_LENGTH = 9
PSEUDOCOUNT = 1e-3
EXPECTED_ANCHOR_IC_RHO = 0.533


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def load_anchor_ic(path: Path) -> pd.DataFrame:
    """Mean IC over the allele's anchor positions, plus its peptide count."""
    blob = json.loads(path.read_text())
    alleles = blob["alleles"] if "alleles" in blob else blob
    rows = []
    for allele, rec in alleles.items():
        ic = rec["ic"]
        # negative indices are offsets from the C-terminus (-1 == P9)
        idx = [i if i >= 0 else len(ic) + i for i in rec["anchors"]]
        if not idx:
            continue
        rows.append(
            {
                "allele": allele,
                "anchor_ic": float(np.mean([ic[i] for i in idx])),
                "n_anchors": len(idx),
                "n_peptides_anchors_json": rec.get("n_peptides", np.nan),
            }
        )
    return pd.DataFrame(rows)


def build_pwms(peptides: pd.DataFrame) -> dict[str, np.ndarray]:
    """One 9 x 20 position weight matrix per allele, with pseudocounts."""
    index = {aa: i for i, aa in enumerate(AA)}
    pwms: dict[str, np.ndarray] = {}
    for allele, group in peptides.groupby("allele"):
        seqs = [p for p in group["peptide"].astype(str) if len(p) == PEPTIDE_LENGTH]
        if not seqs:
            continue
        counts = np.full((PEPTIDE_LENGTH, len(AA)), PSEUDOCOUNT)
        for seq in seqs:
            for pos, aa in enumerate(seq):
                j = index.get(aa)
                if j is not None:
                    counts[pos, j] += 1.0
        pwms[allele] = counts / counts.sum(axis=1, keepdims=True)
    return pwms


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Position-averaged Jensen-Shannon divergence between two PWMs."""
    m = 0.5 * (p + q)

    def kl(a, b):
        return np.sum(a * np.log2(a / b), axis=1)

    return float(np.mean(0.5 * kl(p, m) + 0.5 * kl(q, m)))


def motif_isolation(pwms: dict[str, np.ndarray]) -> pd.DataFrame:
    """Divergence to the nearest other allele in PWM space."""
    names = sorted(pwms)
    rows = []
    for a in names:
        best, partner = np.inf, None
        for b in names:
            if a == b:
                continue
            d = _js_divergence(pwms[a], pwms[b])
            if d < best:
                best, partner = d, b
        rows.append({"allele": a, "nn_dist": best, "nearest_neighbour": partner})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def partial_spearman(frame: pd.DataFrame, x: str, y: str, given: str):
    """Spearman correlation of x and y controlling for `given`."""
    ranked = pd.DataFrame({c: rankdata(frame[c]) for c in (x, y, given)})
    c = ranked.corr()
    denom = np.sqrt((1 - c.loc[x, given] ** 2) * (1 - c.loc[y, given] ** 2))
    if denom == 0:
        return np.nan, np.nan
    rho = (c.loc[x, y] - c.loc[x, given] * c.loc[y, given]) / denom
    n, k = len(frame), 1
    if n - 2 - k <= 0 or abs(rho) >= 1:
        return float(rho), np.nan
    tstat = rho * np.sqrt((n - 2 - k) / (1 - rho**2))
    p = 2 * (1 - tdist.cdf(abs(tstat), n - 2 - k))
    return float(rho), float(p)


def bootstrap_ci(frame, fn, n_boot=10_000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(frame)
    stats = []
    for _ in range(n_boot):
        sample = frame.iloc[rng.integers(0, n, n)]
        try:
            value = fn(sample)
        except Exception:
            continue
        if value is not None and np.isfinite(value):
            stats.append(value)
    if not stats:
        return np.nan, np.nan
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--anchors", type=Path, default=Path("data/processed/anchors.json"))
    ap.add_argument("--auroc", type=Path, default=Path("results/per_allele_auroc_v3.csv"))
    ap.add_argument("--peptides", type=Path, default=None,
                    help="training table (allele, peptide) used to rebuild PWMs")
    ap.add_argument("--isolation", type=Path, default=None,
                    help="optional pre-computed isolation (allele, nn_dist)")
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.isolation is None and args.peptides is None:
        ap.error("give --peptides to recompute isolation, or --isolation to reuse it")

    ic = load_anchor_ic(args.anchors)
    perf = pd.read_csv(args.auroc)
    keep = [c for c in ("allele", "auroc", "peptide_count") if c in perf.columns]
    perf = perf[keep]

    if args.isolation is not None:
        iso = pd.read_csv(args.isolation)[["allele", "nn_dist"]]
        iso_source = f"reused from {args.isolation}"
    else:
        peptides = pd.read_csv(args.peptides)
        iso = motif_isolation(build_pwms(peptides))
        iso_source = f"recomputed from {args.peptides}"

    df = ic.merge(perf, on="allele", how="inner").merge(iso, on="allele", how="left")
    if "peptide_count" not in df.columns:
        df["peptide_count"] = df["n_peptides_anchors_json"]
    df["log_count"] = np.log10(df["peptide_count"].astype(float))

    args.outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.outdir / "motif_geometry_per_allele.csv", index=False)

    volume = df.dropna(subset=["anchor_ic", "auroc", "log_count"])
    both = df.dropna(subset=["anchor_ic", "auroc", "log_count", "nn_dist"])

    print(f"isolation: {iso_source}")
    print(f"alleles with anchor IC + AUROC + count : {len(volume)}")
    print(f"alleles additionally carrying isolation: {len(both)}")

    check_rho, _ = spearmanr(volume["anchor_ic"], volume["auroc"])
    print(f"\nsanity check, anchor_ic vs auroc: {check_rho:+.3f} "
          f"(Section 4.5.1 reports {EXPECTED_ANCHOR_IC_RHO:+.3f})")
    if abs(check_rho - EXPECTED_ANCHOR_IC_RHO) > 0.005:
        print("  WARNING: does not reproduce the reported value. The anchor IC "
              "computed here is not the quantity Section 4.5.1 describes; "
              "resolve before using anything below.", file=sys.stderr)

    results = []

    def add_pairwise(frame, x, y, question):
        rho, p = spearmanr(frame[x], frame[y])
        lo, hi = bootstrap_ci(
            frame, lambda s, x=x, y=y: spearmanr(s[x], s[y])[0],
            args.n_boot, args.seed)
        results.append(dict(test="pairwise", x=x, y=y, given="", n=len(frame),
                            rho=rho, p=p, ci_lo=lo, ci_hi=hi, question=question))

    def add_partial(frame, x, y, given, question):
        rho, p = partial_spearman(frame, x, y, given)
        lo, hi = bootstrap_ci(
            frame, lambda s, x=x, y=y, g=given: partial_spearman(s, x, y, g)[0],
            args.n_boot, args.seed)
        results.append(dict(test="partial", x=x, y=y, given=given, n=len(frame),
                            rho=rho, p=p, ci_lo=lo, ci_hi=hi, question=question))

    add_pairwise(volume, "anchor_ic", "auroc", "baseline")
    add_pairwise(volume, "log_count", "auroc", "baseline")
    add_pairwise(volume, "anchor_ic", "log_count", "are they confounded")
    add_partial(volume, "anchor_ic", "auroc", "log_count",
                "Q1 is anchor IC a data-volume effect")
    add_partial(volume, "log_count", "auroc", "anchor_ic",
                "Q1 converse")

    if len(both) > 3:
        add_pairwise(both, "nn_dist", "auroc", "baseline")
        add_pairwise(both, "anchor_ic", "nn_dist", "are they confounded")
        add_partial(both, "nn_dist", "auroc", "log_count",
                    "Q2 is isolation a data-volume effect")
        add_partial(both, "nn_dist", "auroc", "anchor_ic",
                    "Q3 does isolation survive anchor IC")
        add_partial(both, "anchor_ic", "auroc", "nn_dist",
                    "Q3 does anchor IC survive isolation")

    out = pd.DataFrame(results)
    out.to_csv(args.outdir / "motif_geometry_partials.csv", index=False)

    print()
    for _, r in out.iterrows():
        label = f"{r.x} ~ {r.y}" + (f" | {r.given}" if r.given else "")
        print(f"  {label:<38} rho={r.rho:+.3f}  p={r.p:<10.4g} "
              f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  n={r.n}   {r.question}")

    print(f"\nwrote {args.outdir/'motif_geometry_partials.csv'}")
    print(f"wrote {args.outdir/'motif_geometry_per_allele.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())