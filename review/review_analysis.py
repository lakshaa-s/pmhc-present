"""External review analyses: gaps found by re-analysing the committed result tables.

Adds four things the project does not currently compute:

  1. STRUCTURAL CONSENSUS  Do the four folding architectures add to *each other*?
                           The project tests each individually against sequence,
                           never their average. Tested on v4 (4 architectures) and
                           v2 (3 -- AF3 was never run there), with allele-cluster
                           bootstrap on the paired difference.

  2. POWER + CEILING       RQ2's null is currently explained by "sample size is the
                           likelier explanation" (PROGRESS.md). This puts a number
                           on it, and adds an independent oracle ceiling from the
                           margin table.

  3. CALIBRATION           Every metric in the project is rank-based and therefore
                           blind to whether the score means anything as a
                           probability. It does not. Leave-one-allele-out
                           temperature scaling is fitted as the fix.

  4. DECISION METRICS      pAUC at low FPR and PPV@k, which reorder the structural
                           models relative to AUROC.

Reads only files already in git. Writes review_*.csv + REVIEW figure inputs.
Structural readout is pae_anchors_ic, per-allele z-scored, sign-flipped -- the
same convention as scripts/auroc_structure.py.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parent
STRUCT_V4 = ["esmfold2", "af2", "af3", "boltz"]
STRUCT_V2 = ["esmfold2", "af2", "boltz"]          # AF3 never run on v2
READOUT = "pae_anchors_ic"
N_BOOT = 4000


def auroc(y, s):
    y = np.asarray(y)
    r = rankdata(np.asarray(s, float))
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def zscore_within_allele(values, alleles):
    return values.groupby(alleles).transform(lambda x: (x - x.mean()) / x.std(ddof=0))


def load_panel(version, arch):
    """One row per complex: label + z-scored score per model. Sequence only exists for v4."""
    frames = {m: pd.read_csv(ROOT / f"pae_{m}_{version}.csv") for m in arch}
    base = next(iter(frames.values()))
    out = base[["allele", "peptide"]].copy()
    out["label"] = (base["kind"] == "binder").astype(int)
    for m, d in frames.items():
        out = out.merge(
            d[["allele", "peptide", READOUT]].rename(columns={READOUT: m}),
            on=["allele", "peptide"], validate="one_to_one",
        )
    if version == "v4":
        seq = pd.read_csv(ROOT / "results/sequence_v4.csv")
        out = out.merge(
            seq[["allele", "peptide", "score"]].rename(columns={"score": "sequence"}),
            on=["allele", "peptide"], validate="one_to_one",
        )
    assert out.notna().all().all(), "missing values after merge"

    raw = out.copy()
    for m in arch:                      # lower PAE = more binder-like
        out[m] = zscore_within_allele(-out[m], out["allele"])
    if "sequence" in out:
        out["sequence"] = zscore_within_allele(out["sequence"], out["allele"])
    out["consensus"] = out[arch].mean(axis=1)
    return out, raw


def cluster_bootstrap_diff(sa, sb, y, alleles, n_boot=N_BOOT, seed=11):
    """Paired AUROC difference, resampling ALLELES not complexes.

    The project's bootstrap resamples complexes, which are nested within alleles;
    this is the cluster-aware version. Reported as a sensitivity check.
    """
    rng = np.random.default_rng(seed)
    groups = np.array(sorted(pd.unique(alleles)))
    index_of = {a: np.flatnonzero(alleles.values == a) for a in groups}
    d = []
    for _ in range(n_boot):
        pick = rng.choice(groups, len(groups), replace=True)
        i = np.concatenate([index_of[a] for a in pick])
        if len(np.unique(y[i])) < 2:
            continue
        d.append(auroc(y[i], sa[i]) - auroc(y[i], sb[i]))
    return float(np.mean(d)), *(float(v) for v in np.percentile(d, [2.5, 97.5]))


# ---------------------------------------------------------------- decision metrics
def partial_auroc(y, s, max_fpr):
    """McClish-standardised pAUC over FPR in [0, max_fpr]. 0.5 = chance."""
    o = np.argsort(-np.asarray(s, float))
    y = np.asarray(y)[o]
    P, N = y.sum(), len(y) - y.sum()
    tpr = np.r_[0, np.cumsum(y) / P]
    fpr = np.r_[0, np.cumsum(1 - y) / N]
    k = fpr <= max_fpr
    area = np.trapezoid(np.append(tpr[k], np.interp(max_fpr, fpr, tpr)),
                        np.append(fpr[k], max_fpr))
    return float(0.5 * (1 + (area - max_fpr ** 2 / 2) / (max_fpr - max_fpr ** 2 / 2)))


def ppv_at_k(y, s, k):
    return float(np.asarray(y)[np.argsort(-np.asarray(s, float))[:k]].mean())


# ---------------------------------------------------------------- calibration
def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def temperature_scale_loao(p, y, alleles):
    """Fit one temperature per held-out allele. In-sample T would flatter the result."""
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

    def nll(T, mask):
        q = np.clip(1 / (1 + np.exp(-logit[mask] / T)), 1e-9, 1 - 1e-9)
        return -np.mean(y[mask] * np.log(q) + (1 - y[mask]) * np.log(1 - q))

    out = np.zeros_like(p, dtype=float)
    for a in pd.unique(alleles):
        te = (alleles == a).values
        T = minimize_scalar(lambda T: nll(T, ~te), bounds=(0.05, 20), method="bounded").x
        out[te] = 1 / (1 + np.exp(-logit[te] / T))
    T_all = minimize_scalar(lambda T: nll(T, np.ones(len(p), bool)),
                            bounds=(0.05, 20), method="bounded").x
    return out, float(T_all)


# ---------------------------------------------------------------- power
def power_curve(y, s_base, s_effect, sizes, n_sim=300, n_inner=150, seed=0):
    """P(paired bootstrap CI excludes zero) vs panel size, at the OBSERVED effect size.

    Resamples the existing panel up to size n, so it inherits the real score
    distribution rather than assuming one.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for n in sizes:
        hits = usable = 0
        for _ in range(n_sim):
            i = rng.integers(0, len(y), n)
            yy = y[i]
            if yy.sum() < 2 or (1 - yy).sum() < 2:
                continue
            usable += 1
            a, b = s_base[i], s_effect[i]
            d = []
            for _ in range(n_inner):
                j = rng.integers(0, n, n)
                if len(np.unique(yy[j])) < 2:
                    continue
                d.append(auroc(yy[j], b[j]) - auroc(yy[j], a[j]))
            if np.percentile(d, 2.5) > 0:
                hits += 1
        out[n] = hits / usable
    return out


def main():
    v4, _ = load_panel("v4", STRUCT_V4)
    v2, _ = load_panel("v2", STRUCT_V2)
    y4, y2 = v4.label.values, v2.label.values
    models4 = ["sequence", *STRUCT_V4]
    res = {}

    # --- 1. consensus -------------------------------------------------------
    v4["cons3"] = v4[STRUCT_V2].mean(axis=1)          # like-for-like with v2
    res["auroc_v4"] = {m: auroc(y4, v4[m]) for m in [*models4, "consensus", "cons3"]}
    res["auroc_v2"] = {m: auroc(y2, v2[m]) for m in [*STRUCT_V2, "consensus"]}

    best4 = max(STRUCT_V4, key=lambda m: res["auroc_v4"][m])
    best2 = max(STRUCT_V2, key=lambda m: res["auroc_v2"][m])
    res["paired"] = {
        f"consensus(4) - {best4} [v4]":
            cluster_bootstrap_diff(v4.consensus.values, v4[best4].values, y4, v4.allele),
        "sequence - consensus(4) [v4]":
            cluster_bootstrap_diff(v4.sequence.values, v4.consensus.values, y4, v4.allele),
        f"consensus(3) - {best2} [v2]":
            cluster_bootstrap_diff(v2.consensus.values, v2[best2].values, y2, v2.allele),
        f"sequence - {best4} [v4, cluster]":
            cluster_bootstrap_diff(v4.sequence.values, v4[best4].values, y4, v4.allele),
    }
    pd.DataFrame([
        dict(comparison=k, delta_auroc=round(m, 4), ci_lo=round(lo, 4),
             ci_hi=round(hi, 4), excludes_zero="yes" if (lo > 0 or hi < 0) else "no")
        for k, (m, lo, hi) in res["paired"].items()
    ]).to_csv(ROOT / "review_consensus_paired_diffs.csv", index=False)

    subsets = [dict(subset=f"drop {d}", members=", ".join(m for m in STRUCT_V4 if m != d),
                    auroc=round(auroc(y4, v4[[m for m in STRUCT_V4 if m != d]].mean(axis=1)), 4))
               for d in STRUCT_V4]
    subsets += [dict(subset="pair", members=f"{a} + {b}",
                     auroc=round(auroc(y4, v4[[a, b]].mean(axis=1)), 4))
                for a, b in itertools.combinations(STRUCT_V4, 2)]
    subsets.append(dict(subset="all four", members=", ".join(STRUCT_V4),
                        auroc=round(res["auroc_v4"]["consensus"], 4)))
    pd.DataFrame(subsets).to_csv(ROOT / "review_consensus_subsets_v4.csv", index=False)

    # why it works: within-allele agreement
    rho = pd.DataFrame({a: {b: np.mean([spearmanr(g[a], g[b]).statistic
                                        for _, g in v4.groupby("allele")])
                            for b in models4} for a in models4})
    rho.round(3).to_csv(ROOT / "review_within_allele_agreement_v4.csv")
    res["rho_struct_struct"] = float(
        rho.loc[STRUCT_V4, STRUCT_V4].values[np.triu_indices(len(STRUCT_V4), 1)].mean())
    res["rho_struct_sequence"] = float(rho.loc[STRUCT_V4, "sequence"].mean())

    pd.DataFrame({m: v4.groupby("allele").apply(lambda g: auroc(g.label, g[m]),
                                                include_groups=False)
                  for m in ["sequence", best4, "consensus"]}).round(4).to_csv(
        ROOT / "review_per_allele_consensus_v4.csv")

    # --- 2. power + ceiling -------------------------------------------------
    blend = (0.7 * rankdata(v4.sequence.values) + 0.3 * rankdata(v4[best4].values)) / len(v4)
    res["power_effect"] = float(auroc(y4, blend) - auroc(y4, v4.sequence.values))
    res["power"] = power_curve(y4, v4.sequence.values, blend, [216, 432, 864, 1728])
    pd.DataFrame([dict(n_complexes=n, power=round(p, 3)) for n, p in res["power"].items()]
                 ).to_csv(ROOT / "review_rq2_power.csv", index=False)

    margins = pd.read_csv(ROOT / "results/rq2_error_overlap_margins.csv")
    mc = ["sequence", *STRUCT_V4]
    res["ceiling"] = dict(
        per_model={m: float(margins[m].mean()) for m in mc},
        oracle_all5=float(margins[mc].max(axis=1).mean()),
        n_sequence_wrong=int((margins.sequence < 0.5).sum()),
        n_rescued_by_structure=int(
            (margins.loc[margins.sequence < 0.5, STRUCT_V4].max(axis=1) > 0.5).sum()),
    )

    # --- 3. calibration ----------------------------------------------------
    seq = pd.read_csv(ROOT / "results/sequence_v4.csv")
    seq = seq.merge(v4[["allele", "peptide"]], on=["allele", "peptide"])
    p, yc = seq.score.values.astype(float), seq.label.values
    q, T_all = temperature_scale_loao(p, yc, seq.allele)
    res["calibration"] = dict(
        prevalence=float(yc.mean()), mean_prediction=float(p.mean()),
        brier=brier(yc, p), ece=ece(yc, p), auroc=auroc(yc, p),
        T_in_sample=T_all,
        brier_loao=brier(yc, q), ece_loao=ece(yc, q), auroc_loao=auroc(yc, q),
    )
    edges = np.linspace(0, 1, 11)
    rel = []
    for i in range(10):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < 9 else p <= 1)
        if m.sum():
            rel.append(dict(bin_lo=edges[i], bin_hi=edges[i + 1], n=int(m.sum()),
                            mean_predicted=round(float(p[m].mean()), 4),
                            observed=round(float(yc[m].mean()), 4)))
    res["reliability"] = rel
    pd.DataFrame(rel).to_csv(ROOT / "review_calibration_reliability_v4.csv", index=False)

    # --- 4. decision metrics ----------------------------------------------
    pd.DataFrame([
        dict(model=m, auroc=round(auroc(y4, v4[m]), 4),
             pauc_fpr10=round(partial_auroc(y4, v4[m], 0.10), 4),
             pauc_fpr20=round(partial_auroc(y4, v4[m], 0.20), 4),
             ppv_top20=round(ppv_at_k(y4, v4[m], 20), 4),
             ppv_top54=round(ppv_at_k(y4, v4[m], 54), 4))
        for m in [*models4, "consensus"]
    ]).to_csv(ROOT / "review_decision_metrics_v4.csv", index=False)

    # --- panel composition -------------------------------------------------
    cols = ["tag", "locus", "allele_slug", "peptide", "note"]
    res["locus_counts"] = {
        v: pd.read_csv(ROOT / f"fold_sets/fold_set_{v}.csv", names=cols, header=None
                       ).locus.value_counts().to_dict()
        for v in ("v2", "v4")
    }
    v4["locus"] = v4.allele.str[4]
    res["locus_margin_v4"] = {
        loc: dict(n=int(len(g)), n_alleles=int(g.allele.nunique()),
                  sequence=auroc(g.label, g.sequence),
                  best_structural=max(auroc(g.label, g[m]) for m in STRUCT_V4))
        for loc, g in v4.groupby("locus")
    }
    per_allele = pd.read_csv(ROOT / "results/per_allele_auroc_v3.csv")
    per_allele["locus"] = per_allele.allele.str[4]
    res["validation_locus"] = per_allele.groupby("locus").auroc.agg(
        ["count", "mean", "median", "std"]).round(4).to_dict("index")

    cov = json.loads((ROOT / "data/processed/iedb_coverage.json").read_text())
    cov = pd.DataFrame({k: v for k, v in cov.items()
                        if k.startswith(("HLA-A*", "HLA-B*", "HLA-C*")) and "/" not in k}).T
    cov.index.name = "allele"
    cov = cov.reset_index()
    cov["locus"] = cov.allele.str[4]
    cov["neg_per_pos"] = cov.neg / cov.pos.replace(0, np.nan)
    res["iedb_neg_per_pos_median"] = cov[cov.pos >= 100].groupby(
        "locus").neg_per_pos.median().round(4).to_dict()

    (ROOT / "review_summary.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in
                      ["auroc_v4", "auroc_v2", "paired", "power", "calibration"]},
                     indent=1, default=lambda o: round(float(o), 4)))


if __name__ == "__main__":
    main()
