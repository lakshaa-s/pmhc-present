"""End-to-end smoke demo on SYNTHETIC data — no real biology, no GPU, no real data.

Run it to watch the full loop work before TRACERx/benchmark data exists:

    python scripts/demo_train.py

It fabricates peptides with a *learnable* presentation rule (an anchor residue plus a
hydrophobic C-terminus, allele-modulated), builds several pseudo-"alleles" at
different frequencies, makes a cluster-based train/val split, trains the NN, and prints
overall + equity-stratified metrics. AUROC should climb above 0.5 and the rare-allele
gap should be reported — proving the wiring, not any scientific claim.
"""
from __future__ import annotations

import numpy as np

from pmhcpresent.io.peptides import AMINO_ACIDS
from pmhcpresent.models.nn import PresentationNet, NetConfig
from pmhcpresent.train import PeptideMHCDataset, TrainConfig, train_model, evaluate
from pmhcpresent.eval.splits import greedy_cluster

RNG = np.random.default_rng(0)
HYDROPHOBIC = set("VILMFWY")


def random_peptide(length: int) -> str:
    return "".join(RNG.choice(list(AMINO_ACIDS), size=length))


def random_pseudoseq() -> str:
    return "".join(RNG.choice(list(AMINO_ACIDS), size=34))


def presentation_rule(pep: str, anchor: str) -> int:
    """Learnable synthetic label: anchor at P2 AND hydrophobic C-terminus → presented."""
    if len(pep) < 3:
        return 0
    score = (pep[1] == anchor) + (pep[-1] in HYDROPHOBIC)
    prob = {0: 0.05, 1: 0.4, 2: 0.85}[score]
    return int(RNG.random() < prob)


def build_dataset(n_per_allele: int = 700):
    # alleles at different "frequencies" → the equity strata
    alleles = {
        "A_common":   {"freq": "common",    "anchor": "L", "pseudo": random_pseudoseq()},
        "B_common":   {"freq": "common",    "anchor": "M", "pseudo": random_pseudoseq()},
        "C_uncommon": {"freq": "uncommon",  "anchor": "K", "pseudo": random_pseudoseq()},
        "D_rare":     {"freq": "rare",      "anchor": "F", "pseudo": random_pseudoseq()},
        "E_very_rare":{"freq": "very_rare", "anchor": "Y", "pseudo": random_pseudoseq()},
    }
    # fewer examples for rarer alleles — mirrors the real measurement-count skew
    counts = {"common": n_per_allele, "uncommon": n_per_allele // 3,
              "rare": n_per_allele // 8, "very_rare": n_per_allele // 20}

    peptides, pseuds, labels, allele_names, strata = [], [], [], [], []
    for name, meta in alleles.items():
        for _ in range(counts[meta["freq"]]):
            length = int(RNG.integers(8, 12))
            pep = random_peptide(length)
            peptides.append(pep)
            pseuds.append(meta["pseudo"])
            labels.append(presentation_rule(pep, meta["anchor"]))
            allele_names.append(name)
            strata.append(meta["freq"])
    return peptides, pseuds, labels, allele_names, strata


def main():
    peptides, pseuds, labels, alleles, strata = build_dataset()
    labels = np.array(labels)
    print(f"{len(peptides)} synthetic pairs, {labels.mean():.1%} positive")

    # cluster-based split so near-identical peptides don't straddle train/val
    clusters = greedy_cluster(peptides, identity_threshold=0.8)
    rng = np.random.default_rng(1)
    uniq = np.unique(clusters)
    val_clusters = set(rng.choice(uniq, size=max(1, len(uniq) // 5), replace=False))
    is_val = np.array([c in val_clusters for c in clusters])

    def subset(mask):
        idx = np.where(mask)[0]
        return (
            [peptides[i] for i in idx], [pseuds[i] for i in idx], labels[idx],
            [alleles[i] for i in idx], [strata[i] for i in idx],
        )

    tr = subset(~is_val)
    va = subset(is_val)
    train_ds = PeptideMHCDataset(tr[0], tr[1], tr[2], alleles=tr[3], strata=tr[4])
    val_ds = PeptideMHCDataset(va[0], va[1], va[2], alleles=va[3], strata=va[4])
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    model = PresentationNet(NetConfig())
    cfg = TrainConfig(epochs=15, batch_size=128, patience=5)
    model, _ = train_model(model, train_ds, val_ds, cfg)

    res = evaluate(model, val_ds, cfg)
    print("\noverall:", {k: round(v, 3) if isinstance(v, float) else v
                         for k, v in res["overall"].items()})
    print("\nequity (AUROC by allele frequency):")
    for stratum, m in res["equity"]["per_stratum"].items():
        print(f"  {stratum:10s} auroc={m['auroc']!s:>6}  n={m['n']}")
    print(f"\nequity gap (best−worst AUROC): {res['equity']['gap']}")


if __name__ == "__main__":
    main()
