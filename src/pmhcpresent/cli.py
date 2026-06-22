"""P11 command-line entry point.

Subcommands are thin stubs that wire the library modules into a runnable pipeline.
Fill in the data-loading specifics once the benchmark/application splits are fixed.

    pmhcpresent parse-netmhcpan out.txt --to predictions.csv
    pmhcpresent struct-features model.pdb --peptide-chain B --mhc-chain A
    pmhcpresent cluster peptides.txt --threshold 0.8
    pmhcpresent train --data labelled.csv --pseudoseq hla_a.json hla_b.json hla_c.json
"""
from __future__ import annotations

import argparse
import sys


def _cmd_parse_netmhcpan(args) -> int:
    from pmhcpresent.io.netmhcpan import parse_netmhcpan_file, records_to_frame

    records = parse_netmhcpan_file(args.path)
    print(f"Parsed {len(records)} predictions "
          f"({sum(r.is_binder for r in records)} binders).")
    if args.to:
        records_to_frame(records).to_csv(args.to, index=False)
        print(f"Wrote {args.to}")
    return 0


def _cmd_struct_features(args) -> int:
    from pmhcpresent.structure.features import extract_structure_features

    sf = extract_structure_features(
        args.pdb, args.peptide_chain, args.mhc_chain,
        pae_json=args.pae_json, ipsae_script=args.ipsae_script,
    )
    for k, v in sf.features.items():
        print(f"  {k:28s} {v}")
    print(f"  fixed-backbone subset: {list(sf.fixed_backbone_subset())}")
    print(f"  refold-required subset: {list(sf.refold_subset())}")
    return 0


def _cmd_cluster(args) -> int:
    from pmhcpresent.eval.splits import greedy_cluster

    peptides = [ln.strip() for ln in open(args.path) if ln.strip()]
    cids = greedy_cluster(peptides, args.threshold)
    print(f"{len(peptides)} peptides → {len(set(cids))} clusters "
          f"at identity ≥ {args.threshold}")
    return 0


def _cmd_train(args) -> int:
    import numpy as np
    import pandas as pd

    from pmhcpresent.io.pseudoseq import load_pseudosequences_json
    from pmhcpresent.train import PeptideMHCDataset, TrainConfig, train_model, evaluate
    from pmhcpresent.models.nn import PresentationNet, NetConfig
    from pmhcpresent.eval.splits import exact_dedup_cluster
    from pmhcpresent.eval.stratified import assign_frequency_bins

    df = pd.read_csv(args.data)
    pseudo = load_pseudosequences_json(args.pseudoseq)  # one or more JSON paths

    # Derive equity strata from per-allele peptide count (training-data
    # representation, the RQ1 axis). Count positives per allele, map each row's
    # allele to that count, then bin into rare/low/medium/high.
    pos_counts = df[df[args.label_col] == 1].groupby(args.allele_col).size()
    row_counts = df[args.allele_col].map(pos_counts).fillna(0).to_numpy()
    df["_stratum"] = assign_frequency_bins(row_counts, args.count_bins, args.count_labels)

    ds = PeptideMHCDataset.from_frame(
        df, pseudo,
        peptide_col=args.peptide_col, allele_col=args.allele_col,
        label_col=args.label_col, stratum_col="_stratum",
    )

    # cluster-based holdout so similar peptides don't leak across the split
    clusters = exact_dedup_cluster(ds.peptides, ds.alleles)
    rng = np.random.default_rng(42)
    uniq = np.unique(clusters)
    val_clusters = set(rng.choice(uniq, size=max(1, len(uniq) // 5), replace=False))
    is_val = np.array([c in val_clusters for c in clusters])

    def subset(mask):
        idx = np.where(mask)[0]
        alleles = [ds.alleles[i] for i in idx] if ds.alleles else None
        strata = ds.strata[idx] if ds.strata is not None else None
        peps = [ds.peptides[i] for i in idx]
        pseuds = [pseudo.get(a) for a in alleles] if alleles else None
        return PeptideMHCDataset(
            peps, pseuds, ds.labels[idx], alleles=alleles, strata=strata
        )

    train_ds, val_ds = subset(~is_val), subset(is_val)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    model = PresentationNet(NetConfig())
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size)
    model, _ = train_model(model, train_ds, val_ds, cfg)

    res = evaluate(model, val_ds, cfg)
    print("overall:", res["overall"])
    if "equity" in res:
        print("equity gap:", res["equity"]["gap"])
    if args.save:
        import torch
        torch.save(model.state_dict(), args.save)
        print(f"saved weights → {args.save}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pmhcpresent", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("parse-netmhcpan", help="Parse NetMHCpan-4.1 output")
    a.add_argument("path")
    a.add_argument("--to", help="write predictions to CSV")
    a.set_defaults(func=_cmd_parse_netmhcpan)

    b = sub.add_parser("struct-features", help="Extract AlphaFold structure features")
    b.add_argument("pdb")
    b.add_argument("--peptide-chain", required=True)
    b.add_argument("--mhc-chain", required=True)
    b.add_argument("--pae-json")
    b.add_argument("--ipsae-script")
    b.set_defaults(func=_cmd_struct_features)

    c = sub.add_parser("cluster", help="Greedy peptide clustering for leakage control")
    c.add_argument("path", help="text file, one peptide per line")
    c.add_argument("--threshold", type=float, default=0.8)
    c.set_defaults(func=_cmd_cluster)

    t = sub.add_parser("train", help="Train the sequence presentation model")
    t.add_argument("--data", required=True, help="CSV with peptide/allele/label columns")
    t.add_argument("--pseudoseq", required=True, nargs="+",
                   help="allele→pseudosequence JSON file(s), e.g. hla_a/b/c.json")
    t.add_argument("--peptide-col", default="peptide")
    t.add_argument("--allele-col", default="allele")
    t.add_argument("--label-col", default="label")
    t.add_argument("--count-bins", type=float, nargs="+",
                   default=[0, 1000, 2500, 6500, 12000, 100000],
                   help="peptide-count bin edges for equity stratification")
    t.add_argument("--count-labels", nargs="+",
                   default=["rare", "low", "medium", "high", "very_high"],
                   help="labels for the count bins (len == len(count-bins) - 1)")
    t.add_argument("--cluster-threshold", type=float, default=0.8)
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--save", help="path to save trained weights (.pt)")
    t.set_defaults(func=_cmd_train)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())