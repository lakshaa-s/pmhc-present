"""P11 command-line entry point.

Subcommands are thin stubs that wire the library modules into a runnable pipeline.
Fill in the data-loading specifics once the benchmark/application splits are fixed.

    pmhcpresent parse-netmhcpan out.txt --to predictions.csv
    pmhcpresent struct-features model.pdb --peptide-chain B --mhc-chain A
    pmhcpresent cluster peptides.txt --threshold 0.8
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

    peptides = [l.strip() for l in open(args.path) if l.strip()]
    cids = greedy_cluster(peptides, args.threshold)
    print(f"{len(peptides)} peptides → {len(set(cids))} clusters "
          f"at identity ≥ {args.threshold}")
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

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
