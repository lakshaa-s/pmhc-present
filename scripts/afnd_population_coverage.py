"""Whose repertoire does the per-allele evaluation actually cover?

The 123-allele validation panel is an artefact of which alleles have enough
held-out 9mers to score, not of which alleles people carry. That invites an
obvious objection: a panel chosen for data availability could be unrepresentative
of real repertoires, and per-allele AUROC averaged over it could overstate — or
understate — what the model would do in any actual population.

AFND answers this directly, because it reports allele frequency per population
sample rather than per region. For each population and each of HLA-A, -B, -C:

  COVERAGE      what fraction of that locus's allele-frequency mass is carried by
                alleles the model was scored on

  EXPECTED      the frequency-weighted mean per-allele AUROC over the covered
                alleles, i.e. the AUROC a randomly drawn allele copy from that
                population would be predicted at

Two things this is not. It is not a new evaluation — no peptide is rescored; it
reweights AUROCs already computed on the validation split (v3 model). And the
expected figure is conditional on covered alleles, so it must be read alongside
coverage, never on its own.

Population samples with fewer than --min-n typed individuals are dropped, and a
population x locus record is dropped if its reported frequencies sum to less than
--min-mass, which indicates incomplete typing at that locus rather than a genuine
repertoire.

Usage:
  python afnd_population_coverage.py \
      --afnd data/raw/afnd/afnd.tsv \
      --per-allele results/afnd_frequency_per_allele.csv \
      --out results/afnd_population_coverage.csv
"""
import argparse
import numpy as np
import pandas as pd


def load_afnd(path, min_n):
    af = pd.read_csv(path, sep="\t")
    # n and the frequency arrive as strings in the published TSV — thousands
    # separators in n, and occasional blanks in both
    for c in ("n", "alleles_over_2n"):
        af[c] = pd.to_numeric(af[c].astype(str).str.replace(",", ""), errors="coerce")
    af = af.dropna(subset=["alleles_over_2n", "n"])
    af = af[af.n >= min_n]
    af = af[af.gene.isin(["A", "B", "C"])].copy()
    af["allele_full"] = "HLA-" + af.allele.astype(str)
    return af


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--afnd", default="data/raw/afnd/afnd.tsv")
    ap.add_argument("--per-allele", default="results/afnd_frequency_per_allele.csv")
    ap.add_argument("--out", default="results/afnd_population_coverage.csv")
    ap.add_argument("--min-n", type=int, default=50,
                    help="drop population samples with fewer typed individuals")
    ap.add_argument("--min-mass", type=float, default=0.5,
                    help="drop population x locus records whose frequencies sum below this")
    args = ap.parse_args()

    af = load_afnd(args.afnd, args.min_n)
    per = pd.read_csv(args.per_allele)
    auroc = dict(zip(per.allele, per.auroc))
    print(f"{len(af):,} rows, {af.population.nunique():,} populations (n >= {args.min_n})")
    print(f"{len(per)} scored alleles; unweighted mean AUROC {per.auroc.mean():.4f}")

    rows = []
    for (pop, gene), g in af.groupby(["population", "gene"]):
        mass = g.alleles_over_2n.sum()
        if mass < args.min_mass:
            continue
        g = g.assign(a=g.allele_full.map(auroc))
        cov = g.dropna(subset=["a"])
        if cov.empty:
            continue
        rows.append(dict(
            population=pop, gene=gene, n_alleles_reported=len(g),
            n_alleles_scored=len(cov), mass_reported=mass,
            coverage=cov.alleles_over_2n.sum() / mass,
            expected_auroc=np.average(cov.a, weights=cov.alleles_over_2n)))

    R = pd.DataFrame(rows).sort_values(["gene", "expected_auroc"])
    R.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(R)} population x locus records, "
          f"{R.population.nunique()} populations)")

    print("\n=== by locus ===")
    print(R.groupby("gene").agg(
        median_coverage=("coverage", "median"),
        median_expected=("expected_auroc", "median"),
        p10_expected=("expected_auroc", lambda v: v.quantile(0.10)),
        p90_expected=("expected_auroc", lambda v: v.quantile(0.90)),
    ).round(4).to_string())

    print("\n=== does weighting by real repertoires change the headline? ===")
    print(f"  unweighted mean over scored alleles     {per.auroc.mean():.4f}")
    print(f"  median population-weighted expectation  {R.expected_auroc.median():.4f}")
    print(f"  difference                              {R.expected_auroc.median() - per.auroc.mean():+.4f}")
    print(f"  worst population x locus record         {R.expected_auroc.min():.4f} "
          f"({R.loc[R.expected_auroc.idxmin(), 'population']}, "
          f"HLA-{R.loc[R.expected_auroc.idxmin(), 'gene']})")
    lo = R[R.coverage < 0.8]
    print(f"  records with <80% mass covered          {len(lo)} of {len(R)}")


if __name__ == "__main__":
    main()
