"""Does per-allele performance track European allele frequency specifically?

The question
------------
The literature review committed to stratifying evaluation by population allele
frequency. The project's existing equity analysis uses *peptide counts* — a
data-representation measure — so the commitment is unmet: nothing here has tested
whether alleles common in non-European populations are predicted worse than alleles
common in European ones.

That is the ancestry version of the question, and it is different from the one
already answered. Existing findings: motif nearest-neighbour distance predicts
per-allele AUROC (-0.291) while sample size does not (-0.118). If Europe-biased
training data were the driver, European frequency should predict performance beyond
what overall frequency explains.

Method, and why it is built this way
-------------------------------------
AFND gives allele frequency per population, with 1,493 free-text population labels of
the form "Australia New South Wales Caucasian" or "Brazil Puyanawa" — a country
followed by a descriptor. There is no region or ancestry column.

Populations are therefore mapped to **geographic regions via country**, by
longest-prefix match against an explicit country list. Two reasons for using country
rather than the ancestry descriptors in the labels: country-to-region is a citable
classification rather than one invented here, and the descriptors are inconsistent in
ways that would make any grouping contestable. The mapping is written to a CSV on
first run so it can be inspected, corrected and committed rather than living inside
the code, and unmapped populations are reported rather than silently dropped.

Per allele, a sample-size-weighted mean frequency is computed within each region, and
then three things are asked:

  PER REGION    Does frequency in each region predict per-allele AUROC? Reported for
                every region with enough alleles, so Europe can be compared against
                the others rather than examined alone.

  ENRICHMENT    Does a Europe-enrichment index — log ratio of European frequency to
                frequency elsewhere — predict AUROC? This is the direct form of the
                question and does not confound "common everywhere" with "common in
                Europe".

  PARTIAL       Does European frequency predict AUROC after controlling for global
                frequency? Regional frequencies are strongly correlated with each
                other, because common alleles tend to be common everywhere, so the
                partial is the test that distinguishes Europe-specific bias from
                overall commonness.

Three limitations, all of which weaken any positive finding and must be reported
--------------------------------------------------------------------------------
AFND's own population sample is not representative of humanity: European and North
American populations are heavily overrepresented in the database. So "European
frequency" is measured well and "frequency elsewhere" less well, and the independent
variable carries some of the bias the analysis is trying to detect.

This is an ecological analysis at the level of alleles. It says nothing about
individuals or about clinical outcomes, and must not be written as if it does.

And the region mapping is a modelling choice. It is auditable rather than
authoritative, and a different defensible mapping could give a different answer.

Usage:
    python scripts/afnd_frequency_analysis.py \
        --afnd data/raw/afnd/afnd.tsv \
        --auroc results/per_allele_auroc_v3.csv \
        --motif data/processed/motif_distinctiveness.csv \
        --region-map data/processed/afnd_country_regions.csv \
        --out results/afnd_frequency.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Country -> region. UN geoscheme at the sub-continental level, collapsed to the
# granularity the frequency data can support. Written out on first run so it can be
# audited and edited; the file wins over this default if present.
DEFAULT_REGIONS: dict[str, str] = {
    # Europe
    "Austria": "Europe", "Belgium": "Europe", "Bulgaria": "Europe",
    "Croatia": "Europe", "Czech": "Europe", "Denmark": "Europe",
    "England": "Europe", "Estonia": "Europe", "Finland": "Europe",
    "France": "Europe", "Georgia": "Europe", "Germany": "Europe",
    "Greece": "Europe", "Hungary": "Europe", "Iceland": "Europe",
    "Ireland": "Europe", "Italy": "Europe", "Latvia": "Europe",
    "Lithuania": "Europe", "Macedonia": "Europe", "Netherlands": "Europe",
    "Norway": "Europe", "Poland": "Europe", "Portugal": "Europe",
    "Romania": "Europe", "Russia": "Europe", "Serbia": "Europe",
    "Slovakia": "Europe", "Slovenia": "Europe", "Spain": "Europe",
    "Sweden": "Europe", "Switzerland": "Europe", "Ukraine": "Europe",
    "United Kingdom": "Europe", "Wales": "Europe", "Scotland": "Europe",
    "Azores": "Europe", "Belarus": "Europe", "Bosnia": "Europe",
    "Armenia": "Europe", "Cyprus": "Europe", "Sardinia": "Europe",
    "Corsica": "Europe", "Albania": "Europe", "Moldova": "Europe",
    # Sub-Saharan Africa
    "Burkina Faso": "Africa", "Cameroon": "Africa", "Congo": "Africa",
    "Ethiopia": "Africa", "Gambia": "Africa", "Ghana": "Africa",
    "Guinea": "Africa", "Ivory Coast": "Africa", "Kenya": "Africa",
    "Mali": "Africa", "Nigeria": "Africa", "Rwanda": "Africa",
    "Senegal": "Africa", "South Africa": "Africa", "Sudan": "Africa",
    "Tanzania": "Africa", "Uganda": "Africa", "Zambia": "Africa",
    "Zimbabwe": "Africa", "Mozambique": "Africa", "Botswana": "Africa",
    "Benin": "Africa", "Burundi": "Africa", "Cape Verde": "Africa",
    "Central African": "Africa", "Chad": "Africa", "Eritrea": "Africa",
    "Gabon": "Africa", "Madagascar": "Africa", "Malawi": "Africa",
    "Namibia": "Africa", "Niger": "Africa", "Somalia": "Africa",
    # North Africa and West Asia
    "Algeria": "North Africa & West Asia", "Egypt": "North Africa & West Asia",
    "Iran": "North Africa & West Asia", "Iraq": "North Africa & West Asia",
    "Israel": "North Africa & West Asia", "Jordan": "North Africa & West Asia",
    "Kuwait": "North Africa & West Asia", "Lebanon": "North Africa & West Asia",
    "Libya": "North Africa & West Asia", "Morocco": "North Africa & West Asia",
    "Oman": "North Africa & West Asia", "Saudi": "North Africa & West Asia",
    "Syria": "North Africa & West Asia", "Tunisia": "North Africa & West Asia",
    "Turkey": "North Africa & West Asia", "Yemen": "North Africa & West Asia",
    "United Arab": "North Africa & West Asia", "Palestin": "North Africa & West Asia",
    # South and Central Asia
    "Afghanistan": "South & Central Asia", "Bangladesh": "South & Central Asia",
    "India": "South & Central Asia", "Kazakhstan": "South & Central Asia",
    "Nepal": "South & Central Asia", "Pakistan": "South & Central Asia",
    "Sri Lanka": "South & Central Asia", "Uzbekistan": "South & Central Asia",
    "Kyrgyzstan": "South & Central Asia", "Tajikistan": "South & Central Asia",
    "Mongolia": "South & Central Asia", "Bhutan": "South & Central Asia",
    # East and Southeast Asia
    "China": "East & Southeast Asia", "Hong Kong": "East & Southeast Asia",
    "Indonesia": "East & Southeast Asia", "Japan": "East & Southeast Asia",
    "Korea": "East & Southeast Asia", "Malaysia": "East & Southeast Asia",
    "Philippines": "East & Southeast Asia", "Singapore": "East & Southeast Asia",
    "Taiwan": "East & Southeast Asia", "Thailand": "East & Southeast Asia",
    "Vietnam": "East & Southeast Asia", "Cambodia": "East & Southeast Asia",
    "Laos": "East & Southeast Asia", "Myanmar": "East & Southeast Asia",
    # Americas
    "Argentina": "Americas", "Bolivia": "Americas", "Brazil": "Americas",
    "Canada": "Americas", "Chile": "Americas", "Colombia": "Americas",
    "Costa Rica": "Americas", "Cuba": "Americas", "Ecuador": "Americas",
    "Guatemala": "Americas", "Mexico": "Americas", "Paraguay": "Americas",
    "Peru": "Americas", "United States": "Americas", "USA": "Americas",
    "Uruguay": "Americas", "Venezuela": "Americas", "Jamaica": "Americas",
    "Trinidad": "Americas", "Nicaragua": "Americas", "Honduras": "Americas",
    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania",
    "Papua New Guinea": "Oceania", "Fiji": "Oceania", "Samoa": "Oceania",
    "Tonga": "Oceania", "Vanuatu": "Oceania", "New Caledonia": "Oceania",
    "Solomon": "Oceania", "Cook Islands": "Oceania",
}


def build_region_map(pops: pd.Series, path: Path) -> pd.DataFrame:
    """Longest-prefix match population label -> country -> region."""
    if path.exists():
        m = pd.read_csv(path)
        print(f"region map loaded from {path} ({len(m)} populations)")
        return m
    keys = sorted(DEFAULT_REGIONS, key=len, reverse=True)   # longest first
    rows = []
    for p in sorted(pops.unique()):
        region, country = None, None
        for k in keys:
            if p.startswith(k):
                region, country = DEFAULT_REGIONS[k], k
                break
        rows.append({"population": p, "country": country, "region": region})
    m = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(path, index=False)
    print(f"region map WRITTEN to {path} — inspect and correct it, then rerun")
    return m


def partial_spearman(x, y, z):
    xr, yr, zr = (stats.rankdata(v) for v in (x, y, z))
    Z = np.column_stack([np.ones_like(zr), zr])
    rx = xr - Z @ np.linalg.lstsq(Z, xr, rcond=None)[0]
    ry = yr - Z @ np.linalg.lstsq(Z, yr, rcond=None)[0]
    r, p = stats.pearsonr(rx, ry)
    return float(r), float(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--afnd", required=True)
    ap.add_argument("--auroc", required=True, help="per-allele AUROC table")
    ap.add_argument("--motif", default="data/processed/motif_distinctiveness.csv")
    ap.add_argument("--region-map",
                    default="data/processed/afnd_country_regions.csv")
    ap.add_argument("--min-n", type=int, default=50,
                    help="drop AFND populations sampled below this many individuals")
    ap.add_argument("--out", default="results/afnd_frequency.csv")
    args = ap.parse_args()

    af = pd.read_csv(args.afnd, sep="\t")
    af = af[af.group == "hla"]
    af = af[af.gene.isin(["A", "B", "C"])]
    af = af.dropna(subset=["alleles_over_2n"])
    # n and the frequency arrive as strings in the published TSV — thousands
    # separators in n, and occasional blanks in both
    for c in ("n", "alleles_over_2n"):
        af[c] = pd.to_numeric(
            af[c].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce")
    af = af.dropna(subset=["alleles_over_2n", "n"])
    af = af[af.n >= args.min_n]
    af["allele_full"] = "HLA-" + af.allele.astype(str)
    print(f"AFND: {len(af):,} rows, {af.allele_full.nunique():,} alleles, "
          f"{af.population.nunique():,} populations (n >= {args.min_n})")

    rmap = build_region_map(af.population, Path(args.region_map))
    af = af.merge(rmap, on="population", how="left")

    unmapped = af[af.region.isna()]
    if len(unmapped):
        top = (unmapped.population.value_counts().head(8).index.tolist())
        print(f"\n  {unmapped.population.nunique()} populations unmapped "
              f"({len(unmapped):,} rows, "
              f"{len(unmapped) / len(af):.1%}), e.g. {top[:5]}")
        print("  these are DROPPED; add them to the region map to include them")
    af = af.dropna(subset=["region"])

    # sample-size-weighted mean frequency per allele per region
    af["w"] = af.n * af.alleles_over_2n
    g = af.groupby(["allele_full", "region"]).agg(
        wsum=("w", "sum"), nsum=("n", "sum"), npop=("population", "nunique"))
    g["freq"] = g.wsum / g.nsum
    freq = g.reset_index().pivot(index="allele_full", columns="region",
                                 values="freq").fillna(0.0)

    auroc = pd.read_csv(args.auroc)
    col = next((c for c in ("auroc", "val_auroc", "model_auroc")
                if c in auroc.columns), None)
    if col is None:
        raise SystemExit(f"no AUROC column in {args.auroc}: {list(auroc.columns)}")
    d = auroc[["allele", col]].rename(columns={col: "auroc"}).merge(
        freq, left_on="allele", right_index=True, how="inner")
    print(f"\n{len(d)} of {len(auroc)} alleles matched to AFND frequencies")
    if len(d) < 20:
        raise SystemExit("too few matched alleles to analyse")

    regions = [c for c in freq.columns]
    d["global_freq"] = d[regions].mean(axis=1)

    print("\n=== does frequency in each region predict per-allele AUROC? ===")
    print(f"{'region':<28} {'rho':>8} {'p':>8}   {'alleles present':>16}")
    rows = []
    for r in regions:
        present = (d[r] > 0).sum()
        rr = stats.spearmanr(d[r], d.auroc)
        rows.append({"region": r, "rho": round(rr[0], 3), "p": round(rr[1], 4),
                     "n_alleles_present": int(present)})
        print(f"{r:<28} {rr[0]:>+8.3f} {rr[1]:>8.4f}   {present:>16}")

    if "Europe" not in regions:
        raise SystemExit("no Europe region in the map — cannot run the enrichment test")

    # Europe enrichment: log ratio of European frequency to frequency elsewhere
    others = [r for r in regions if r != "Europe"]
    eps = 1e-4
    d["elsewhere"] = d[others].mean(axis=1)
    d["europe_enrichment"] = np.log2((d.Europe + eps) / (d.elsewhere + eps))

    print("\n=== the direct question ===")
    re_ = stats.spearmanr(d.europe_enrichment, d.auroc)
    print(f"  Europe enrichment vs AUROC        rho {re_[0]:+.3f}  p {re_[1]:.4f}")
    rg = stats.spearmanr(d.global_freq, d.auroc)
    print(f"  global frequency  vs AUROC        rho {rg[0]:+.3f}  p {rg[1]:.4f}")
    pe = partial_spearman(d.Europe, d.auroc, d.global_freq)
    print(f"  European freq | global freq       rho {pe[0]:+.3f}  p {pe[1]:.4f}")

    if Path(args.motif).exists():
        m = pd.read_csv(args.motif)[["allele", "nn_dist"]]
        dm = d.merge(m, on="allele", how="inner")
        if len(dm) > 20:
            rm = stats.spearmanr(dm.nn_dist, dm.auroc)
            pm = partial_spearman(dm.nn_dist, dm.auroc, dm.europe_enrichment)
            print(f"\n=== against the established motif-distance finding ===")
            print(f"  motif distance vs AUROC           rho {rm[0]:+.3f}  "
                  f"p {rm[1]:.4f}")
            print(f"  motif distance | Europe enrichment rho {pm[0]:+.3f}  "
                  f"p {pm[1]:.4f}")
            print("  (if motif distance survives and enrichment does not, the")
            print("   deficit is about motif structure rather than ancestry)")

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    d.to_csv(str(Path(args.out).with_name(
        Path(args.out).stem + "_per_allele.csv")), index=False)

    print("""
LIMITATIONS, to be reported with any figure from this analysis.

AFND's population sample is not representative: European and North American
populations are heavily overrepresented in the database itself, so European frequency
is estimated from more data than frequency elsewhere. The independent variable
carries some of the bias the analysis is meant to detect.

This is ecological, at allele level. It says nothing about individuals or outcomes.

The region mapping is a modelling choice, written to a CSV so it can be audited. A
different defensible mapping could give a different answer, and the analysis should
be rerun under at least one alternative before any positive result is reported.""")
    print(f"\nWrote {args.out} and the per-allele table")


if __name__ == "__main__":
    main()