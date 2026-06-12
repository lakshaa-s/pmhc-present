#!/usr/bin/env bash
# Run NetMHCpan-4.1 on Beta. Output is parsed locally by p11.io.netmhcpan.
# Usage: ./run_netmhcpan.sh peptides.pep HLA-A02:01 out.txt
set -euo pipefail
PEPTIDES="${1:?peptide file}"
ALLELE="${2:?allele, e.g. HLA-A02:01}"
OUT="${3:?output path}"

# -BA adds binding-affinity columns; -xls writes tab-separated output for big runs.
netMHCpan -p -BA -a "$ALLELE" -f "$PEPTIDES" > "$OUT"
echo "Wrote $OUT"
