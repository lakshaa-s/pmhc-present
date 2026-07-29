"""hamming_cluster must not depend on PYTHONHASHSEED.

Regression test for a silent reproducibility bug: `uniq` was built from a set
comprehension, and because Python randomises string hashing per process, set
iteration order varied between runs. Greedy single-linkage is order-dependent,
so this changed cluster *membership*, not merely the ids -- two runs of the same
command produced splits differing by ~86 rows in 167k, with no error raised.
"""

import subprocess
import sys

SNIPPET = (
    "from pmhcpresent.eval.splits import hamming_cluster;"
    "peps=['SIINFEKL','SIINFEKM','SIINFEKA','GILGFVFTL','GILGFVFTV','AAAAAAAAA',"
    "'AAAAAAAAB','KLVEKVLAV','KLVEKVLAI','LLLETLPEL'];"
    "alls=['A']*5+['B']*5;"
    "print(list(hamming_cluster(peps, alls)))"
)


def _run(hashseed):
    out = subprocess.run(
        [sys.executable, "-c", SNIPPET],
        capture_output=True, text=True, check=True,
        env={"PYTHONHASHSEED": str(hashseed), "PATH": "/usr/bin:/bin"},
    )
    return out.stdout.strip()


def test_clustering_is_hashseed_independent():
    results = {_run(seed) for seed in (0, 1, 2, 12345)}
    assert len(results) == 1, f"clustering varies with PYTHONHASHSEED: {results}"
