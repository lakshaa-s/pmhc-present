from pathlib import Path
from pmhcpresent.io.netmhcpan import parse_netmhcpan_file

FIXTURE = Path(__file__).parent / "fixtures" / "netmhcpan_sample.txt"


def test_parses_all_rows():
    recs = parse_netmhcpan_file(FIXTURE)
    assert len(recs) == 3


def test_binder_classification():
    recs = parse_netmhcpan_file(FIXTURE)
    by_pep = {r.peptide: r for r in recs}
    assert by_pep["SLLMWITQV"].bind_level == "SB"
    assert by_pep["SLLMWITQV"].is_binder
    assert by_pep["GILGFVFTL"].bind_level == "WB"
    assert by_pep["AAAWILKDV"].bind_level is None
    assert not by_pep["AAAWILKDV"].is_binder


def test_numeric_fields():
    recs = parse_netmhcpan_file(FIXTURE)
    r = {x.peptide: x for x in recs}["SLLMWITQV"]
    assert abs(r.score_el - 0.9671) < 1e-6
    assert abs(r.rank_el - 0.041) < 1e-6
    assert abs(r.aff_nm - 23.45) < 1e-6
    assert r.allele == "HLA-A*02:01"
