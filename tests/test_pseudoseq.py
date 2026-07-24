import pytest

from pmhcpresent.io.pseudoseq import (
    PseudoSequenceMap,
    load_pseudosequences,
    normalize_allele,
)


def test_normalize_equivalent_forms():
    # the three common spellings of the same allele collapse to one key
    forms = ["HLA-A*02:01", "HLA-A02:01", "A0201", "A*02:01"]
    normed = {normalize_allele(f) for f in forms}
    assert len(normed) == 1
    assert normed.pop() == "HLA-A*02:01"


def test_map_lookup_with_name_variants():
    m = PseudoSequenceMap({"HLA-A*02:01": "Y" * 34})
    assert m.get("HLA-A02:01") == "Y" * 34   # different spelling still hits
    assert m.get("A0201") == "Y" * 34
    assert m.get("HLA-B*07:02") is None
    assert "HLA-A02:01" in m


def test_load_from_file(tmp_path):
    f = tmp_path / "pseudo.dat"
    f.write_text(
        "# comment line\n"
        "HLA-A*02:01 " + "Y" * 34 + "\n"
        "HLA-B*07:02 " + "F" * 34 + "\n"
    )
    m = load_pseudosequences(f)
    assert len(m) == 2
    assert m.pseudoseq_len == 34
    assert m.get("A0201") == "Y" * 34


def test_empty_file_raises(tmp_path):
    f = tmp_path / "empty.dat"
    f.write_text("# only a comment\n")
    with pytest.raises(ValueError):
        load_pseudosequences(f)
