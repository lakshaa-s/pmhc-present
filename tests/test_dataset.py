import pytest

pytest.importorskip("torch")

from pmhcpresent.train import PeptideMHCDataset


def test_dataset_shapes():
    peps = ["SIINFEKL", "GILGFVFTL", "SLLMWITQV"]
    pseuds = ["Y" * 34] * 3
    labels = [1, 0, 1]
    ds = PeptideMHCDataset(peps, pseuds, labels, max_pep_len=15, pseudoseq_len=34)
    assert len(ds) == 3
    pep, mhc, y = ds[0]
    assert pep.shape == (15,)
    assert mhc.shape == (34,)
    assert y.item() in (0.0, 1.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        PeptideMHCDataset(["AAA"], ["Y" * 34], [1, 0])


def test_from_frame_drops_missing_alleles(capsys):
    import pandas as pd

    from pmhcpresent.io.pseudoseq import PseudoSequenceMap

    df = pd.DataFrame({
        "peptide": ["SIINFEKL", "GILGFVFTL"],
        "allele": ["HLA-A*02:01", "HLA-Z*99:99"],   # second has no pseudoseq
        "label": [1, 0],
    })
    pmap = PseudoSequenceMap({"HLA-A*02:01": "Y" * 34})
    ds = PeptideMHCDataset.from_frame(df, pmap)
    assert len(ds) == 1
    assert "dropped 1" in capsys.readouterr().out
