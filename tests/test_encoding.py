from pmhcpresent.io.peptides import encode_sequence, encode_batch, PAD_IDX, UNK_IDX, AA_TO_IDX


def test_padding_and_length():
    enc = encode_sequence("SIINFEKL", max_len=15)
    assert enc.shape == (15,)
    assert enc[8:].tolist() == [PAD_IDX] * 7


def test_unknown_residue():
    enc = encode_sequence("SIBNFEKL", max_len=10)  # B is non-standard
    assert enc[2] == UNK_IDX


def test_known_residue_mapping():
    enc = encode_sequence("A", max_len=3)
    assert enc[0] == AA_TO_IDX["A"]


def test_batch_shape():
    batch = encode_batch(["SIINFEKL", "GILGFVFTL"], max_len=12)
    assert batch.shape == (2, 12)
