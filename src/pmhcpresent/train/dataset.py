"""Dataset wiring peptides + allele pseudosequences + labels into the NN.

Encodings are precomputed once in ``__init__`` (cheap, numpy) so ``__getitem__`` is a
tensor slice. ``__getitem__`` returns only the three tensors the model trains on
(peptide, mhc, label); allele and stratum labels are kept as dataset attributes so
``evaluate`` can align ordered predictions back to them without stuffing Python
objects through the DataLoader collate.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from pmhcpresent.io.peptides import encode_batch
from pmhcpresent.io.pseudoseq import PseudoSequenceMap


class PeptideMHCDataset(Dataset):
    def __init__(
        self,
        peptides: list[str],
        pseudoseqs: list[str],
        labels,
        *,
        max_pep_len: int = 15,
        pseudoseq_len: int = 34,
        alleles: list[str] | None = None,
        strata=None,
    ):
        if not (len(peptides) == len(pseudoseqs) == len(labels)):
            raise ValueError("peptides, pseudoseqs, labels must be the same length")

        self.pep = torch.from_numpy(encode_batch(peptides, max_pep_len))
        self.mhc = torch.from_numpy(encode_batch(pseudoseqs, pseudoseq_len))
        self.labels = np.asarray(labels, dtype=np.float32)
        self.y = torch.from_numpy(self.labels)

        self.peptides = list(peptides)
        self.alleles = list(alleles) if alleles is not None else None
        self.strata = np.asarray(strata) if strata is not None else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return self.pep[i], self.mhc[i], self.y[i]

    @classmethod
    def from_frame(
        cls,
        df,
        pseudo_map: PseudoSequenceMap,
        *,
        peptide_col: str = "peptide",
        allele_col: str = "allele",
        label_col: str = "label",
        stratum_col: str | None = None,
        max_pep_len: int = 15,
        pseudoseq_len: int = 34,
        drop_missing: bool = True,
    ) -> "PeptideMHCDataset":
        """Build from a DataFrame, mapping alleles → pseudosequences.

        Rows whose allele has no pseudosequence are dropped (``drop_missing``) with the
        count reported, rather than silently mis-encoded.
        """
        peptides, pseuds, labels, alleles, strata = [], [], [], [], []
        missing = 0
        strat_vals = df[stratum_col].tolist() if stratum_col else [None] * len(df)

        for pep, allele, label, strat in zip(
            df[peptide_col], df[allele_col], df[label_col], strat_vals
        ):
            ps = pseudo_map.get(str(allele))
            if ps is None:
                missing += 1
                if drop_missing:
                    continue
                raise KeyError(f"No pseudosequence for allele {allele!r}")
            peptides.append(str(pep))
            pseuds.append(ps)
            labels.append(label)
            alleles.append(str(allele))
            strata.append(strat)

        if missing and drop_missing:
            print(f"[PeptideMHCDataset] dropped {missing} rows with no pseudosequence")

        return cls(
            peptides, pseuds, labels,
            max_pep_len=max_pep_len, pseudoseq_len=pseudoseq_len,
            alleles=alleles,
            strata=strata if stratum_col else None,
        )
