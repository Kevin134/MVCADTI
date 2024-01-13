import pdb
import torch
import numpy as np
from chemprop.features import mol2graph
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from torch.utils.data import Dataset


CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARISOSMISET = {
     "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
     "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
     "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
     "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
     "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
     "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
     "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
     "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64
}

class CustomDataSet(Dataset):
    def __init__(self, args, data):
        self.data = data
        self.protein_max_length = args.protein_max_length

    def __getitem__(self, idx):
        return self.encode(self.data[idx])

    def encode(self, item):
        smiles, protein, label = item
        protein = protein[: self.protein_max_length]
        protein_mask = [1] * len(protein)
        label = int(float(label))

        return smiles, protein, protein_mask, label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        batch_smiles = []
        batch_protein, batch_protein_mask = [], []
        batch_labels = []

        for smiles, protein, protein_mask, label in batch:
            batch_smiles.append(smiles)
            batch_protein.append([CHARPROTSET[x] for x in protein])
            # batch_protein.append(protein)
            batch_protein_mask.append(protein_mask)
            batch_labels.append(label)

        batch_mol_graph = mol2graph(batch_smiles)  # 转换为分子图

        """ amino acid sequence """
        batch_protein_str = torch.tensor(sequence_padding(batch_protein), dtype=torch.long)
        batch_protein_mask = torch.tensor(sequence_padding(batch_protein_mask), dtype=torch.long)

        """ label """
        batch_label = torch.tensor(batch_labels, dtype=torch.long)

        return {
            "mol_graph": batch_mol_graph,
            "protein": batch_protein_str,
            "protein_mask": batch_protein_mask,
            "label": batch_label
        }

    @staticmethod
    def collate_fn_ngram(batch):
        batch_smiles, batch_smiles_ids = [], []
        batch_protein, batch_protein_mask = [], []
        batch_labels = []

        for smiles, protein, protein_mask, label in batch:
            batch_smiles.append(smiles)
            batch_smiles_ids.append([CHARISOSMISET[x] for x in smiles])
            # batch_protein.append([CHARPROTSET[x] for x in protein])
            batch_protein.append(protein)
            batch_protein_mask.append(protein_mask)
            batch_labels.append(label)

        """ Smiles """
        batch_mol_graph = mol2graph(batch_smiles)  # 转换为分子图
        batch_smiles_mask = []
        for item in batch_smiles_ids:
            batch_smiles_mask.append([1] * len(item))
        batch_smiles_ids = torch.tensor(sequence_padding(batch_smiles_ids), dtype=torch.long)
        batch_smiles_mask = torch.tensor(sequence_padding(batch_smiles_mask), dtype=torch.long)

        """ amino acid sequence """
        batch_protein_str = torch.tensor(sequence_padding(batch_protein), dtype=torch.long)
        batch_protein_mask = torch.tensor(sequence_padding(batch_protein_mask), dtype=torch.long)

        """ label """
        batch_label = torch.tensor(batch_labels, dtype=torch.long)

        return {
            "mol_graph": batch_mol_graph,
            "smiles": batch_smiles_ids,
            "smiles_mask": batch_smiles_mask,
            "protein": batch_protein_str,
            "protein_mask": batch_protein_mask,
            "label": batch_label
        }


class InferenceDataSet(Dataset):
    def __init__(self, args, data):
        self.data = data
        self.protein_max_length = args.protein_max_length

    def __getitem__(self, idx):
        return self.encode(self.data[idx])

    def encode(self, item):
        d_id, smiles, p_id, src_protein, protein = item
        protein = protein[: self.protein_max_length]
        protein_mask = [1] * len(protein)

        return d_id, smiles, p_id, src_protein, protein, protein_mask

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn_ngram(batch):
        batch_drug_id, batch_smiles, batch_smiles_ids = [], [], []
        batch_protein_id, batch_protein, batch_protein_mask = [], [], []
        batch_src_protein = []

        for d_id, smiles, p_id, src_protein, protein, protein_mask in batch:
            batch_drug_id.append(d_id)
            batch_smiles.append(smiles)
            batch_smiles_ids.append([CHARISOSMISET[x] for x in smiles])
            batch_protein_id.append(p_id)
            batch_protein.append(protein)
            batch_protein_mask.append(protein_mask)
            batch_src_protein.append(src_protein)

        """ Smiles """
        batch_mol_graph = mol2graph(batch_smiles)  # 转换为分子图
        batch_smiles_mask = []
        for item in batch_smiles_ids:
            batch_smiles_mask.append([1] * len(item))
        batch_smiles_ids = torch.tensor(sequence_padding(batch_smiles_ids), dtype=torch.long)
        batch_smiles_mask = torch.tensor(sequence_padding(batch_smiles_mask), dtype=torch.long)

        """ amino acid sequence """
        batch_protein_str = torch.tensor(sequence_padding(batch_protein), dtype=torch.long)
        batch_protein_mask = torch.tensor(sequence_padding(batch_protein_mask), dtype=torch.long)

        return {
            "mol_graph": batch_mol_graph,
            "src_smiles": batch_smiles,
            "smiles": batch_smiles_ids,
            "smiles_mask": batch_smiles_mask,
            "protein": batch_protein_str,
            "protein_mask": batch_protein_mask,
            "src_proteins": batch_src_protein,
            "d_id": batch_drug_id,
            "p_id": batch_protein_id
        }


def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs, dtype='int64')