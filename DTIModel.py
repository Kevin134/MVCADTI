# -*- coding: utf-8 -*-
import pdb

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MPNN import MPN
from models.ProteinEncoder import ProteinEncoder, AttentionPooling, Encoder
from models.Trm import ProbAttention, ProbMultiHeadAttention
from models.SmilesCNN import SmilesEncoder
from models.MHA import MultiHeadAttention
from models.utilities import LSTMPooling


class CrossSeNetAttention(nn.Module):
    def __init__(self, input_dim, reduction_ratio=16):
        super().__init__()
        self.squeeze = nn.Linear(input_dim, input_dim // reduction_ratio)
        self.excitation = nn.Linear(input_dim // reduction_ratio, input_dim)
        self.attention_polling = AttentionPooling(input_dim)

    def forward(self, src, tgt, tgt_mask):
        # batch_size, seq_len, input_dim = x.size()
        # squeeze_output = torch.mean(x, dim=1)  # Global average pooling
        squeeze_output = self.attention_polling(tgt, tgt_mask)
        squeeze_output = F.relu(self.squeeze(squeeze_output))
        excitation_output = torch.sigmoid(self.excitation(squeeze_output)).unsqueeze(1)
        return src * excitation_output

class LocalAttention(nn.Module):
    def __init__(self, input_dim, window_size=5):
        super(LocalAttention, self).__init__()
        self.input_dim = input_dim
        self.window_size = window_size

        # 定义局部注意力权重的学习参数
        self.query_projection = nn.Linear(input_dim, input_dim)
        self.key_projection = nn.Linear(input_dim, input_dim)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        attention_scores = torch.zeros(seq_len, seq_len).to(x.device)

        for t in range(seq_len):
            start = max(0, t - self.window_size)
            end = min(seq_len, t + self.window_size)

            query = self.query_projection(x[:, t, :]).unsqueeze(1)  # (batch_size, 1, input_dim)
            keys = self.key_projection(x[:, start:end, :])  # (batch_size, window_size*2+1, input_dim)

            scores = torch.matmul(query, keys.permute(0, 2, 1))  # (batch_size, 1, window_size*2+1)
            attention_scores[t, start:end] = scores.squeeze(1)

        # 如果存在mask，则将mask应用到注意力分数上
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=1)
        output = torch.matmul(attention_weights.unsqueeze(1), x).squeeze(1)

        return output


def swiglu(x, dim):
    x = torch.chunk(x, 2, dim=dim)
    return F.silu(x[0]) * x[1]


class Model(nn.Module):
    def __init__(self, args, vocab_size=None):
        super().__init__()
        """ protein """
        # self.protein_encoder = ProteinEncoder(
        #     vocab_size=vocab_size, hidden_dim=args.protein_hidden_size, conv_nums=args.protein_conv,
        #     kernel_size=[3, 7, 11, 17, 23]
        # )
        self.protein_encoder = Encoder(
            args=args, vocab_size=vocab_size, hid_dim=args.protein_hidden_size, n_layers=3, kernel_size=7, dropout=0.1
        )

        """" drug """
        self.drug_encoder = MPN(args)
        self.smiles_encoder = SmilesEncoder(args, 70, hid_dim=args.hidden_size, n_layers=3, kernel_size=5, dropout=0.1)

        self.atom_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.bond_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.mol_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)

        """ interaction """
        self.atom_smiles_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        self.bond_smiles_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        self.mol_protein_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        # self.protein_mol_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)

        self.mol_layerNorm = nn.LayerNorm(args.hidden_size)
        self.protein_layerNorm = nn.LayerNorm(args.hidden_size)

        self.protein_attention_pooling = AttentionPooling(args.hidden_size)
        self.mol_attention_pooling = AttentionPooling(args.hidden_size)

        self.interaction_layer = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size * 4, args.hidden_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size, 2)
        )

    def forward(self, mol_graph, smiles, smiles_mask, protein, protein_mask):
        # mol_embed, mol_mask = self.drug_encoder(mol_graph)
        """ drug """
        # 每一层的node编码和bond编码
        atom_embed, atom_mask, bond_embed, bond_mask = self.drug_encoder(mol_graph)  # Graph
        smiles_embed = self.smiles_encoder(smiles)  # Smiles

        atom_cross_embed = self.atom_smiles_attention(
            atom_embed, smiles_embed, smiles_embed, key_padding_mask=(smiles_mask == 0)
        )[0]
        bond_cross_embed = self.bond_smiles_attention(
            bond_embed, smiles_embed, smiles_embed, key_padding_mask=(smiles_mask == 0)
        )[0]

        atom_embed = self.atom_mlp(torch.concat([atom_embed, atom_cross_embed], dim=-1))
        bond_embed = self.bond_mlp(torch.concat([bond_embed, bond_cross_embed], dim=-1))
        mol_embed = self.mol_layerNorm(self.mol_mlp(torch.concat([atom_embed, bond_embed], dim=-1)))

        """ protein """
        protein_embed = self.protein_layerNorm(self.protein_encoder(protein, protein_mask))  # sequence

        """ interaction """
        mol_cross_embed = self.mol_protein_attention(
            mol_embed, protein_embed, protein_embed, key_padding_mask=(protein_mask == 0)
        )[0]
        protein_cross_embed = self.mol_protein_attention(
            protein_embed, mol_embed, mol_embed, key_padding_mask=(atom_mask == 0)
        )[0]

        mol_embed = self.mol_attention_pooling(mol_cross_embed, atom_mask)
        protein_embed = self.protein_attention_pooling(protein_cross_embed, protein_mask)

        pair_embed = torch.cat([mol_embed, protein_embed], dim=-1)
        output = self.interaction_layer(pair_embed)

        return output


class Model_Interpretation(nn.Module):
    def __init__(self, args, vocab_size=None):
        super().__init__()
        """ protein """
        # self.protein_encoder = ProteinEncoder(
        #     vocab_size=vocab_size, hidden_dim=args.protein_hidden_size, conv_nums=args.protein_conv,
        #     kernel_size=[3, 7, 11, 17, 23]
        # )
        self.protein_encoder = Encoder(
            args=args, vocab_size=vocab_size, hid_dim=args.protein_hidden_size, n_layers=3, kernel_size=7, dropout=0.1
        )

        """" drug """
        self.drug_encoder = MPN(args)
        self.smiles_encoder = SmilesEncoder(args, 70, hid_dim=args.hidden_size, n_layers=3, kernel_size=5, dropout=0.1)

        self.atom_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.bond_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.mol_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)

        """ interaction """
        self.atom_smiles_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        self.bond_smiles_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        self.mol_protein_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        self.protein_mol_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)

        self.mol_layerNorm = nn.LayerNorm(args.hidden_size)
        self.protein_layerNorm = nn.LayerNorm(args.hidden_size)

        self.protein_attention_pooling = AttentionPooling(args.hidden_size)
        self.mol_attention_pooling = AttentionPooling(args.hidden_size)

        self.interaction_layer = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size * 4, args.hidden_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size, 2)
        )

    def forward(self, mol_graph, smiles, smiles_mask, protein, protein_mask):
        # mol_embed, mol_mask = self.drug_encoder(mol_graph)
        """ drug """
        # 每一层的node编码和bond编码
        atom_embed, atom_mask, bond_embed, bond_mask = self.drug_encoder(mol_graph)  # Graph
        smiles_embed = self.smiles_encoder(smiles)  # Smiles

        atom_cross_embed = self.atom_smiles_attention(
            atom_embed, smiles_embed, smiles_embed, key_padding_mask=(smiles_mask == 0)
        )[0]
        bond_cross_embed = self.bond_smiles_attention(
            bond_embed, smiles_embed, smiles_embed, key_padding_mask=(smiles_mask == 0)
        )[0]

        atom_embed = self.atom_mlp(torch.concat([atom_embed, atom_cross_embed], dim=-1))
        bond_embed = self.bond_mlp(torch.concat([bond_embed, bond_cross_embed], dim=-1))
        mol_embed = self.mol_layerNorm(self.mol_mlp(torch.concat([atom_embed, bond_embed], dim=-1)))

        """ protein """
        protein_embed = self.protein_layerNorm(self.protein_encoder(protein, protein_mask))  # sequence

        """ interaction """
        mol_cross_embed, attention_weight1 = self.mol_protein_attention(
            mol_embed, protein_embed, protein_embed, key_padding_mask=(protein_mask == 0)
        )
        protein_cross_embed, attention_weight2 = self.protein_mol_attention(
            protein_embed, mol_embed, mol_embed, key_padding_mask=(atom_mask == 0)
        )

        mol_embed = self.mol_attention_pooling(mol_cross_embed, atom_mask)
        protein_embed = self.protein_attention_pooling(protein_cross_embed, protein_mask)

        pair_embed = torch.cat([mol_embed, protein_embed], dim=-1)
        output = self.interaction_layer(pair_embed)

        return output, attention_weight1, attention_weight2

class Model_Interpretation_2(nn.Module):
    def __init__(self, args, vocab_size=None):
        super().__init__()
        """ protein """
        # self.protein_encoder = ProteinEncoder(
        #     vocab_size=vocab_size, hidden_dim=args.protein_hidden_size, conv_nums=args.protein_conv,
        #     kernel_size=[3, 7, 11, 17, 23]
        # )
        self.protein_encoder = Encoder(
            args=args, vocab_size=vocab_size, hid_dim=args.protein_hidden_size, n_layers=3, kernel_size=7, dropout=0.1
        )

        """" drug """
        self.drug_encoder = MPN(args)
        self.smiles_encoder = SmilesEncoder(args, 70, hid_dim=args.hidden_size, n_layers=3, kernel_size=5, dropout=0.1)

        self.atom_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.bond_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.mol_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)

        """ interaction """
        self.atom_smiles_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        self.bond_smiles_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        self.mol_protein_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        # self.protein_mol_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)

        self.mol_layerNorm = nn.LayerNorm(args.hidden_size)
        self.protein_layerNorm = nn.LayerNorm(args.hidden_size)

        self.protein_attention_pooling = AttentionPooling(args.hidden_size)
        self.mol_attention_pooling = AttentionPooling(args.hidden_size)

        self.interaction_layer = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size * 4, args.hidden_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size, 2)
        )

    def forward(self, mol_graph, smiles, smiles_mask, protein, protein_mask):
        # mol_embed, mol_mask = self.drug_encoder(mol_graph)
        """ drug """
        # 每一层的node编码和bond编码
        atom_embed, atom_mask, bond_embed, bond_mask = self.drug_encoder(mol_graph)  # Graph
        smiles_embed = self.smiles_encoder(smiles)  # Smiles

        atom_cross_embed, atom_cross_weights = self.atom_smiles_attention(
            atom_embed, smiles_embed, smiles_embed, key_padding_mask=(smiles_mask == 0)
        )
        # atom_cross_embed = self.atom_smiles_attention(
        #     atom_embed, smiles_embed, smiles_embed, key_padding_mask=(smiles_mask == 0)
        # )[0]
        bond_cross_embed, bond_cross_weights = self.bond_smiles_attention(
            bond_embed, smiles_embed, smiles_embed, key_padding_mask=(smiles_mask == 0)
        )

        atom_embed = self.atom_mlp(torch.concat([atom_embed, atom_cross_embed], dim=-1))
        bond_embed = self.bond_mlp(torch.concat([bond_embed, bond_cross_embed], dim=-1))
        mol_embed = self.mol_layerNorm(self.mol_mlp(torch.concat([atom_embed, bond_embed], dim=-1)))

        """ protein """
        protein_embed = self.protein_layerNorm(self.protein_encoder(protein, protein_mask))  # sequence

        """ interaction """
        mol_cross_embed, attention_weight1 = self.mol_protein_attention(
            mol_embed, protein_embed, protein_embed, key_padding_mask=(protein_mask == 0)
        )
        protein_cross_embed, attention_weight2 = self.mol_protein_attention(
            protein_embed, mol_embed, mol_embed, key_padding_mask=(atom_mask == 0)
        )

        mol_embed = self.mol_attention_pooling(mol_cross_embed, atom_mask)
        protein_embed = self.protein_attention_pooling(protein_cross_embed, protein_mask)

        pair_embed = torch.cat([mol_embed, protein_embed], dim=-1)
        output = self.interaction_layer(pair_embed)

        return output, atom_cross_weights, bond_cross_weights


class Model_A(nn.Module):
    def __init__(self, args, vocab_size=None):
        super().__init__()
        """ protein """
        # self.protein_encoder = ProteinEncoder(
        #     vocab_size=vocab_size, hidden_dim=args.protein_hidden_size, conv_nums=args.protein_conv,
        #     kernel_size=[3, 7, 11, 17, 23]
        # )
        self.protein_encoder = Encoder(
            args=args, vocab_size=vocab_size, hid_dim=args.protein_hidden_size, n_layers=3, kernel_size=7, dropout=0.1
        )

        """" drug """
        self.drug_encoder = MPN(args)
        self.smiles_encoder = SmilesEncoder(args, 70, hid_dim=args.hidden_size, n_layers=3, kernel_size=5, dropout=0.1)

        self.atom_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.bond_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.mol_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)

        """ interaction """
        self.atom_smiles_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        # self.bond_smiles_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        self.mol_protein_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)

        self.mol_layerNorm = nn.LayerNorm(args.hidden_size)
        self.protein_layerNorm = nn.LayerNorm(args.hidden_size)

        self.protein_attention_pooling = AttentionPooling(args.hidden_size)
        self.mol_attention_pooling = AttentionPooling(args.hidden_size)

        self.interaction_layer = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size * 4, args.hidden_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size, 2)
        )

    def forward(self, mol_graph, smiles, smiles_mask, protein, protein_mask):
        # mol_embed, mol_mask = self.drug_encoder(mol_graph)
        """ drug """
        # 每一层的node编码和bond编码
        atom_embed, atom_mask, _, _ = self.drug_encoder(mol_graph)  # Graph
        smiles_embed = self.smiles_encoder(smiles)  # Smiles

        atom_cross_embed = self.atom_smiles_attention(
            atom_embed, smiles_embed, smiles_embed, key_padding_mask=(smiles_mask == 0)
        )[0]
        # bond_cross_embed = self.bond_smiles_attention(
        #     bond_embed, smiles_embed, smiles_embed, key_padding_mask=(smiles_mask == 0)
        # )[0]

        atom_embed = self.atom_mlp(torch.concat([atom_embed, atom_cross_embed], dim=-1))
        # bond_embed = self.bond_mlp(torch.concat([bond_embed, bond_cross_embed], dim=-1))
        # mol_embed = self.mol_layerNorm(self.mol_mlp(torch.concat([atom_embed, bond_embed], dim=-1)))
        mol_embed = self.mol_layerNorm(atom_embed)

        """ protein """
        protein_embed = self.protein_layerNorm(self.protein_encoder(protein, protein_mask))  # sequence

        """ interaction """
        mol_cross_embed = self.mol_protein_attention(
            mol_embed, protein_embed, protein_embed, key_padding_mask=(protein_mask == 0)
        )[0]
        protein_cross_embed = self.mol_protein_attention(
            protein_embed, mol_embed, mol_embed, key_padding_mask=(atom_mask == 0)
        )[0]

        mol_embed = self.mol_attention_pooling(mol_cross_embed, atom_mask)
        protein_embed = self.protein_attention_pooling(protein_cross_embed, protein_mask)

        pair_embed = torch.cat([mol_embed, protein_embed], dim=-1)
        output = self.interaction_layer(pair_embed)

        return output


class Model_B(nn.Module):
    def __init__(self, args, vocab_size=None):
        super().__init__()
        """ protein """
        # self.protein_encoder = ProteinEncoder(
        #     vocab_size=vocab_size, hidden_dim=args.protein_hidden_size, conv_nums=args.protein_conv,
        #     kernel_size=[3, 7, 11, 17, 23]
        # )
        self.protein_encoder = Encoder(
            args=args, vocab_size=vocab_size, hid_dim=args.protein_hidden_size, n_layers=3, kernel_size=7, dropout=0.1
        )

        """" drug """
        self.drug_encoder = MPN(args)
        self.smiles_encoder = SmilesEncoder(args, 70, hid_dim=args.hidden_size, n_layers=3, kernel_size=5, dropout=0.1)

        self.atom_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.bond_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.mol_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)

        """ interaction """
        self.atom_smiles_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        self.bond_smiles_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        self.mol_protein_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        # self.protein_mol_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)

        self.mol_layerNorm = nn.LayerNorm(args.hidden_size)
        self.protein_layerNorm = nn.LayerNorm(args.hidden_size)

        self.protein_attention_pooling = AttentionPooling(args.hidden_size)
        self.mol_attention_pooling = AttentionPooling(args.hidden_size)

        self.interaction_layer = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size * 4, args.hidden_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size, 2)
        )

    def forward(self, mol_graph, smiles, smiles_mask, protein, protein_mask):
        # mol_embed, mol_mask = self.drug_encoder(mol_graph)
        """ drug """
        # 每一层的node编码和bond编码
        _, _, bond_embed, bond_mask = self.drug_encoder(mol_graph)  # Graph
        smiles_embed = self.smiles_encoder(smiles)  # Smiles

        # atom_cross_embed = self.atom_smiles_attention(
        #     atom_embed, smiles_embed, smiles_embed, key_padding_mask=(smiles_mask == 0)
        # )[0]
        bond_cross_embed = self.bond_smiles_attention(
            bond_embed, smiles_embed, smiles_embed, key_padding_mask=(smiles_mask == 0)
        )[0]

        # atom_embed = self.atom_mlp(torch.concat([atom_embed, atom_cross_embed], dim=-1))
        bond_embed = self.bond_mlp(torch.concat([bond_embed, bond_cross_embed], dim=-1))
        mol_embed = self.mol_layerNorm(bond_embed)

        """ protein """
        protein_embed = self.protein_layerNorm(self.protein_encoder(protein, protein_mask))  # sequence

        """ interaction """
        mol_cross_embed = self.mol_protein_attention(
            mol_embed, protein_embed, protein_embed, key_padding_mask=(protein_mask == 0)
        )[0]
        protein_cross_embed = self.mol_protein_attention(
            protein_embed, mol_embed, mol_embed, key_padding_mask=(bond_mask == 0)
        )[0]

        mol_embed = self.mol_attention_pooling(mol_cross_embed, bond_mask)
        protein_embed = self.protein_attention_pooling(protein_cross_embed, protein_mask)

        pair_embed = torch.cat([mol_embed, protein_embed], dim=-1)
        output = self.interaction_layer(pair_embed)

        return output

class Model_C(nn.Module):
    def __init__(self, args, vocab_size=None):
        super().__init__()
        """ protein """
        # self.protein_encoder = ProteinEncoder(
        #     vocab_size=vocab_size, hidden_dim=args.protein_hidden_size, conv_nums=args.protein_conv,
        #     kernel_size=[3, 7, 11, 17, 23]
        # )
        self.protein_encoder = Encoder(
            args=args, vocab_size=vocab_size, hid_dim=args.protein_hidden_size, n_layers=3, kernel_size=7, dropout=0.1
        )

        """" drug """
        self.drug_encoder = MPN(args)
        self.smiles_encoder = SmilesEncoder(args, 70, hid_dim=args.hidden_size, n_layers=3, kernel_size=5, dropout=0.1)

        self.atom_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.bond_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.mol_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)

        """ interaction """
        self.atom_smiles_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        self.bond_smiles_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        self.mol_protein_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        # self.protein_mol_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)

        self.mol_layerNorm = nn.LayerNorm(args.hidden_size)
        self.protein_layerNorm = nn.LayerNorm(args.hidden_size)

        self.protein_attention_pooling = AttentionPooling(args.hidden_size)
        self.mol_attention_pooling = AttentionPooling(args.hidden_size)

        self.interaction_layer = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size * 4, args.hidden_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size, 2)
        )

    def forward(self, mol_graph, smiles, smiles_mask, protein, protein_mask):
        # mol_embed, mol_mask = self.drug_encoder(mol_graph)
        """ drug """
        # 每一层的node编码和bond编码
        atom_embed, atom_mask, bond_embed, bond_mask = self.drug_encoder(mol_graph)  # Graph
        # smiles_embed = self.smiles_encoder(smiles)  # Smiles

        # atom_cross_embed = self.atom_smiles_attention(
        #     atom_embed, smiles_embed, smiles_embed, key_padding_mask=(smiles_mask == 0)
        # )[0]
        # bond_cross_embed = self.bond_smiles_attention(
        #     bond_embed, smiles_embed, smiles_embed, key_padding_mask=(smiles_mask == 0)
        # )[0]

        # atom_embed = self.atom_mlp(torch.concat([atom_embed, atom_cross_embed], dim=-1))
        # bond_embed = self.bond_mlp(torch.concat([bond_embed, bond_cross_embed], dim=-1))
        mol_embed = self.mol_layerNorm(self.mol_mlp(torch.concat([atom_embed, bond_embed], dim=-1)))

        """ protein """
        protein_embed = self.protein_layerNorm(self.protein_encoder(protein, protein_mask))  # sequence

        """ interaction """
        mol_cross_embed = self.mol_protein_attention(
            mol_embed, protein_embed, protein_embed, key_padding_mask=(protein_mask == 0)
        )[0]
        protein_cross_embed = self.mol_protein_attention(
            protein_embed, mol_embed, mol_embed, key_padding_mask=(atom_mask == 0)
        )[0]

        mol_embed = self.mol_attention_pooling(mol_cross_embed, atom_mask)
        protein_embed = self.protein_attention_pooling(protein_cross_embed, protein_mask)

        pair_embed = torch.cat([mol_embed, protein_embed], dim=-1)
        output = self.interaction_layer(pair_embed)

        return output


class Model_D(nn.Module):
    def __init__(self, args, vocab_size=None):
        super().__init__()
        """ protein """
        self.protein_encoder = Encoder(
            args=args, vocab_size=vocab_size, hid_dim=args.protein_hidden_size, n_layers=3, kernel_size=7, dropout=0.1
        )

        """" drug """
        self.drug_encoder = MPN(args)
        self.smiles_encoder = SmilesEncoder(args, 70, hid_dim=args.hidden_size, n_layers=3, kernel_size=5, dropout=0.1)

        self.atom_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.bond_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.mol_mlp = nn.Linear(args.hidden_size * 2, args.hidden_size)

        """ interaction """
        self.atom_smiles_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        self.bond_smiles_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        self.mol_protein_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)
        # self.protein_mol_attention = MultiHeadAttention(embed_dim=args.hidden_size, num_heads=8, dropout=0.1, bias=True)

        self.mol_layerNorm = nn.LayerNorm(args.hidden_size)
        self.protein_layerNorm = nn.LayerNorm(args.hidden_size)

        self.protein_attention_pooling = AttentionPooling(args.hidden_size)
        self.mol_attention_pooling = AttentionPooling(args.hidden_size)

        self.interaction_layer = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size * 4, args.hidden_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size, 2)
        )

    def forward(self, mol_graph, smiles, smiles_mask, protein, protein_mask):
        # mol_embed, mol_mask = self.drug_encoder(mol_graph)
        """ drug """
        # 每一层的node编码和bond编码
        atom_embed, atom_mask, bond_embed, bond_mask = self.drug_encoder(mol_graph)  # Graph
        smiles_embed = self.smiles_encoder(smiles)  # Smiles

        atom_cross_embed = self.atom_smiles_attention(
            atom_embed, smiles_embed, smiles_embed, key_padding_mask=(smiles_mask == 0)
        )[0]
        bond_cross_embed = self.bond_smiles_attention(
            bond_embed, smiles_embed, smiles_embed, key_padding_mask=(smiles_mask == 0)
        )[0]

        atom_embed = self.atom_mlp(torch.concat([atom_embed, atom_cross_embed], dim=-1))
        bond_embed = self.bond_mlp(torch.concat([bond_embed, bond_cross_embed], dim=-1))
        mol_embed = self.mol_layerNorm(self.mol_mlp(torch.concat([atom_embed, bond_embed], dim=-1)))

        """ protein """
        protein_embed = self.protein_layerNorm(self.protein_encoder(protein, protein_mask))  # sequence

        """ interaction """
        # mol_cross_embed = self.mol_protein_attention(
        #     mol_embed, protein_embed, protein_embed, key_padding_mask=(protein_mask == 0)
        # )[0]
        # protein_cross_embed = self.mol_protein_attention(
        #     protein_embed, mol_embed, mol_embed, key_padding_mask=(atom_mask == 0)
        # )[0]

        mol_embed = self.mol_attention_pooling(mol_embed, atom_mask)
        protein_embed = self.protein_attention_pooling(protein_embed, protein_mask)

        pair_embed = torch.cat([mol_embed, protein_embed], dim=-1)
        output = self.interaction_layer(pair_embed)

        return output