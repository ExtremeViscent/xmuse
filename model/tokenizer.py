from typing import Optional, List, Tuple
import torch
import torch.nn as nn


class ContinuousTokenizer(nn.Module):
    def __init__(
        self, 
        n_tokens: int = 184, 
        dim: int = 512
    ):
        """
        Tokenizer for continuous parameters.
        Args:
            ranges: list of (min, max) tuples for each continuous parameter
            n_tokens: number of continuous parameters
            dim: embedding dimension
        """

        super().__init__()
        self.type_emb = nn.Parameter(torch.zeros(1, 1, dim))
        self.id_emb   = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, x_z):  # (B, n_tokens)
        """
        Args:
            x_z: tensor of shape (B, n_tokens) with continuous parameter values
        Returns:
            Tensor of shape (B, n_tokens, dim) with embeddings
        """
        B, n_tokens = x_z.shape
        assert n_tokens == self.id_emb.shape[1], "Input token count does not match tokenizer configuration."
        
        # Pass through MLP
        x_emb = self.mlp(x_z.unsqueeze(-1))  # (B, n_tokens, dim)

        # Add type and id embeddings
        out = x_emb + self.type_emb + self.id_emb  # (B, n_tokens, dim)
        return out
    

class DiscreteTokenizer(nn.Module):
    def __init__(
        self,
        cat_sizes: List[int],
        n_tokens: int = 43,
        dim: int = 512
    ):
        """
        Tokenizer for categorical parameters. 
            tok_i = E_i[idx_i] + E_var_id[i] + E_type_disc
        Args:
            cat_sizes: list with number of categories for each categorical parameter
            n_tokens: number of categorical parameters
            dim: embedding dimension
        """

        super().__init__()
        self.type_emb = nn.Parameter(torch.zeros(1, 1, dim))
        self.id_emb   = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)

        # Create embedding tables for each categorical parameterk
        self.tables = nn.ModuleList()
        for size in cat_sizes:
            emb_table = nn.Embedding(size, dim)
            nn.init.normal_(emb_table.weight, mean=0.0, std=0.02)
            self.tables.append(emb_table)

    def forward(self, x_cat):  # (B, n_tokens)
        """
        Args:
            x_cat: tensor of shape (B, n_tokens) with categorical parameter indices
        Returns:
            Tensor of shape (B, n_tokens, dim) with embeddings
        """
        B, n_tokens = x_cat.shape
        assert n_tokens == self.id_emb.shape[1], "Input token count does not match tokenizer configuration."

        emb_list = []
        for i in range(n_tokens):
            idxs = x_cat[:, i]  # (B,)
            emb = self.tables[i](idxs)  # (B, dim)
            emb_list.append(emb.unsqueeze(1))  # (B, 1, dim)
        
        x_emb = torch.cat(emb_list, dim=1)  # (B, n_tokens, dim)

        # Add type and id embeddings
        out = x_emb + self.type_emb + self.id_emb  # (B, n_tokens, dim)
        return out
