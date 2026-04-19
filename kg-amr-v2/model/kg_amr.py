import os
cwd = os.getcwd()
assert "KG-AMR" in cwd, f"ABORT: Must run inside project root, got {cwd}"

import torch
import torch.nn as nn
import pytorch_lightning as pl
from paths import KG_EMBED_DIM, GENOMIC_HIDDEN, FUSED_DIM

# Torch F1 for validation (macro)
def f1_score_torch(preds, targets, num_classes=2):
    from sklearn.metrics import f1_score
    return torch.tensor(f1_score(targets.cpu(), preds.cpu(), average="macro"), device=preds.device)

class KGAMR(pl.LightningModule):
    def __init__(self, kmer_dim, num_genes):
        super().__init__()
        self.save_hyperparameters()
        # Genomic encoder
        self.genomic_encoder = nn.Sequential(
            nn.Linear(kmer_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        # KG encoder with self-attention pooling
        self.gene_attn = nn.Linear(KG_EMBED_DIM, 1)  # kept for checkpoint compat
        self.gene_attn_q = nn.Linear(256, KG_EMBED_DIM, bias=False)  # cross-attn: query from genomic encoder (256-dim → KG_EMBED_DIM)
        # Cross-attention fusion
        self.proj_g = nn.Linear(256, FUSED_DIM)
        self.proj_k = nn.Linear(KG_EMBED_DIM, FUSED_DIM)
        self.gate_mlp = nn.Sequential(
            nn.Linear(FUSED_DIM * 2, FUSED_DIM),
            nn.ReLU(),
            nn.Linear(FUSED_DIM, FUSED_DIM),
            nn.Sigmoid()
        )
        # Primary classifier
        self.classifier = nn.Sequential(
            nn.Linear(FUSED_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        # Auxiliary gene detection head
        self.gene_head = nn.Sequential(
            nn.Linear(FUSED_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, num_genes),
            nn.Sigmoid()
        )

    def forward(self, kmer_x, gene_embeds):
        g = self.genomic_encoder(kmer_x)
        # Cross-attention: each genome queries the gene most aligned with its hidden state
        # g is the 256-dim genomic_encoder output (pre-proj_g), used as the query
        query       = self.gene_attn_q(g).unsqueeze(2)              # [B, KG_EMBED_DIM, 1]
        attn_scores = torch.bmm(gene_embeds, query) / (gene_embeds.shape[-1] ** 0.5)  # [B, n_genes, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)   # [batch, n_genes, 1]
        k = (attn_weights * gene_embeds).sum(dim=1)        # [batch, KG_EMBED_DIM]
        g_p = self.proj_g(g)
        k_p = self.proj_k(k)
        gate = self.gate_mlp(torch.cat([g_p, k_p], dim=-1))
        fused = gate * g_p + (1 - gate) * k_p
        logits = self.classifier(fused)
        gene_preds = self.gene_head(fused)
        return logits, gene_preds, attn_weights.squeeze(-1), gate

    def training_step(self, batch, _):
        kmer_x, gene_embeds, y_amr, y_genes = batch
        logits, gene_preds, _, _ = self(kmer_x, gene_embeds)
        loss_amr = nn.CrossEntropyLoss()(logits, y_amr)
        loss_gene = nn.BCELoss()(gene_preds, y_genes.float())
        loss = loss_amr + 0.3 * loss_gene
        self.log("train_loss", loss)
        self.log("train_loss_amr", loss_amr)
        self.log("train_loss_gene", loss_gene)
        return loss

    def validation_step(self, batch, _):
        kmer_x, gene_embeds, y_amr, y_genes = batch
        logits, gene_preds, _, _ = self(kmer_x, gene_embeds)
        preds = torch.argmax(logits, dim=1)
        self.log("val_f1_macro", f1_score_torch(preds, y_amr))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# --- Smoke test ---
if __name__ == "__main__":
    kmer_dim = 10000
    num_genes = 500
    model = KGAMR(kmer_dim=kmer_dim, num_genes=num_genes)
    kmer_dummy = torch.randn(32, kmer_dim)
    gene_dummy = torch.randn(32, 20, KG_EMBED_DIM)
    logits, gene_preds, attn_weights, gate = model(kmer_dummy, gene_dummy)
    print(f"logits:       {logits.shape}")
    print(f"gene_preds:   {gene_preds.shape}")
    print(f"attn_weights: {attn_weights.shape}")
    print(f"gate:         {gate.shape}")
    assert logits.shape == (32, 2)
    assert attn_weights.shape == (32, 20)
    print("✅ Smoke test passed")
