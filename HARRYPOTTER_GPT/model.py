import torch
from torch import nn

class FullyConnectedBlock(nn.Module):
    def __init__(self, embedding_dim, dropout=0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.model(x)


class DecoderBlock(nn.Module):
    def __init__(self, embededding_dim, num_heads, sequence_length, dropout=0):
        super().__init__()
        assert embededding_dim % num_heads == 0, "Invalid number of heads for embedding dimension."
        self.layer_norm = nn.LayerNorm(embededding_dim)
        self.attention = nn.MultiheadAttention(embededding_dim, num_heads, dropout=dropout)
        self.fully_connected = nn.Sequential(
            nn.LayerNorm(embededding_dim),
            FullyConnectedBlock(embededding_dim, dropout=dropout)
        )
        self.register_buffer(
            name="mask",
            tensor=torch.tril(torch.ones(sequence_length, sequence_length))
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = x + self.attention(x, x, x, need_weights=False, attn_mask=self.mask, is_causal=True)[0]
        x = x + self.fully_connected(x)
        return x


class GPT(nn.Module):
    def __init__(self, sequence_length, vocab_size, embedding_dim, num_heads, num_blocks, device="cpu", dropout=0):
        super().__init__()
        self.sequence_length = sequence_length
        self.tokens = nn.Embedding(vocab_size, embedding_dim)
        self.positions = nn.Embedding(sequence_length, embedding_dim)
        self.blocks = nn.ModuleList([
            DecoderBlock(embedding_dim, num_heads, sequence_length, dropout=dropout) for _ in range(num_blocks)
        ])
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
        self.device = device

    def forward(self, prompt):
        seq_length = prompt.shape[1]
        position = torch.arange(0, seq_length).unsqueeze(0).to(self.device)

        token_embeddings = self.tokens(prompt)
        position_embeddings = self.positions(position)
        x = token_embeddings + position_embeddings

        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)

        logits = self.lm_head(x)
        return logits

    def generate(self, prompt, max_new_tokens, k=None, temp=1):
        with torch.no_grad():
            self.eval()
            for _ in range(max_new_tokens):
                curr_prompt = prompt if prompt.size(1) <= self.sequence_length else prompt[:, -self.sequence_length:]

                logits = self.forward(curr_prompt)
                logits = logits[:, -1, :]/temp

                if k:
                    top_k, _ = torch.topk(logits, k)
                    logits[logits < top_k[:, [-1]]] = -float('Inf')

                probabilities = nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                prompt = torch.cat((prompt, next_token), dim=1)
        return prompt

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
