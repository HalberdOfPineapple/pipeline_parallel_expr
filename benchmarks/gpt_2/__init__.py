import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

from collections import OrderedDict
from typing import List, Tuple, Dict, Any, Union
from .flatten_sequential import flatten_sequential

def build_GPT2(
        num_layers: int,        # e.g., 12 for GPT-2 small
        num_heads: int,         # e.g., 12 for GPT-2 small
        d_model: int,           # e.g., 768 for GPT-2 small
        d_ff: int,              # e.g., 3072 for GPT-2 small
        vocab_size: int,        # Size of the vocabulary
        max_seq_len: int,       # Maximum sequence length
        dropout: float = 0.1    # Dropout rate
    ) -> nn.ModuleList:

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len):
            super().__init__()
            self.pos_encoding = nn.Parameter(torch.zeros(max_len, d_model))

        def forward(self, x):
            return x + self.pos_encoding[:x.size(1), :]

    class GPT2Block(nn.Module):
        def __init__(self, d_model, num_heads, d_ff, dropout):
            super().__init__()
            self.ln_1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )

        def forward(self, x, attn_mask=None):
            attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
            x = x + attn_output
            x = self.ln_1(x)
            x = x + self.mlp(x)
            x = self.ln_2(x)
            return x
    model = nn.Sequential(OrderedDict([
        ('embedding', nn.Embedding(vocab_size, d_model)),
        ('pos_encoding', PositionalEncoding(d_model, max_seq_len)),
        ('blocks', nn.Sequential(*[GPT2Block(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])),
        ('ln_f', nn.LayerNorm(d_model)),
        ('head', nn.Linear(d_model, vocab_size, bias=False))
    ]))
    model = flatten_sequential(model)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    model.apply(init_weights)
    return model

# Example usage
def gpt2_small(vocab_size: int = 50257, max_seq_len: int = 1024):
    # https://github.com/openai/gpt-2/blob/master/model_card.md
    return build_GPT2(
        num_layers=12, 
        num_heads=12, 
        d_model=768, 
        d_ff=3072, 
        vocab_size=vocab_size, 
        max_seq_len=max_seq_len,
    )


# ========================
EXAMPLE_TEXTS = [
    "The greatest glory in living lies not in never falling, but in rising every time we fall.",
    "The way to get started is to quit talking and begin doing.",
    "Your time is limited, so don't waste it living someone else's life.",
    "If life were predictable it would cease to be life, and be without flavor.",
    "If you set your goals ridiculously high and it's a failure, you will fail above everyone else's success.",
    "Life is what happens when you're busy making other plans."
]


def load_text_from_file(filename: str) -> str:
    text_list = []
    with open(filename, 'r') as f:
        for line in f:
            text_list.append(line.strip())
    return text_list

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.inputs = [self.tokenizer.encode(text, max_length=max_length, truncation=True) for text in texts]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = self.inputs[idx]
        return torch.tensor(input_ids, dtype=torch.long)

def build_train_stuffs(model: nn.Module, batch_size: int, max_length: int, dataset_texts: List[str] =EXAMPLE_TEXTS):

    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # dataset = TextDataset(dataset_texts, tokenizer, max_length)

    # def collate_fn_padd(batch):
    #     '''
    #     Padds batch of variable length

    #     note: it converts things ToTensor manually here since the ToTensor transform
    #     assume it takes in images rather than arbitrary tensors.
    #     '''
    #     ## get sequence lengths
    #     lengths = torch.tensor([ t.shape[0] for t in batch ])
    #     ## padd
    #     batch = [ torch.Tensor(t) for t in batch ]
    #     batch = torch.nn.utils.rnn.pad_sequence(batch)
    #     ## compute mask
    #     mask = (batch != 0)
    #     return batch, lengths, mask

    # train_loader = DataLoader(
    #     dataset, 
    #     batch_size=batch_size,
    #     shuffle=True,
    #     collate_fn=collate_fn_padd,
    # )
    

    # 3. define loss function
    # GPT-2 uses Language Modeling, so we use CrossEntropyLoss, but it's applied differently.
    criterion = nn.CrossEntropyLoss()

    # 4. define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    return train_loader, criterion, optimizer
