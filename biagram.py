import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
context_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_emb = 32
n_hidden = 64
dropout = 1
n_blocks = 3
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False) # (32, 8)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size # 8
    
    def forward(self, x):
        B,T,C = x.shape # (64, 8, 32)

        q = self.query(x) # (64, 8, 32) @ (32, 8) = (64, 8, 8)
        k = self.key(x) # (64, 8, 8)
        v = self.value(x) # (64, 8, 8)

        wei = q @ k.transpose(-2,-1) * self.head_size ** -0.5 # (64, 8, 8) @ (64, 8, 8)

        wei = wei.masked_fill(self.tril[:T,:T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)

        out = wei @ v # (64, 8, 8) @ (64, 8, 8)
        
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(head_num)])
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
        
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.ReLU(),
            nn.Linear(n_emb,n_emb),
            nn.Dropout(dropout),
        )
    
    def forward(self,x):
        return self.block(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.head_size = n_embd // n_head # 32 // 4 = 8
        self.sa_heads = MultiHeadAttention(n_head, self.head_size)
        self.feedforward = FeedForward()
        self.norm1 = nn.LayerNorm(n_emb)
        self.norm2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.sa_heads(self.norm1(x))
        x = x + self.feedforward(self.norm2(x))
        return x
    

# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_table = nn.Embedding(context_size, n_emb)

        self.blocks = nn.Sequential(*[Block(n_embd=n_emb, n_head=4) for _ in range(n_blocks)])
        self.ly_norm = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
            
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C) -broadcast-> (B, T, C)
        # print(tok_emb.shape, pos_emb.shape)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ly_norm(x)
        logits = self.lm_head(x) # (B, T, vocab_size )

        # logits = self.block1(tok_emb)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-context_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx




model = BigramLanguageModel()
m = model.to(device)

parameters = model.parameters()
print(sum(p.nelement() for p in parameters)) # number of 

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # break
    

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))