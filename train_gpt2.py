from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
import math


'''
assert  :  dam bao dieu kien dung
'''


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # Check if the number of embddings is divisible by the number of heads
        self.config = config
       # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self,x):
        B, T, C = x.size() # batch size, sequence length, embdding dimensionality (n_embd)
        #calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", 
        # hs is "head size"
        #  C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer

        qkv = self.c_attn(x)

        # split the result into query, key, and value
        q, k, v = qkv.split(self.n_embd, dim=2)
        # transpose to get dimensions B, nh, T, hs
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention matmul(q , transposed(k))   
        att= torch.matmul(q, k.transpose(-2, -1))
        # scale the matmul
        att = att * (1.0/math.sqrt(k.size(-1)))
        # disable invalid locations: 
        att = att.masked_fill(self.bias[:, :, :T, :T]==0, float('-inf'))
        # softmax
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs )
        y = y.transpose(1,2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y
        

class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)  # Fully connected layer
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)  # Fully connected layer
        self.act = nn.GELU(approximate='tanh')  # Activation function (Gaussian Error Linear Unit

    def forward(self, x):
        x = self.act(self.c_fc(x))
        x = self.c_proj(x)
        return x

class  GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = GPT2MLP(config) # Multi-layer perceptron

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))[0]
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # Size of the block
    vocab_size: int = 50256 # 50000 BPE megres + 256 bytes for special tokens (e.g. [PAD], [CLS], [SEP], [MASK])+ 1 for the end of text token
    n_embd : int = 768
    n_layer: int = 12
    n_head: int = 12


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict (
            wte     = nn.Embedding(config.vocab_size, config.n_embd),  # Word token embdding
            wpe     = nn.Embedding(config.block_size, config.n_embd),   # Word position embdding
            h       = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)]), # Transformer  GPT2Block
            ln_f    = nn.LayerNorm(config.n_embd) ,# Layer normalization
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # Language model head
    def forward(self, idx):
        B ,T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # create position tensor
        pos_emb = self.transformer['wpe'] (pos) # get the position embdding
        tok_emb = self.transformer['wte'](idx) # get the token embdding
        x = tok_emb + pos_emb # add the embddings together
        # forwad block transformer
        for block in self.transformer['h']:
            x = block(x)
        # forward the final layer norm
        x = self.transformer['ln_f'](x)
        # forward the language model head
        logits = self.lm_head(x)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        # load GPT2 weight from huggingface
        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
        from transformers import GPT2LMHeadModel
        print(f'Loading {model_type} weights from huggingface')
        # n_layer, n_embd, n_head
        config_arg = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M parameters
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 345M parameters
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 774M parameters
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), # 1558M parameters

        }[model_type]

        config_arg['vocab_size'] = 50257 # always 50257 the same for GPT model checkpoint
        config_arg['block_size'] = 1024  # always 1024 the same for GPT model checkpoint

        # create from scratch initialized gpt model
        config = GPTConfig(**config_arg)
        model = GPT(config)
        sd =  model.state_dict()
        sd_keys = sd.keys()

        # create new dict key  and dict not ending with 'attn.bias'
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # dissable this mask / buffer

        # init hugingface model
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = hf_model.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # dissable this mask / buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # dissable this mask / buffer

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            # check k endswith  in transposed
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
model.eval()
model.to('cuda')

# prefix
import tiktoken
enc = tiktoken.get_encoding('gpt2')

tokens = enc.encode('Hello, my dog is cute')
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x  = tokens.to('cuda')

torch.manual_seed(24)
torch.cuda.manual_seed(24)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) # (B,T,Vocab_size)
        # text logits at the last position
        logits = logits[:, -1, :] # (B, Vocab_size)
        probs = torch.softmax(logits, dim=-1) # (B, Vocab_size)
        # do top k sampling 50 (hugingface pipeline default)
        # top k_probs here becomes (5:50) top k indices is (5:50)
        top_k_probs, top_k_indices = torch.topk(probs, 50, dim=-1)
        # select token from top k probabilities
        idx = torch.multinomial(top_k_probs, 1) # (B,1 )
        xcol = torch.gather(top_k_indices,-1, idx) # (B,1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    token = x[i, : max_length].tolist()
    decoded = enc.decode(token)
    print(f"Sample {i+1}: {decoded}")
