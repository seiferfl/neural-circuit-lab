import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import einops

class Transformer(nn.Module):
    def __init__(self, d_model,d_mlp,d_heads,d_vocab,num_heads,n_ctx,act_type):
        super().__init__()
        self.embed = Embed(d_vocab=d_vocab,d_model=d_model)
        self.pos_embed = PosEmbed(n_ctx=n_ctx,d_model=d_model)
        self.attention = Attention(d_model=d_model,num_heads=num_heads,d_head=d_heads,n_ctx=n_ctx)
        self.mlp = MLP(d_model=d_model,d_mlp=d_mlp,act_type=act_type)
        self.unembed = Unembed(d_vocab=d_vocab,d_model=d_model)


    def forward(self,x):
        x = self.embed(x)       #embed the tokens
        
        x = self.pos_embed(x)   #x= x + pos_embed
        
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return self. unembed(x)
    
class Embed(nn.Module):
    """
    Embeds the one-hot encoded input tokens with the embedding matrix W_E
    """
    def __init__(self, d_vocab , d_model):
        super().__init__()
        self.W_E = nn.Parameter(t.randn(d_model,d_vocab)/np.sqrt(d_model))

    def forward(self, x):
        return t.einsum("dbp -> bpd",self.W_E[:,x])
    
class PosEmbed(nn.Module):
    """
    To each embedded token at position i, we add a vector W_pos[i,:].
    The matrix W_pos is learned
    """
    def __init__(self, n_ctx,d_model):
        super().__init__()
        self.W_pos = nn.Parameter(t.randn(n_ctx,d_model)/np.sqrt(d_model))

    def forward(self,x):
        return x + self.W_pos
    
class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx):
        super().__init__()
        self.W_K = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_Q = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_V = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_O = nn.Parameter(t.randn(d_model, d_head * num_heads) / np.sqrt(d_model))

        # causal mask (1 = keep, 0 = block)
        self.register_buffer("mask", t.tril(t.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.n_ctx = n_ctx                # <-- remember it if you still need it later

    def forward(self, x):
        K = t.einsum("bpd, ihd -> biph", x, self.W_K)
        Q = t.einsum("bpd, ihd -> biph", x, self.W_Q)
        V = t.einsum("bpd, ihd -> biph", x, self.W_V)

        scores = t.einsum("biph, biqh -> biqp", K, Q)            # [batch, head, q, k]
        scores = scores.masked_fill(self.mask[None, None, :, :] == 0, -1e10)

        attn = F.softmax(scores / np.sqrt(self.d_head), dim=-1)
        z = t.einsum("biph, biqp -> biqh", V, attn)

        z_flat = einops.rearrange(z, "b i q h -> b q (i h)")
        out = t.einsum("bqf, df -> bqd", z_flat, self.W_O)
        return out

class MLP(nn.Module):
    def __init__(self, d_model,d_mlp,act_type):
        super().__init__()
        self.W_in = nn.Parameter(t.randn(d_mlp,d_model)/np.sqrt(d_model))
        self.b_in = nn.Parameter(t.zeros(d_mlp))

        self.W_out = nn.Parameter(t.randn(d_model,d_mlp)/np.sqrt(d_model))
        self.b_out = nn.Parameter(t.zeros(d_model))

        self.act_type = act_type
        assert act_type in ["ReLU","GeLU"]

    def forward(self,x):
        x = t.einsum("bpd, md -> bpm",x,self.W_in) + self.b_in
        if self.act_type == "ReLU":
            x = F.relu(x)

        elif self.act_type == "GeLU":
            x = F.gelu(x)

        x = t.einsum("bpm,dm -> bpd",x,self.W_out)
        return x
    
class Unembed(nn.Module):
    """
    Unembedding to the tokens
    """
    def __init__(self, d_vocab,d_model):
        super().__init__()
        self.W_U = nn.Parameter(t.randn(d_model,d_vocab)/np.sqrt(d_vocab))

    def forward(self,x):

        return x @ self.W_U