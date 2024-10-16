import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

# Hyperparameters 
batch_size = 64 # How many independent sequence will be process in parallel?
block_size = 64 # what is the maximum content length for prediction
max_iter = 2000
eval_interval = 5 # estimating loss after 300 intervals
learning_rate = 1e-03
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # setting up the device to be apple's GPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # this is used for NVIDIA's GPU which our mac don't have
eval_iters = 200 # for estimating average loss
n_embd = 420
n_head = 6
n_layer = 6 # number of transformer block
dropout = 0.2

#BPE Tokenizer

with open('Shakespeare.txt','r') as file:
    text = file.read()

tokens = text.encode("utf-8") # raw bytes
tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
  # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
  newids = []
  i = 0
  while i < len(ids):
    # if we are not at the very last position AND the pair matches, replace it
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

# ---
vocab_size = 420 # the desired final vocabulary size
num_merges = vocab_size - 256
ids = list(tokens) # copy so we don't destroy the original list

merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  #print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx

  #Encoder Decoder for BPE
def encode(text):
  # given a string, return list of integers (the tokens)
  tokens = list(text.encode("utf-8"))
  while len(tokens) >= 2:
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    if pair not in merges:
      break # nothing else can be merged
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

# Lets encode the entire dataset
data = torch.tensor(encode(text),dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n] # We will train on 90%
val_data = data[n:]
#print(data[:10])

torch.manual_seed(7)

    # Getting the data
with open('Shakespeare','r') as file:
  text = file.read()

    # All unique characters
char = sorted(list(set(text)))
vocab_size = len(char)
    #All unique words
words = list(set(text.split()))
special_words = [ '\n', ' ']
words.extend(special_words)
vocab_size = len(words)
    #print(vocab_size)
    #Creating Mapping from characters to integers, Effectively the Tokenizer of our model
stoi = {i:s for s,i in enumerate(char)}
itos = {i:s for i,s in enumerate(char)}
encode = lambda s: [stoi[c] for c in s] # Encode: taking a string and converting into list of  integers
decode = lambda l: ''.join([itos[c] for c in l]) # Decode: taking a list of integers and converting into a string

    # Lets encode the entire dataset
data = torch.tensor(encode(text),dtype=torch.long)

    # Let's split the data set into train and validation
n = int(0.9*len(data))
train_data = data[:n] # We will train on 90%
val_data = data[n:]
   # print(data[:10])
   # (vocab_size)

# Data loading
def get_batch(split):
    # Generate a small batch of data of context x and target y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size,(batch_size,)) # randomly picking batch_size integers between 0 and len(data)- block_size
    x = torch.stack([data[i:block_size+i] for i in ix]) # getting the context for randomly chose batches and then stacking it
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # getting the target
    x,y = x.to(device), y.to(device) # make sure that data is loaded to GPU device
    return x, y



def save_checkpoint(model, optimizer, iter, filename='checkpoint.pth'):
    checkpoint = {
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at iteration {iter}.")

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iter = checkpoint['iter'] + 1
    print(f"Checkpoint loaded. Resuming training from iteration {start_iter}.")
    return start_iter

# Getting the average loss over batches
@torch.no_grad() # Memory efficiency
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean() # we are taking the mean(over multiple batches) so that we get a better estimate
    model.train()
    return out
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Creating the head for attention
class Head(nn.Module): 
    """One head of sefl-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias= False)
        self.query = nn.Linear(n_embd, head_size, bias= False)
        self.value = nn.Linear(n_embd, head_size, bias= False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size))) # tril is not a parameter in pytorch so we have to register it using register_buffer. This is the same thing we have seen in Attention, the lower triangular matrix.
    
        self.dropout = nn.Dropout(dropout) # For regularization

        
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) # (B,T, head_size)
        q = self.query(x) # (B,T, head_size)

        # Getting Attention !!
        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5 # (B,T,head_size)@(B,head_size,T) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) # (B,T,T), making sure that the present token get the attention of previous token not from the future token
        wei = F.softmax(wei, dim=-1) #(B,T,T), converting into probabilities
        wei = self.dropout(wei)

        # Weighted aggregation of values
        v = self.value(x)
        out = wei @ v # (B,T,T) @ (B,T,C) --> (B,T,C)
        return out

# Multihead Attention
class MultiHeadAttention(nn.Module):
    """ Multiple Head of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.head = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout) # For regularization

    def forward(self, x):
        out = torch.cat([h(x) for h in self.head], dim=-1)
        out = self.dropout(self.proj(out)) # Applying the projection matrix W^o, can look up in the paper.
        return out # Concatenating the heads over the channel dimensions

# Feed-Forward Layer
class FeedForward(nn.Module):
    """A simple linear layer followed by non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # This inner multliplication of 4 introduces some extra parameters in the feedforward layer which improve the computation. This is what is described in the attention all you need paper.
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # This is the Projection layer. The above two are simple feed forward layers. 
            nn.Dropout(dropout) # For regularization
        )

    def forward(self, x):
        return self.net(x)
    
# The Transformer Block
class Block(nn.Module):
    """Transformer block : Communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embeddding dimension, n_head: the number of heads we would like
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # the layernorm layer. The normalization happens along the embedding dimension. The batch and time both act like a batch dimension and along the 'C' dimesion normalization occur same as in batchnorm but there we normalizes along the batch 
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # x is added as a form of skip/Residual connection. Note that x and self.sa have same dimensions so we don't need any extra matris of parameters there
        x= x + self.ffwd(self.ln2(x)) # Similarly as above x is added in the form of residual connection. Layer norm is added before feedforward and attention instead of after, these days. 
        return x


class BigramLanguageModel(nn.Module): # gpt-model

    def __init__(self): # as vocab size is globally defined
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # Embedding the positon as well :) . Think of it kind of positional encoding
        self.block = nn.Sequential(*[Block(n_embd, n_head= n_head) for _ in range(n_layer)])  # Putting transformer block multiple times so that it can communicate and compute over the data to have better understanding
        self.ln_f = nn.LayerNorm(n_embd) # A final layer norm right before getting the output
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) C--> n_embd !!!. Keep in mind that Embedding layer itself do the one_hot encoding so (B,T) ---> (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T,device=device)) # (T) ---> (T,C), Note that using this layer will constrain us over 'T' to be never greater than the block_size because the positional encoding layer have input restricted to block_size 
        x = tok_emb + pos_emb # postional embedding will be broadcasted. This is the positional encoding step. (B,T,C)
        x = self.block(x) # Calling the transformer block multiple times
        logits = self.lm_head(x) # (B,T, Vocab_size) getting the final logits with correct dimensions
        if targets==None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C) # Because Pytorch only want the context in its 1 dimension
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)

        return logits, loss
    
    def generate(self,idx,max_new_tokens): # The job of this function is to take an input idx with batch size B and sequence T and continue the generation along T, T+1, T+2 upto max_num_Tokens for all chunks of batch B.
        # idx is a (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size token
            idx_cond = idx[:,-block_size:]
            # Get the prediction
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:,-1,:] # Become (B,C)
            # applying the softmax to get the probabilitie
            probs = F.softmax(logits,dim=-1) #(B,C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs,num_samples=1) #(B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx,idx_next),dim=1) # (B,T+1)

        return idx

model = BigramLanguageModel()
m = model.to(device) # moving the model parameters to the device
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a Pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

# Attempt to load a checkpoint
checkpoint_file = 'checkpoint.pth'
try:
    start_iter = load_checkpoint(model, optimizer, filename=checkpoint_file)
except FileNotFoundError:
    start_iter = 0  # Start from scratch if no checkpoint is found

# Training the gpt
for iter in range(1000):
    
    # every once in a while evaluate loss on train and val set
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter} : train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    # Sample a batch from the data 
    xb, yb = get_batch('train')

   
    logits, loss = m(xb,yb) # Forward Prop
    optimizer.zero_grad(set_to_none=True) # setting gradient zero form the previous iteration
    loss.backward() # Backward Prop
    optimizer.step() # Update parameters

      # Save checkpoint periodically
    if iter % eval_interval == 0:
        save_checkpoint(model, optimizer, iter, filename=checkpoint_file)

# Generate from the model
context = torch.zeros((1,1),dtype=torch.long,device=device) # while inference we are making sure that context is on the device
print(decode(m.generate(context,max_new_tokens=400)[0].tolist())) # tolist here will convert the pytorch tensor into python list so that we can use in the decode function
