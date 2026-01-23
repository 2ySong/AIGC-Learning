import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import tiktoken
import math
import os

device='cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists("sales_textbook.txt"):
    url=""
    with open ("sales_textbook.txt","wb")as f:
        f.write(requests.get(url).content)
    
with open ("sales_textbook.txt",'r')as f:
    text=f.read()

tokenizer=tiktoken.get_encoding("cl100k_base")
tokenized_text=tokenizer.encode(text)
max_token_value=tokenizer.n_vocab
train_index=int(len(tokenized_text)*0.9)
tokenized_text=torch.tensor(tokenized_text)
train_data=tokenized_text[:train_index]
test_data=tokenized_text[train_index:]

def get_batch(data,content_length,batch_size):
    index=torch.randint(0,len(data)-content_length,batch_size)
    x=torch.stack([data[idx:idx+content_length]for idx in index])
    y=torch.stack([data[idx+1:idx+1+content_length]for idx in index])
    return x.to(device),y.to(device)

class PositionEmbedding(nn.Module):
    def __init__(self,d_model, max_length=100):
        super().__init__()
        position_lookup_table=torch.zeros(max_length,d_model)
        position=torch.arange(0,max_length).unsqueeze(1)
        div_term=torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        position_lookup_table[:,0::2]=torch.sin(position*div_term)
        position_lookup_table[:,1::2]=torch.cos(position*div_term)
        position_lookup_table=position_lookup_table.unsqueeze(0)

        self.register_buffer("position_lookup_table",position_lookup_table)
    def forward(self,x):
        position=self.position_lookup_table[:,x.size(1)]
        return x+position
    
class Multi_Attention(nn.Module):
    def __init__(self,batch_size, d_model, content_length,num_head,dropout,bias=False):
        super().__init__()
        self.Wq=nn.Linear(d_model,d_model,bias)
        self.Wk=nn.Linear(d_model,d_model,bias)
        self.Wv=nn.Linear(d_model,d_model,bias)
        self.Wo=nn.Linear(d_model,d_model,bias)

        self.content_length=content_length
        self.d_model=d_model
        self.num_head=num_head
        self.div_head=d_model//num_head
        self.attn_drop=nn.Dropout(dropout)
        self.resi_drop=nn.Dropout(dropout)
    def forward(self,x):
        Q=self.Wq(x)
        K=self.Wk(x)
        V=self.Wv(x)

        Q=Q.view(x.shape[0],x.shape[1],self.num_head,self.div_head).permute(0,2,1,3)
        K=K.view(x.shape[0],x.shape[1],self.num_head,self.div_head).permute(0,2,1,3)
        V=V.view(x.shape[0],x.shape[1],self.num_head,self.div_head).permute(0,2,1,3)

        output=Q @ K.transpose(-2,-1)/math.sqrt(self.div_head)
        mask=torch.triu(torch.ones(self.content_length,self.content_length),diagonal=1).bool().to(device=device)
        output=output.masked_fill(mask==1,float("-inf"))
        output=F.softmax(output,dim=-1)
        output=self.attn_drop(output)
        output=output @ V

        output=output.permute(0,2,1,3).contiguous().view(x.shape[0],x.shape[1],self.d_model)
        output=self.Wo(output)
        output=self.resi_drop(output)

        return output
    
class FeedForward(nn.Module):
    def __init__(self, d_model,dropout):
        super().__init__()
        self.linear1=nn.Linear(d_model,d_model*4)
        self.Relu=nn.ReLU()
        self.linear2=nn.Linear(d_model*4,d_model)
        self.drop=nn.Dropout(dropout)
    def forward(self,x):
        x=self.linear1(x)
        x=self.Relu(x)
        x=self.linear2(x)
        x=self.drop(x)
        return x
    
class GPT(nn.Module):
    def __init__(self,batch_size, d_model, content_length, num_head, dropout):
        super().__init__()
        self.batch_size=batch_size
        self.content_length=content_length
        self.token_embedding=nn.Embedding(max_token_value,d_model)
        self.position=PositionEmbedding(d_model)
        self.ln_1=nn.LayerNorm(d_model)
        self.ln_2=nn.LayerNorm(d_model)
        self.FeedForward=FeedForward(d_model,dropout)
        self.Multi_Atten=Multi_Attention(batch_size,d_model,content_length,num_head,dropout,bias=False)
        self.last_linear=nn.Linear(d_model,max_token_value)

    def forward(self,x):
        token_embedding=self.token_embedding(x)
        output=self.position(token_embedding)

        output=output+self.Multi_Atten(self.ln_1(output))
        output=output+self.FeedForward(self.ln_2(output))
        logits=self.last_linear(output)
        return logits

def generate_token(start_token,model,content_length=10):
    model.eval()
    idx=start_token
    with torch.no_grad:
        for _ in range(content_length):
            idx_cond=idx[:,-model.content_length:]
            logits=model(idx_cond)
            prob=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(prob,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=-1)
    return idx

model=GPT(batch_size=4,d_model=64,content_length=16,num_head=4,dropout=0.1)
model=model.to(device)

optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
model.train()
max_iter=1000
eval_iter=100
for step in range (max_iter):
    real_input,real_output=get_batch(train_data,model.content_length,model.batch_size)
    logits=model(real_input)
    B,T,C=logits.shape()
    loss=F.cross_entropy(logits.view(-1,C),real_output.view(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % eval_iter==0:
        print(f"step:{step},loss={loss.item():.4f}")

start_token=torch.zeros((1,1),type=torch.long,device=device)
generate_text=generate_token(start_token,model,content_length=50)
result=tokenizer.decode(generate_text[0].tolist())
print(result)