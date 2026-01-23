import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
import tiktoken
import math

def get_batch(data):
    index=torch.randint(0,len(data)-content_length,(batch_size,))
    x=torch.stack([data[i:i+content_length]for i in index])
    y=torch.stack([data[i+1:i+1+content_length]for i in index])

    return x.to(device='cuda'),y.to(device='cuda')

if not os.path.exists("sales_textbook.txt"):
    url = "https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true"
    with open("sales_textbook.txt", "wb") as f:
        f.write(requests.get(url).content)

with open("sales_textbook.txt", "r") as f:
    text = f.read()

tokenizer = tiktoken.get_encoding("cl100k_base")
d_model = 64
content_length = 16
batch_size = 4
dropout = 0.1

tokenized_text=tokenizer.encode(text)
max_value_token=tokenizer.n_vocab
text_length=len(tokenized_text)
train_index=int(text_length*0.9)
tokenized_text=torch.tensor(tokenized_text)
train_data=tokenized_text[:train_index]
test_data=tokenized_text[train_index:]

class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_value=100):
        super().__init__()

        position_lookup_table=torch.zeros(max_value,d_model)
        position=torch.arange(0,max_value,1).unsqueeze(1)
        div_term=torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        position_lookup_table[:,0::2]=torch.sin(position*div_term)
        position_lookup_table[:,1::2]=torch.cos(position*div_term)

        position_lookup_table=position_lookup_table.unsqueeze(0)
        self.register_buffer('position_lookup_table',position_lookup_table)
    def forward(self,x):
        positions=self.position_lookup_table[:,:x.size(1)]
        return x+positions

class Multi_Attention(nn.Module):
    def __init__(self, d_model,batch_size,num_heads, bias=False):
        super().__init__()
        self.Wq=nn.Linear(d_model,d_model,bias)
        self.Wk=nn.Linear(d_model,d_model,bias)
        self.Wv=nn.Linear(d_model,d_model,bias)
        self.Wo=nn.Linear(d_model,d_model,bias)
        self.d_model=d_model
        self.batch_size=batch_size
        self.num_heads=num_heads
        self.heads_div=self.d_model//num_heads
        self.atten_drop=nn.Dropout(dropout)
        self.resi_drop=nn.Dropout(dropout)

    def forward(self,x):
        Q=self.Wq(x)
        K=self.Wk(x)
        V=self.Wv(x)

        Q=Q.view(x.shape[0],x.shape[1],self.num_heads,self.heads_div).permute(0,2,1,3)
        K=K.view(x.shape[0],x.shape[1],self.num_heads,self.heads_div).permute(0,2,1,3)
        V=V.view(x.shape[0],x.shape[1],self.num_heads,self.heads_div).permute(0,2,1,3)

        output=Q @ K.transpose(-2,-1)/math.sqrt(self.heads_div)
        mask=torch.triu(torch.ones(x.shape[1],x.shape[1]),diagonal=1).bool().to(x.device)
        output=output.masked_fill(mask==1,float('-inf'))
        output=F.softmax(output,dim=-1)
        output=self.atten_drop(output)
        output=output @ V

        output=output.permute(0,2,1,3).contiguous().view(x.shape(0),x.shape(1),d_model)
        output=self.Wo(output)
        output=self.resi_drop(output)

        return output
    

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1=nn.Linear(d_model,d_model*4)
        self.Relu=nn.ReLU()
        self.linear2=nn.Linear(d_model*4,d_model)
    def forward(self,x):
        x=self.linear1(x)
        x=self.Relu(x)
        x=self.linear2(x)
        return x

class GPT(nn.Module):
    def __init__(self, d_model,num_heads,max_value_token):
        super().__init__()
        self.token_embedding=nn.Embedding(max_value_token,d_model)
        self.position_embedding=PositionEmbedding(d_model)
        self.ln_1=nn.LayerNorm(d_model)
        self.multi_head=Multi_Attention(d_model,batch_size,num_heads)
        self.ln_2=nn.LayerNorm(d_model)
        self.FeedForward=FeedForward(d_model)
        self.last_linear=nn.Linear(d_model,max_value_token)

    def forward(self,x):
        token_embedding=self.token_embedding(x)
        x=self.position_embedding(token_embedding)

        x=x+self.multi_head(self.ln_1(x)) 
        x=self.FeedForward(self.ln_2(x))+x
        logits=self.last_linear(x)
        return logits

model = GPT(d_model, num_heads=4, max_value_token=max_value_token)

real_input, real_target=get_batch(train_data)

logits=model(real_input)
print(logits.shape)
