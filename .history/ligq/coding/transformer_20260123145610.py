import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
import tiktoken
import math

if not os.path.exists('sales_textbook.txt'):
    url='https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true'
    with open("sales_textbook.txt", "wb") as f:
        f.write(requests.get(url).content)

with open('sales_textbook.txt', 'r') as f:
    text=f.read()

#print(text)

#超参数设置
tokenizer = tiktoken.get_encoding("cl100k_base")
context_length=16
d_model=64 #词嵌入维度
batch_size=4
number_head=4

tokenized_text = tokenizer.encode(text)
max_token_value = max(tokenized_text)+1  #词表大小
tokenized_text=torch.tensor(tokenized_text,dtype=torch.long)
#切分出来训练的和测试的数据
train_index=int(len(tokenized_text)*0.9)
train_data=tokenized_text[:train_index]
test_data=tokenized_text[train_index:]

data=train_data
index=torch.randint(0, len(data)-context_length, (batch_size,))
x_batch=torch.stack([data[i:i+context_length] for i in index])
y_batch=torch.stack([data[i+1:i+1+context_length] for i in index])

input_embedding__lookup_table=nn.Embedding(max_token_value, d_model)

x_batch_embedding=input_embedding__lookup_table(x_batch)
y_batch_embedding=input_embedding__lookup_table(y_batch)

#位置信息矩阵
position_lookup_table=torch.zeros(context_length, d_model)
position=torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
div_term=torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
position_lookup_table[:, 0::2] = torch.sin(position * div_term)
position_lookup_table[:, 1::2] = torch.cos(position * div_term)
position_lookup_table=position_lookup_table.unsqueeze(0) 

#print(x_batch_embedding.shape,y_batch_embedding.shape,position_lookup_table.shape)

x=x_batch_embedding+position_lookup_table
y=y_batch_embedding+position_lookup_table
#计算qkv
Wq=nn.Linear(d_model,d_model)
Wk=nn.Linear(d_model,d_model)
Wv=nn.Linear(d_model,d_model)

Q=Wq(x)
K=Wk(x)
V=Wv(x)
#print(Q.shape,K.shape,V.shape) 
Q=Q.view(batch_size, context_length, number_head, d_model//number_head).permute(0,2,1,3)
K=K.view(batch_size, context_length, number_head, d_model//number_head).permute(0,2,1,3)
V=V.view(batch_size, context_length, number_head, d_model//number_head).permute(0,2,1,3)

output=Q @ K.transpose(-2,-1)/math.sqrt(d_model//number_head)
mask=torch.triu(torch.ones(context_length,context_length),diagonal=1).bool()
output=output.masked_fill(mask, float('-inf'))
output=F.softmax(output,dim=-1)
output=output @ V
output=output.permute(0,2,1,3).contiguous().view(batch_size, context_length, d_model)
#print(output.shape)  #torch.Size([4, 16, 64])
#残差链接
output=output+x
#层归一化
layer_norm=nn.LayerNorm(d_model)
output=layer_norm(output)
#前馈网络
output=nn.Linear(d_model, d_model*4)(output)
output=F.relu(output)
output=nn.Linear(d_model*4, d_model)(output)
#残差链接
output=output+x
#层归一化
output=layer_norm(output)

#最后一层线性映射到词表大小
final_linear=nn.Linear(d_model, max_token_value)
logits=final_linear(output)

logits=F.softmax(logits,dim=-1)
