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