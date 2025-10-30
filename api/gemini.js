// api/gemini.js
import express from "express";
import bodyParser from "body-parser";
import fetch from "node-fetch";

const app = express();
app.use(bodyParser.json());

app.post("/api/gemini", async (req, res) => {
  try {
    const { prompt } = req.body;
    
    if (!prompt) {
      return res.status(400).json({ error: "Missing prompt" });
    }
    
    const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
    
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=AIzaSyDyZ2I-K6GN4cnzcqgseb9PPrLurQX1pi8`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
        }),
      }
    );

    const data = await response.json();

    // 🪄 Extract just the text part
    const text =
      data?.candidates?.[0]?.content?.parts?.[0]?.text ||
      "No text response from model.";

    res.json({ text });
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

app.get("/api/cbow", (req, res) => {
  res.type("text/plain").send(`
from functools import partial
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

from torchtext import datasets
#from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

if torch.cuda.is_available():
    device=torch.device(type='cuda',index=0)
else:
    device=torch.device(type='cpu',index=0)

train_data=datasets.IMDB(split='train')
eval_data=datasets.IMDB(split='test')

mapped_train_data=[]
for label,review in train_data:
    mapped_train_data.append(review)

mapped_eval_data=[]
for label,review in eval_data:
    mapped_eval_data.append(review)

tokenizer = get_tokenizer("basic_english", language="en")

min_word_freq=20 
def build_vocab(mapped_train_data, tokenizer):        
    vocab = build_vocab_from_iterator(
        map(tokenizer, mapped_train_data),
        specials=["<unk>"],
        min_freq=min_word_freq
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab

vocab=build_vocab(mapped_train_data,tokenizer)

vocab_size=vocab.__len__()

window_size=4 
max_seq_len=256
max_norm=1
embed_dim=300
batch_size=16
text_pipeline = lambda x: vocab(tokenizer(x)) 

def collate_cbow(batch, text_pipeline):
    
     batch_input_words, batch_target_word = [], []
     
     for review in batch:
        
         review_tokens_ids = text_pipeline(review)
            
         if len(review_tokens_ids) < window_size * 2 + 1:
             continue
                
         if max_seq_len:
             review_tokens_ids = review_tokens_ids[:max_seq_len]
             
         for idx in range(len(review_tokens_ids) - window_size * 2):
             current_ids_sequence = review_tokens_ids[idx : (idx + window_size * 2 + 1)]
             target_word = current_ids_sequence.pop(window_size)
             input_words = current_ids_sequence
             batch_input_words.append(input_words)
             batch_target_word.append(target_word)
     
     batch_input_words = torch.tensor(batch_input_words, dtype=torch.long)
     batch_target_word = torch.tensor(batch_target_word, dtype=torch.long)
     
     return batch_input_words, batch_target_word

def collate_skipgram(batch, text_pipeline):
    
    batch_input_word, batch_target_words = [], []
    
    for review in batch:
        review_tokens_ids = text_pipeline(review)

        if len(review_tokens_ids) < window_size * 2 + 1:
            continue

        if max_seq_len:
            review_tokens_ids = review_tokens_ids[:max_seq_len]

        for idx in range(len(review_tokens_ids) - window_size * 2):
            current_ids_sequence = review_tokens_ids[idx : (idx + window_size * 2 + 1)]
            input_word = current_ids_sequence.pop(window_size)
            target_words = current_ids_sequence

            for target_word in target_words:
                batch_input_word.append(input_word)
                batch_target_words.append(target_word)

    batch_input_word = torch.tensor(batch_input_word, dtype=torch.long)
    batch_target_words = torch.tensor(batch_target_words, dtype=torch.long)
    return batch_input_word, batch_target_words

traindl_cbow = DataLoader(
        mapped_train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_cbow,text_pipeline=text_pipeline)
    )

traindl_skipgram = DataLoader(
        mapped_train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_skipgram,text_pipeline=text_pipeline)
    )

evaldl_cbow = DataLoader(
        mapped_eval_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_cbow,text_pipeline=text_pipeline)
    )

evaldl_skipgram = DataLoader(
        mapped_eval_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_skipgram,text_pipeline=text_pipeline)
    )


class CBOW(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            max_norm=max_norm
        )
        self.linear = nn.Linear(
            in_features=embed_dim,
            out_features=vocab_size,
        )

    def forward(self, x):
        x = self.embeddings(x)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x

class SkipGram(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            max_norm=max_norm
        )
        self.linear = nn.Linear(
            in_features=embed_dim,
            out_features=vocab_size,
        )

    def forward(self, x):
        x = self.embeddings(x)
        x = self.linear(x)
        return x

def train_one_epoch(model,dataloader):
    model.train()
    running_loss = []

    for i, batch_data in enumerate(dataloader):
        inputs = batch_data[0].to(device)
        targets = batch_data[1].to(device)
        opt.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        opt.step()

        running_loss.append(loss.item())

    epoch_loss = np.mean(running_loss)
    print("Train Epoch Loss:",round(epoch_loss,3))
    loss_dict["train"].append(epoch_loss)

def validate_one_epoch(model,dataloader):
    model.eval()
    running_loss = []

    with torch.no_grad():
        for i, batch_data in enumerate(dataloader, 1):
            inputs = batch_data[0].to(device)
            targets = batch_data[1].to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            running_loss.append(loss.item())


    epoch_loss = np.mean(running_loss)
    print("Validation Epoch Loss:",round(epoch_loss,3))
    loss_dict["val"].append(epoch_loss)

loss_fn=nn.CrossEntropyLoss()
n_epochs=5
loss_dict={}
loss_dict["train"]=[]
loss_dict["val"]=[]

choice=input("Enter cbow/skipgram:")
if choice=="cbow":
    model=CBOW(vocab_size).to(device)
    dataloader_train=traindl_cbow
    dataloader_val=evaldl_cbow
elif choice=="skipgram":
    model=SkipGram(vocab_size).to(device)
    dataloader_train=traindl_skipgram
    dataloader_val=evaldl_skipgram

opt=optim.Adam(params=model.parameters(),lr=0.001)

for e in range(n_epochs):
    print("Epoch=",e+1)
    train_one_epoch(model,dataloader_train)
    validate_one_epoch(model,dataloader_val)

trimmed_model=model.embeddings

emb1=trimmed_model(torch.tensor([vocab.lookup_indices(["film"])]).to(device))
emb2=trimmed_model(torch.tensor([vocab.lookup_indices(["movie"])]).to(device))
print(emb1.shape, emb2.shape)
cos=torch.nn.CosineSimilarity(dim=2)
print(cos(emb1,emb2))

emb1=trimmed_model(torch.tensor([vocab.lookup_indices(["his"])]).to(device))
emb2=trimmed_model(torch.tensor([vocab.lookup_indices(["from"])]).to(device))
print(cos(emb1,emb2))

emb1=trimmed_model(torch.tensor([vocab.lookup_indices(["he"])]).to(device))
emb2=trimmed_model(torch.tensor([vocab.lookup_indices(["were"])]).to(device))
print(cos(emb1,emb2))

    `);
});

export default app;
