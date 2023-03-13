import os
import argparse

import yaml
from tqdm.auto import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from torchmetric.text.perplexity import Perplexity

from data import Prepocessing, TrainDataset, build_vocab, TrainDataUtils
from model import PoemModel


def run(args, device):
    # build dataset
    vocab = build_vocab(args['train_df'])
    preprocess = Prepocessing(vocab)
    train_loader = TrainDataUtils.load_data_loader(args['train_df'], preprocess, args['batch_size'], True)
    val_loader = TrainDataUtils.load_data_loader(args['val_df'], preprocess, 8, False)

    model = PoemModel(len(vocab), **args['model'])
    model.to(device=device)

    total_steps = int(len(train_loader) * args['epochs'])
    optimizer = optim.AdamW(model.params())  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = OneCycleLR(optimizer, 
                            max_lr=args['lr'], 
                            total_steps=total_steps, 
                            pct_start=args['pct_start'])

    creterion = nn.CrossEntropyLoss()
    best_score = -1
    for epoch in range(args['epochs']):
        avg_loss = 0.
        model.train()
        pbar = tqdm(train_loader,total=len(train_loader),leave=False)
        for x, y in pbar:
            x = x.to(device = device)
            y = y.to(device = device)
            logits = model(x)
            loss = creterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            lossf = loss.item()
            pbar.set_postfix(loss = lossf)
            avg_loss += loss.item() / len(train_loader)
            
        model.eval()
        val_preds = []
        val_per = Perplexity(ignore_index=-1)
        with torch.no_grad():
            for i,(x, y) in enumerate(val_loader):
                x = x.to(device = device)
                if epoch == 0:
                    val_labels.extend(y)
                mask = x > 0
                logits = model(x, attention_mask=mask).cpu()
                y_pred = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).tolist()
                val_preds.extend(y_pred)
                val_per.update(logits, y)

        accuracy = np.mean(val_preds)
        val_per = val_per.compute()
        print(f"Epoch {epoch} accuracy = {accuracy:.4f} perplexity = {val_per:.4f}")
        if val_per <= best_score or best_score == -1:
            best_score = val_per
            torch.save(model.state_dict(),
                        os.path.join(args['ckpt_path'], f"model_{val_per:.4f}.pt"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('--config')
    parser.add_argument('--device')

    cmd = parser.parse_args()

    with open(cmd.config) as f:
        args = yaml.safe_load(f)

    store_path = os.path.split(args['ckpt_path'])[0]
    os.makedirs(store_path, exist_ok=True)

    run(args, cmd.device)

