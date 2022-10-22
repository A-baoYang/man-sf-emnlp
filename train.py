from __future__ import division
from __future__ import print_function

import argparse
import datetime
from glob import glob
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import pandas as pd
import pickle
import random
from sklearn.metrics import (
    classification_report,
    matthews_corrcoef,
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/mansf-stocknet")
from tqdm import tqdm

from dataloader import MixDataset
from model import MANSF
from utils import build_wiki_relation, accuracy


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
)
parser.add_argument(
    "--fastmode",
    action="store_true",
    default=False,
    help="Validate during training pass.",
)
parser.add_argument(
    "--sparse",
    action="store_true",
    default=False,
    help="GAT with sparse version or not.",
)
parser.add_argument("--seed", type=int, default=14, help="Random seed.")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
parser.add_argument(
    "--batch_size", type=int, default=1, help="Number of batch size to train."
)
parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate.")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument("--hidden", type=int, default=64, help="Number of hidden units.")
parser.add_argument(
    "--nb_heads", type=int, default=8, help="Number of head attentions."
)
parser.add_argument(
    "--dropout", type=float, default=0.38, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)
parser.add_argument("--patience", type=int, default=100, help="Patience")
parser.add_argument("--window", type=int, default=5, help="Window of trading day")
parser.add_argument(
    "--max_tweet_num",
    type=int,
    default=5,
    help="Max number of tweets used for training per day per stock",
)
parser.add_argument(
    "--max_tweet_len", type=int, default=30, help="Max length of tweets embedding"
)
parser.add_argument(
    "--text_ft_dim", type=int, default=384, help="Dimension of tweets embedding"
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def gen_dataset(dataframe_dict: dict, start_date: str, end_date: str):
    """generate data with different date range"""
    for name in dataframe_dict.keys():
        dataframe_dict[name] = (
            dataframe_dict[name].drop_duplicates().reset_index(drop=True)
        )
        dataframe_dict[name] = dataframe_dict[name][
            (dataframe_dict[name]["date"].notnull())
            & (dataframe_dict[name]["date"] >= start_date)
            & (dataframe_dict[name]["date"] <= end_date)
        ].reset_index(drop=True)
    prices, tweets = dataframe_dict["price"], dataframe_dict["tweet"]
    prices = prices.sort_values("date").reset_index(drop=True)
    tweets = (
        tweets.groupby(["stock", "date"], as_index=False)
        .agg({"text": "\n".join})
        .fillna("")
        .reset_index(drop=True)
    )
    mix = (
        pd.merge(prices, tweets, on=["stock", "date"], how="left")
        .fillna("")
        .reset_index(drop=True)
    )
    mix_pv = pd.pivot(mix, index="date", columns="stock").reset_index()
    mix_pv["text"] = mix_pv["text"].fillna("")
    for col in ["movement_perc", "high", "low", "close"]:
        mix_pv[col] = mix_pv[col].fillna(0.0)
    return mix_pv


def train(args, model, trainloader, valloader, edge_index, edge_type):
    model.train()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.BCELoss().cuda()
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        loss_train = 0.0
        metrics = {k: [] for k in ["accuracy", "roc_auc", "f1", "recall", "precision"]}
        for i, data in enumerate(tqdm(trainloader, desc="Training")):
            text_tensor, price_tensor, label_tensor = data
            optimizer.zero_grad()
            # forward pass
            output = model(text_tensor, price_tensor, edge_index, edge_type)
            _output, _label = (
                output.argmax(axis=1).unsqueeze(1).float(),
                label_tensor.squeeze(0).float(),
            )
            _loss_train = Variable(loss_fn(_output, _label), requires_grad=True)
            metrics["accuracy"].append(accuracy_score(_output.cpu(), _label.cpu()))
            metrics["roc_auc"].append(roc_auc_score(_output.cpu(), _label.cpu()))
            metrics["f1"].append(f1_score(_output.cpu(), _label.cpu()))
            metrics["recall"].append(recall_score(_output.cpu(), _label.cpu()))
            metrics["precision"].append(precision_score(_output.cpu(), _label.cpu()))
            # backward
            _loss_train.backward()
            optimizer.step()
            loss_train += _loss_train.mean().item()
            torch.cuda.empty_cache()
        # 每 epoch 計算分類準確率
        # trainset
        print("[Epoch %s] Train Loss: %.3f \n" % (epoch, loss_train))
        # _, labelss, metrics = get_predictions(clf, trainloader, compute_acc=True, compute_loss=False)
        print(
            "    Train F1: %.3f \n" % (np.mean(metrics["f1"])),
            "    Train Recall: %.3f \n" % (np.mean(metrics["recall"])),
            "    Train Precision: %.3f \n" % (np.mean(metrics["precision"])),
            "    Train ROC AUC: %.3f \n " % (np.mean(metrics["roc_auc"])),
            "    Train Accuracy: %.3f \n " % (np.mean(metrics["accuracy"])),
        )
        # valset
        val_preds, val_labelss, val_metrics, loss_val, val_attn = get_predictions(
            model, valloader, edge_index, edge_type, validation=True
        )
        print("[Epoch %s] Val Loss: %.3f \n" % (epoch, loss_val))
        print(
            "    Val F1: %.3f \n" % (np.mean(val_metrics["f1"])),
            "    Val Recall: %.3f \n" % (np.mean(val_metrics["recall"])),
            "    Val Precision: %.3f \n" % (np.mean(val_metrics["precision"])),
            "    Val ROC AUC: %.3f \n " % (np.mean(val_metrics["roc_auc"])),
            "    Val Accuracy: %.3f \n " % (np.mean(val_metrics["accuracy"])),
        )
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": loss_train, "Validation": loss_val},
            epoch * len(trainloader) + i,
        )
        for name in metrics:
            writer.add_scalars(
                f"Training vs. Validation {name.upper()}",
                {
                    "Training": np.mean(metrics[name]),
                    "Validation": np.mean(val_metrics[name]),
                },
                epoch * len(trainloader) + i,
            )
        np.save("output/preds-mansf-stocknet", val_preds.cpu())
        np.save("output/attns-mansf-stocknet", val_attn.cpu())
        torch.save(model.state_dict(), "finetuned/mansf-stocknet.pth")
        writer.flush()


def get_predictions(model, dataloader, edge_index, edge_type, validation=False):
    predictions, attns, labelss = None, None, None
    losses = 0.0
    metrics = {k: [] for k in ["accuracy", "roc_auc", "f1", "recall", "precision"]}
    loss_fn = nn.BCELoss().cuda()
    with torch.no_grad():
        for data in tqdm(dataloader):
            if next(model.parameters()).is_cuda:
                if validation:
                    text_tensor, price_tensor, label_tensor = data
                else:
                    text_tensor, price_tensor = data
                output, _ = model(
                    text_tensor, price_tensor, edge_index, edge_type, return_attn=True
                )
                if validation:
                    _output, _label = (
                        output.argmax(axis=1).unsqueeze(1).float(),
                        label_tensor.squeeze(0).float(),
                    )
                    _loss = loss_fn(_output, _label)
                    losses += _loss.mean().item()
                    metrics["accuracy"].append(
                        accuracy_score(_output.cpu(), _label.cpu())
                    )
                    metrics["roc_auc"].append(
                        roc_auc_score(_output.cpu(), _label.cpu())
                    )
                    metrics["f1"].append(f1_score(_output.cpu(), _label.cpu()))
                    metrics["recall"].append(recall_score(_output.cpu(), _label.cpu()))
                    metrics["precision"].append(
                        precision_score(_output.cpu(), _label.cpu())
                    )
                torch.cuda.empty_cache()
                if predictions is None:
                    predictions, attns = output, _[1]
                    if validation:
                        labelss = label_tensor
                else:
                    predictions = torch.cat((predictions, output))
                    attns = torch.cat((attns, _[1]))
                    if validation:
                        labelss = torch.cat((labelss, label_tensor))
    if validation:
        return predictions, labelss, metrics, losses, attns
    return predictions, attns


def test_dict():
    pred_dict = dict()
    with open("label_data.p", "rb") as fp:
        true_label = pickle.load(fp)
    with open("price_feature_data.p", "rb") as fp:
        feature_data = pickle.load(fp)
    with open("text_feature_data.p", "rb") as fp:
        text_ft_data = pickle.load(fp)
    model.eval()
    test_acc = []
    test_loss = []
    li_pred = []
    li_true = []
    for dates in feature_data.keys():
        test_text = torch.tensor(text_ft_data[dates], dtype=torch.float32).cuda()
        test_price = torch.tensor(feature_data[dates], dtype=torch.float32).cuda()
        test_label = torch.LongTensor(true_label[dates]).cuda()
        output = model(test_text, test_price, edge_index, edge_type)
        output = F.softmax(output, dim=1)
        pred_dict[dates] = output.cpu().detach().numpy()
        loss_test = F.nll_loss(output, torch.max(test_label, 1)[0])
        acc_test = accuracy(output, torch.max(test_label, 1)[1])
        a = torch.max(output, 1)[1].cpu().numpy()
        b = torch.max(test_label, 1)[1].cpu().numpy()
        li_pred.append(a)
        li_true.append(b)
        test_loss.append(loss_test.item())
        test_acc.append(acc_test.item())
    iop = f1_score(
        np.array(li_true).reshape((-1,)),
        np.array(li_pred).reshape((-1,)),
        average="micro",
    )
    mat = matthews_corrcoef(
        np.array(li_true).reshape((-1,)), np.array(li_pred).reshape((-1,))
    )
    print(
        "Test set results:",
        "loss= {:.4f}".format(np.array(test_loss).mean()),
        "accuracy= {:.4f}".format(np.array(test_acc).mean()),
        "F1 score={:.4f}".format(iop),
        "MCC = {:.4f}".format(mat),
    )
    with open("pred_dict.p", "wb") as fp:
        pickle.dump(pred_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return iop, mat


# if __name__ == "__main__":

stock_dir = "stocknet-dataset/price/preprocessed"
tweet_dir = "stocknet-dataset/tweet_preprocessed"
rels_dir = "Temporal_Relational_Stock_Ranking/data/relation/wikidata"
market_names = ["NASDAQ", "NYSE"]
args.stock_list = sorted(
    list(
        set(
            [path.split("/")[-1].replace(".txt", "") for path in glob(stock_dir + "/*")]
        )
        & set([path.split("/")[-1] for path in glob(tweet_dir + "/*")])
    )
)
args.n_stock = len(args.stock_list)  # the number of stocks
n_day = 5  # the backward-looking window T
n_tweet = 5  # max num of tweets per day, I suppose 1 tweet per stock per day
n_price_feat = 3  # price feature dim  (normalized high/low/close)
n_tweet_feat = 384  # text embedding dim

prices, tweets = pd.DataFrame(), pd.DataFrame()
for stock in tqdm(args.stock_list):
    _p, _t = pd.read_table(
        os.path.join(stock_dir, f"{stock}.txt"), header=None
    ), pd.read_json(os.path.join(tweet_dir, f"{stock}"), orient="records", lines=True)
    _p["stock"], _t["stock"] = stock, stock
    prices, tweets = pd.concat([prices, _p]), pd.concat([tweets, _t])

prices.columns = [
    "date",
    "movement_perc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "stock",
]
prices = prices.drop(["open", "volume"], axis=1)
tweets["date"] = tweets.created_at.apply(lambda x: x.date())
tweets.date = tweets.date.astype(str)

mix_pv_train = gen_dataset(
    dataframe_dict={"price": prices, "tweet": tweets},
    start_date="2014-01-01",
    end_date="2015-06-30",
)
mix_pv_val = gen_dataset(
    dataframe_dict={"price": prices, "tweet": tweets},
    start_date="2015-07-01",
    end_date="2015-12-31",
)
mix_pv_test = gen_dataset(
    dataframe_dict={"price": prices, "tweet": tweets},
    start_date="2016-01-01",
    end_date="2016-03-31",
)

# preprocess relation data & load
adj = build_wiki_relation(rels_dir, market_names, args.stock_list)
edge_index = torch.index_select(torch.nonzero(adj).t(), 0, torch.tensor([0, 1]))
edge_type = torch.index_select(torch.nonzero(adj).t(), 0, torch.tensor([2])).squeeze(0)

model = MANSF(
    nfeat=64,
    nhid=args.hidden,
    nrel=adj.shape[2],
    nclass=2,
    dropout=args.dropout,
    nheads=args.nb_heads,
    alpha=args.alpha,
    stock_num=args.n_stock,
    text_ft_dim=args.text_ft_dim,
)
# model = nn.DataParallel(model, device_ids=[0, 1])
if args.cuda:
    model.cuda()
    edge_index = edge_index.type(torch.LongTensor).cuda()
    edge_type = edge_type.type(torch.LongTensor).cuda()
    # adj = adj.type(torch.LongTensor).cuda()
    args.device = "cuda"

trainset = MixDataset(
    mode="train",
    data=mix_pv_train,
    window_num=args.window,
    max_tweet_num=args.max_tweet_num,
    max_tweet_len=args.max_tweet_len,
    stock_list=args.stock_list,
)
valset = MixDataset(
    mode="val",
    data=mix_pv_val,
    window_num=args.window,
    max_tweet_num=args.max_tweet_num,
    max_tweet_len=args.max_tweet_len,
    stock_list=args.stock_list,
)
testset = MixDataset(
    mode="test",
    data=mix_pv_test,
    window_num=args.window,
    max_tweet_num=args.max_tweet_num,
    max_tweet_len=args.max_tweet_len,
    stock_list=args.stock_list,
)
trainsampler = RandomSampler(trainset)
valsampler = RandomSampler(valset)
testsampler = RandomSampler(testset)
trainloader = DataLoader(
    trainset, sampler=trainsampler, batch_size=args.batch_size, drop_last=True
)
valloader = DataLoader(
    valset, sampler=valsampler, batch_size=args.batch_size, drop_last=True
)
testloader = DataLoader(
    testset, sampler=testsampler, batch_size=args.batch_size, drop_last=True
)
print(len(trainset), len(valset), len(testset))

train(args, model, trainloader, valloader, edge_index, edge_type)
print("Optimization Finished!")
results = test_dict()
print(results)
