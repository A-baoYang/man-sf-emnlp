import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

# from transformers import BertTokenizer, BertModel


class MixDataset(object):
    def __init__(
        self,
        mode: str,
        data: pd.DataFrame,
        window_num: int,
        max_tweet_num: int,
        max_tweet_len: int,
        stock_list: list,
    ):

        self.mode = mode
        self.price_data = (
            data[["high", "low", "close"]].swaplevel(0, 1, axis=1).sort_index(axis=1)
        )
        self.tweet_data = data["text"].to_dict("records")
        if self.mode != "test":
            self.label_data = data["movement_perc"]
        self.window_num = window_num
        self.max_tweet_num = max_tweet_num
        self.max_tweet_len = max_tweet_len
        self.stock_list = stock_list
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    def __getitem__(self, idx):

        # if idx + 5 < self.label_data.shape[0]:

        tweet_d = self.tweet_data[idx : idx + self.window_num]
        tweets_emb, price_val = torch.tensor([]).cuda(), torch.DoubleTensor([]).cuda()

        if self.mode != "test":
            label_val = torch.tensor([]).cuda()

        for stock in tqdm(self.stock_list, desc="Collect from all stocks"):

            s = [_d[stock] for _d in tweet_d]
            s = [txt.split("\n") if type(txt) == str else [] for txt in s]
            s = [
                txt[: self.max_tweet_num]
                if len(txt) >= self.max_tweet_num
                else txt + [""] * (self.max_tweet_num - len(txt))
                for txt in s
            ]
            s = [self.model.encode(txt, convert_to_tensor=True) for txt in s]
            s_emb = torch.tensor([]).cuda()
            for i in range(len(s)):
                s_emb = torch.cat((s_emb, s[i].unsqueeze(0)), 0)

            tweets_emb = torch.cat((tweets_emb, s_emb.unsqueeze(0)), 0)
            price_values = self.price_data[stock].values[idx : idx + self.window_num]
            price_val = torch.cat(
                (price_val, torch.tensor(price_values).unsqueeze(0).cuda()), axis=0
            )

            if self.mode != "test":
                label_values = (
                    0
                    if self.label_data[stock].values[
                        idx + self.window_num : idx + self.window_num + 1
                    ]
                    <= 0
                    else 1
                )
                label_val = torch.cat(
                    (label_val, torch.tensor(label_values).unsqueeze(0).cuda()), axis=0
                )

        if self.mode != "test":
            return tweets_emb, price_val, label_val.unsqueeze(-1)
        else:
            return tweets_emb, price_val

    def __len__(self):
        return len(self.price_data)


class TextDataset(object):
    def __init__(
        self,
        data: pd.DataFrame,
        max_tweet_num: int,
        max_tweet_len: int,
        stock_list: list,
    ):

        self.data = data.to_dict(orient="records")
        self.max_tweet_num = max_tweet_num
        self.max_tweet_len = max_tweet_len
        self.stock_list = stock_list
        # self.tokenizer = BertTokenizer.from_pretrained("distilbert-base-cased")
        # self.model = BertModel.from_pretrained("distilbert-base-cased")
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    def __getitem__(self, idx):

        d = self.data[idx : idx + self.window_num]
        stock_tweets_emb = torch.tensor([]).cuda()
        for stock in tqdm(self.stock_list):
            s = [_d[stock] for _d in d]
            s = [txt.split("\n") if type(txt) == str else [] for txt in s]
            s = [
                txt[: self.max_tweet_num]
                if len(txt) >= self.max_tweet_num
                else txt + [""] * (self.max_tweet_num - len(txt))
                for txt in s
            ]
            # s = [
            #     self.tokenizer.batch_encode_plus(
            #         txt,
            #         add_special_tokens=True,
            #         max_length=self.max_tweet_len,
            #         return_token_type_ids=True,
            #         padding='max_length',
            #         truncation=True,
            #         return_attention_mask=True,
            #         return_tensors="pt") for txt in s]
            # s = [self.model(**b).pooler_output for b in s]
            s = [self.model.encode(txt, convert_to_tensor=True) for txt in s]
            s_emb = torch.tensor([]).cuda()
            for i in range(len(s)):
                s_emb = torch.cat((s_emb, s[i].unsqueeze(0)), 0)
            stock_tweets_emb = torch.cat((stock_tweets_emb, s_emb.unsqueeze(0)), 0)

        return stock_tweets_emb

    def __len__(self):
        return len(self.data)


class StockDataset(object):
    def __init__(self, data: pd.DataFrame, window_num: int, stock_list: list):

        self.data = data
        self.stock_list = stock_list
        self.window_num = window_num

    def __getitem__(self, idx):

        arr = torch.tensor([]).cuda()
        for stock in self.stock_list:
            price_values = self.data[stock].values[idx : idx + self.window_num]
            arr = torch.cat(
                (arr, torch.tensor(price_values).unsqueeze(0).cuda()), axis=0
            )
        return arr

    def __len__(self):
        return len(self.data)
