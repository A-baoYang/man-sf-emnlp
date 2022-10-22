import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv
from torch_geometric.nn import global_mean_pool

# from layers import GraphAttentionLayer, SpGraphAttentionLayer


class gru(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(gru, self).__init__()
        self.gru1 = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )

    def forward(self, inputs):
        full, last = self.gru1(inputs)
        return full, last


class attn(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(attn, self).__init__()
        self.W1 = nn.Linear(in_shape, out_shape)
        self.W2 = nn.Linear(in_shape, out_shape)
        self.V = nn.Linear(in_shape, 1)

    def forward(self, full, last):
        score = self.V(torch.tanh(self.W1(last) + self.W2(full)))
        attention_weights = F.softmax(score, dim=1)
        context_vector = attention_weights * full
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector


class MANSF(nn.Module):
    def __init__(
        self, nfeat, nhid, nrel, nclass, dropout, alpha, nheads, stock_num, text_ft_dim
    ):
        super(MANSF, self).__init__()
        self.nhid = nhid
        self.grup = [gru(3, self.nhid) for _ in range(stock_num)]
        for i, gru_p in enumerate(self.grup):
            self.add_module("gru_p{}".format(i), gru_p)

        self.attnp = [attn(self.nhid, self.nhid) for _ in range(stock_num)]
        for i, attn_p in enumerate(self.attnp):
            self.add_module("attn_p{}".format(i), attn_p)

        self.tweet_gru = [gru(text_ft_dim, self.nhid) for _ in range(stock_num)]
        for i, tweet_gru_ in enumerate(self.tweet_gru):
            self.add_module("tweet_gru{}".format(i), tweet_gru_)

        self.grut = [gru(self.nhid, self.nhid) for _ in range(stock_num)]
        for i, gru_t in enumerate(self.grut):
            self.add_module("gru_t{}".format(i), gru_t)

        self.attn_tweet = [attn(self.nhid, self.nhid) for _ in range(stock_num)]
        for i, attn_tweet_ in enumerate(self.attn_tweet):
            self.add_module("attn_tweet{}".format(i), attn_tweet_)

        self.attnt = [attn(self.nhid, self.nhid) for _ in range(stock_num)]
        for i, attnt_ in enumerate(self.attnt):
            self.add_module("attnt{}".format(i), attnt_)

        self.bilinear = [
            nn.Bilinear(self.nhid, self.nhid, self.nhid) for _ in range(stock_num)
        ]
        for i, bilinear_ in enumerate(self.bilinear):
            self.add_module("bilinear{}".format(i), bilinear_)

        self.layer_normt = [nn.LayerNorm((1, self.nhid)) for _ in range(stock_num)]
        for i, layer_normt_ in enumerate(self.layer_normt):
            self.add_module("layer_normt{}".format(i), layer_normt_)

        self.layer_normp = [nn.LayerNorm((1, self.nhid)) for _ in range(stock_num)]
        for i, layer_normp_ in enumerate(self.layer_normp):
            self.add_module("layer_normp{}".format(i), layer_normp_)

        # self.linear_x = [nn.Linear(self.nhid, 2) for _ in range(stock_num)]
        # for i, linear_x_ in enumerate(self.linear_x):
        #     self.add_module("linear_x{}".format(i), linear_x_)
        self.linear_x = nn.Linear(self.nhid, 2)
        self.dropout_ratio = dropout
        # self.attentions = [
        #     RGATConv(nfeat, self.nhid, nrel, dropout=self.dropout_ratio, alpha=alpha, concat=True)
        #     for _ in range(nheads)
        # ]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module("attention_{}".format(i), attention)
        self.attentions = RGATConv(
            nfeat,
            self.nhid,
            nrel,
            heads=nheads,
            dropout=self.dropout_ratio,
            alpha=alpha,
            concat=True,
        )
        self.out_att = RGATConv(
            self.nhid * nheads,
            nclass,
            nrel,
            heads=1,
            dropout=self.dropout_ratio,
            alpha=alpha,
            concat=False,
        )
        self.out_classify = nn.Softmax(dim=1)

    def forward(
        self, text_input, price_input, edge_index, edge_type, return_attn=False
    ):
        li = []
        batch_size = text_input.size(0)
        num_tw, tw_ft_dim = text_input.size(3), text_input.size(4)
        num_d = price_input.size(2)
        pr_ft = price_input.size(3)
        num_stocks = price_input.size(1)
        for i in range(num_stocks):  # n_stock
            # price data
            x = self.grup[i](
                price_input[:, i, :, :].reshape((batch_size, num_d, pr_ft)).float()
            )  # [[(b, 5, 64), (1, b, 64)], [(b, 5, 64), (1, b, 64)], ,,,]
            x = (x[0], x[1].reshape((batch_size, 1, self.nhid)))
            x = self.attnp[i](*x).reshape((batch_size, self.nhid))  # (b, 64)
            # x = self.layer_normp[i](x).reshape(batch_size, 64)
            han_li1 = []
            for j in range(num_d):  # n_day
                # tweet of each day
                y = self.tweet_gru[i](
                    text_input[:, i, j, :, :].reshape((batch_size, num_tw, tw_ft_dim))
                )  # [(b, num_tw, emb_dim), (1, b, emb_dim),...]
                y = (y[0], y[1].reshape((batch_size, 1, self.nhid)))
                y = self.attn_tweet[i](*y).reshape((batch_size, self.nhid))  # (b, 64)
                han_li1.append(y)
            # tweets in window days
            # news_vector = torch.Tensor((batch_size, num_d, 64))
            news_vector = torch.cat(han_li1)  # (b * 5, 64)
            text = self.grut[i](
                news_vector.reshape(batch_size, num_d, self.nhid)
            )  # [(b, 5, 64), (1, b, 64),...]
            text = (text[0], text[1].reshape((batch_size, 1, self.nhid)))
            text = self.attnt[i](*text).reshape((batch_size, self.nhid))  # (b, 64)
            # tweet X price
            combined = torch.tanh(
                self.bilinear[i](text, x).reshape((batch_size, self.nhid))
            )  # (b, 64)
            li.append(combined.reshape(batch_size, self.nhid))

        # ft_vec = torch.Tensor((batch_size, num_stocks, 64))
        ft_vec = torch.cat(li).reshape(
            (batch_size, num_stocks, self.nhid)
        )  # (b, n_stock, 64)
        out_1 = torch.tanh(self.linear_x(ft_vec)).squeeze(0)  # (n_stock, 2)
        x = F.dropout(ft_vec, self.dropout_ratio)  # (b, n_stock, 64)
        x = self.attentions(x.squeeze(0), edge_index, edge_type)  # (n_stock, 64)
        # x = torch.cat(
        #     [att(x, edge_index, edge_type) for att in self.attentions], dim=1
        # )  # (b, n_stock, 64*8)
        x = F.dropout(x, self.dropout_ratio)  # (n_stock, 64)
        if return_attn:
            x, _ = self.out_att(x, edge_index, edge_type, return_attention_weights=True)
        else:
            x = self.out_att(x, edge_index, edge_type)
        x = F.elu(x)  # (n_stock, 2)
        if return_attn:
            return self.out_classify(x + out_1), _  # (n_stock, 2)
        else:
            return self.out_classify(x + out_1)  # (n_stock, 2)
