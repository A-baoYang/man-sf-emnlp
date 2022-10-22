import json
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def build_wiki_relation(rels_dir, market_names, stock_pool):

    output_path = os.path.join(rels_dir, "total_wiki_relation")
    if os.path.exists(output_path):
        wiki_relation_embedding = np.load(output_path)

    else:
        tickers, sel_paths = np.array([]), np.array([])
        connections = dict()
        for market_name in market_names:
            connection_file = os.path.join(rels_dir, f"{market_name}_connections.json")
            tic_wiki_file = os.path.join(rels_dir, f"{market_name}_wiki.csv")
            sel_path_file = os.path.join(rels_dir, "selected_wiki_connections.csv")
            # readin tickers
            if not tickers.size:
                tickers = np.genfromtxt(
                    tic_wiki_file, dtype=str, delimiter=",", skip_header=False
                )
            else:
                tickers = np.append(
                    tickers,
                    np.genfromtxt(
                        tic_wiki_file, dtype=str, delimiter=",", skip_header=False
                    ),
                    axis=0,
                )
            # readin selected paths/connections
            if not sel_paths.size:
                sel_paths = np.genfromtxt(
                    sel_path_file, dtype=str, delimiter=" ", skip_header=False
                )
            else:
                sel_paths = np.append(
                    sel_paths,
                    np.genfromtxt(
                        sel_path_file, dtype=str, delimiter=" ", skip_header=False
                    ),
                    axis=0,
                )
            # readin connections
            with open(connection_file, "r") as fin:
                connections.update(json.load(fin))

        # fllter tickers with stock pool
        _ = pd.DataFrame(tickers)
        t_intersect = set(_[0].tolist()) & set(stock_pool)
        t_missed = set(stock_pool) - t_intersect
        tickers = _[_[0].isin(t_intersect)].reset_index(drop=True)
        tickers = (
            pd.concat([tickers, pd.DataFrame({0: list(t_missed), 1: ["unknown"] * 5})])
            .sort_values(0)
            .reset_index(drop=True)
        ).values
        print("#tickers selected:", tickers.shape)
        print("#paths selected:", len(sel_paths))
        sel_paths = set(sel_paths[:, 0])
        print("#connection items:", len(connections))

        # tickers aligned
        wikiid_ticind_dic = {
            tw[-1]: ind for ind, tw in enumerate(tickers) if not tw[-1] == "unknown"
        }
        print("#tickers aligned:", len(wikiid_ticind_dic))

        # get occured paths
        occur_paths = set()
        for sou_item, conns in connections.items():
            for tar_item, paths in conns.items():
                for p in paths:
                    path_key = "_".join(p)
                    if path_key in sel_paths:
                        occur_paths.add(path_key)

        # generate
        valid_path_index = {path: ind for ind, path in enumerate(occur_paths)}
        print("#valid paths:", len(valid_path_index))

        wiki_relation_embedding = np.zeros(
            [tickers.shape[0], tickers.shape[0], len(valid_path_index) + 1], dtype=int
        )
        conn_count = 0
        for sou_item, conns in connections.items():
            for tar_item, paths in conns.items():
                for p in paths:
                    path_key = "_".join(p)
                    if (
                        path_key in valid_path_index
                        and sou_item in wikiid_ticind_dic
                        and tar_item in wikiid_ticind_dic
                    ):
                        aaa = wikiid_ticind_dic[sou_item]
                        bbb = wikiid_ticind_dic[tar_item]
                        ccc = valid_path_index[path_key]

                        wiki_relation_embedding[aaa][bbb][ccc] = 1
                        conn_count += 1
        print(
            "connections count:",
            conn_count,
            "ratio:",
            conn_count / float(tickers.shape[0] * tickers.shape[0]),
        )

        # handle self relation
        for i in range(tickers.shape[0]):
            wiki_relation_embedding[i][i][-1] = 1
        print(wiki_relation_embedding.shape)
        np.save(os.path.join(rels_dir, "total_wiki_relation"), wiki_relation_embedding)

    return torch.tensor(wiki_relation_embedding, dtype=torch.int8)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data():
    print("Loading dataset...")
    adj = np.load("graph.npy")
    adj = sp.coo_matrix(adj, dtype=np.float32)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
