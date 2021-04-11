'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import coo_matrix, vstack

def parse_index_file(filename):
    index = []
    for line in open(filename):
        # Remove leading and tailing spaces.
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)

    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)
    print(len(test_idx_reorder), len(test_idx_range))
    print(test_idx_range)
    #print(test_idx_range)
    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features

def load_from_networkx1(graph):
    #graph = pkl.load(graph)
    adj_mat = nx.adjacency_matrix(graph)

    nodes_list = list(graph.nodes)
    features_mat = np.zeros(shape=(len(nodes_list), max(nodes_list)+1))

    name2id = {}
    for idx in range(len(nodes_list)):
        features_mat[idx][nodes_list[idx]] = 1
        name2id[nodes_list[idx]] = idx
    return coo_matrix(adj_mat), coo_matrix(features_mat), name2id


def load_from_networkx(edges_path, aug=False):
    with open(edges_path, 'r') as f:
        edges = f.readlines()

    edge_tuple_list = []
    for line in edges:
        line = line.strip()
        source, sink = line.split(" ")
        edge_tuple_list.append((int(source), int(sink)))
    G = nx.Graph()
    G.add_edges_from(edge_tuple_list)

    adj_mat = nx.adjacency_matrix(G)
    nodes_list = list(G.nodes)
    features = sp.identity(adj_mat.shape[0])

    # generate name2id dict
    name2id = {}
    for idx in range(len(nodes_list)):
        name2id[nodes_list[idx]] = idx

    if aug:
        adj_mat = adj_mat.toarray()
        for line in edges:
            line = line.strip()
            source, sink = line.split(" ")
            adj_mat[name2id[int(source)]][name2id[int(sink)]] += 1
        adj_mat = coo_matrix(adj_mat)
    return adj_mat, features, name2id


if __name__ == '__main__':
    path = './edges.txt'
    adj, feature, name2id = load_from_networkx(path, aug=True)
    print(type(adj))
    print(adj)
