import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from typing import Tuple, List, Union, Optional

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj: np.ndarray, sizes: List[int], nhood: int = 1) -> np.ndarray:
    """
    隣接行列をバイアスベクトルに変換する関数
    
    Args:
        adj: 隣接行列 [graph, nodes, nodes]
        sizes: 各グラフのノード数
        nhood: 近傍の範囲
    
    Returns:
        バイアス行列
    """
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        
        # より効率的な方法で二値化
        mt[g] = (mt[g][:sizes[g], :sizes[g]] > 0.0).astype(float)
    
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename: str) -> List[int]:
    """Parse index file."""
    index = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                index.append(int(line.strip()))
    except FileNotFoundError:
        raise FileNotFoundError(f"Index file {filename} not found")
    except ValueError as e:
        raise ValueError(f"Invalid integer in index file: {e}")
    
    return index


def sample_mask(idx: List[int], l: int) -> np.ndarray:
    """Create mask."""
    mask = np.zeros(l, dtype=bool)  # 直接boolで初期化
    mask[idx] = True
    return mask


def load_data(dataset_str: str) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, 
                                       np.ndarray, np.ndarray, np.ndarray, 
                                       np.ndarray, np.ndarray]:
    """
    データセット（pubmed, citeseer, cora）を読み込む
    
    Args:
        dataset_str: データセット名
    
    Returns:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    """
    if dataset_str not in {'pubmed', 'citeseer', 'cora'}:
        raise ValueError(f"Unsupported dataset: {dataset_str}")
    
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    
    try:
        for name in names:
            filename = f"data/ind.{dataset_str}.{name}"
            with open(filename, 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Dataset file not found: {e}")

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(f"data/ind.{dataset_str}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    # Citeseerデータセットの特別な処理
    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    # 特徴量の結合と並び替え
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    
    # 隣接行列の作成
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # ラベルの結合と並び替え
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # インデックスの設定
    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))

    # マスクの作成
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    # ラベルのマスク適用
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Features shape: {features.shape}")

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_random_data(size: int, num_classes: int = 7, 
                    adj_density: float = 0.002, 
                    feature_density: float = 0.015,
                    feature_dim: int = 1000) -> Tuple[sp.csr_matrix, sp.csr_matrix, 
                                                     np.ndarray, np.ndarray, np.ndarray, 
                                                     np.ndarray, np.ndarray, np.ndarray]:
    """
    ランダムなグラフデータを生成する
    
    Args:
        size: ノード数
        num_classes: クラス数
        adj_density: 隣接行列の密度
        feature_density: 特徴量行列の密度
        feature_dim: 特徴量の次元数
    
    Returns:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    """
    # ランダムシードの設定（再現性のため）
    np.random.seed(42)
    
    adj = sp.random(size, size, density=adj_density, format='csr')
    # 対称行列にする
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)  # 二値化
    
    features = sp.random(size, feature_dim, density=feature_density, format='csr')
    
    # ラベルの生成
    int_labels = np.random.randint(num_classes, size=size)
    labels = np.zeros((size, num_classes))
    labels[np.arange(size), int_labels] = 1

    # データ分割の改善
    train_size = int(size * 0.6)
    val_size = int(size * 0.2)
    
    # インデックスをシャッフル
    indices = np.random.permutation(size)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    # マスクの作成
    train_mask = sample_mask(train_idx, size)
    val_mask = sample_mask(val_idx, size)
    test_mask = sample_mask(test_idx, size)

    # ラベルのマスク適用
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx: Union[sp.spmatrix, List[sp.spmatrix]]) -> Union[Tuple, List[Tuple]]:
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx: sp.spmatrix) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        return [to_tuple(mx) for mx in sparse_mx]
    else:
        return to_tuple(sparse_mx)


def standardize_data(f: sp.spmatrix, train_mask: np.ndarray) -> np.ndarray:
    """
    特徴量行列を標準化する
    
    Args:
        f: 特徴量行列
        train_mask: 訓練データのマスク
    
    Returns:
        標準化された特徴量行列
    """
    f = f.todense()
    
    # 訓練データのみで統計量を計算
    train_data = f[train_mask, :]
    mu = np.mean(train_data, axis=0)
    sigma = np.std(train_data, axis=0)
    
    # 分散が0の特徴量を除去
    valid_features = np.squeeze(np.array(sigma > 1e-6))
    f = f[:, valid_features]
    
    # 再計算
    train_data = f[train_mask, :]
    mu = np.mean(train_data, axis=0)
    sigma = np.std(train_data, axis=0)
    
    # 標準化
    f = (f - mu) / (sigma + 1e-8)  # 数値安定性のための小さな値を追加
    
    return f


def preprocess_features(features: sp.spmatrix) -> Tuple[np.ndarray, Tuple]:
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj: sp.spmatrix) -> sp.coo_matrix:
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj: sp.spmatrix) -> Tuple:
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def preprocess_adj_bias(adj: sp.spmatrix) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    隣接行列をバイアス用に前処理する
    
    Args:
        adj: 隣接行列
    
    Returns:
        indices, values, shape のタプル
    """
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # セルフループの追加
    adj.data = np.ones_like(adj.data)  # 二値化
    
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    
    adj = adj.astype(np.float32)
    # 元のコメントで指摘されている通り、正しい順序で作成
    indices = np.vstack((adj.row, adj.col)).transpose()
    
    return indices, adj.data, adj.shape


# 新しいユーティリティ関数
def get_data_stats(adj: sp.spmatrix, features: sp.spmatrix, labels: np.ndarray) -> dict:
    """
    データセットの統計情報を取得する
    
    Args:
        adj: 隣接行列
        features: 特徴量行列
        labels: ラベル
    
    Returns:
        統計情報の辞書
    """
    num_nodes = adj.shape[0]
    num_edges = adj.nnz
    num_features = features.shape[1]
    num_classes = labels.shape[1]
    
    # グラフの密度
    density = num_edges / (num_nodes * (num_nodes - 1))
    
    # 特徴量の密度
    feature_density = features.nnz / (features.shape[0] * features.shape[1])
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'num_features': num_features,
        'num_classes': num_classes,
        'graph_density': density,
        'feature_density': feature_density
    }