import time
import os
import scipy.sparse as sp
import numpy as np
import tensorflow as tf
import argparse
from typing import Dict, Tuple, Any
import logging

from models import GAT
from models import SpGAT
from utils import process

# ログの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GATTrainer:
    """GAT（Graph Attention Network）の訓練クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.dataset = config['dataset']
        self.checkpt_file = config['checkpt_file']
        
        # 訓練パラメータ
        self.batch_size = config.get('batch_size', 1)
        self.nb_epochs = config.get('nb_epochs', 100000)
        self.patience = config.get('patience', 100)
        self.lr = config.get('lr', 0.005)
        self.l2_coef = config.get('l2_coef', 0.0005)
        
        # モデルパラメータ
        self.hid_units = config.get('hid_units', [8])
        self.n_heads = config.get('n_heads', [8, 1])
        self.residual = config.get('residual', False)
        self.nonlinearity = config.get('nonlinearity', tf.nn.elu)
        self.model_type = config.get('model', SpGAT)
        self.sparse = config.get('sparse', True)
        
        # ドロップアウト率
        self.attn_drop_rate = config.get('attn_drop', 0.6)
        self.ffd_drop_rate = config.get('ffd_drop', 0.6)
        
        # データ関連の変数
        self.adj = None
        self.features = None
        self.biases = None
        self.nb_nodes = None
        self.ft_size = None
        self.nb_classes = None
        
        # 訓練データ
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        
        self._print_config()
        
    def _print_config(self):
        """設定を出力"""
        logger.info(f'Dataset: {self.dataset}')
        logger.info('----- Optimization hyperparameters -----')
        logger.info(f'Learning rate: {self.lr}')
        logger.info(f'L2 coefficient: {self.l2_coef}')
        logger.info(f'Batch size: {self.batch_size}')
        logger.info(f'Max epochs: {self.nb_epochs}')
        logger.info(f'Patience: {self.patience}')
        logger.info('----- Architecture hyperparameters -----')
        logger.info(f'Number of layers: {len(self.hid_units)}')
        logger.info(f'Hidden units per layer: {self.hid_units}')
        logger.info(f'Attention heads: {self.n_heads}')
        logger.info(f'Residual connections: {self.residual}')
        logger.info(f'Nonlinearity: {self.nonlinearity}')
        logger.info(f'Model: {self.model_type.__name__}')
        logger.info(f'Sparse: {self.sparse}')
        
    def load_data(self):
        """データの読み込みと前処理"""
        logger.info("Loading data...")
        
        # データの読み込み
        self.adj, features, self.y_train, self.y_val, self.y_test, \
        self.train_mask, self.val_mask, self.test_mask = process.load_data(self.dataset)
        
        # 特徴量の前処理
        self.features, _ = process.preprocess_features(features)
        
        # データの形状を取得
        self.nb_nodes = self.features.shape[0]
        self.ft_size = self.features.shape[1]
        self.nb_classes = self.y_train.shape[1]
        
        logger.info(f"Number of nodes: {self.nb_nodes}")
        logger.info(f"Feature size: {self.ft_size}")
        logger.info(f"Number of classes: {self.nb_classes}")
        
        # バッチ次元を追加
        self.features = self.features[np.newaxis]
        self.y_train = self.y_train[np.newaxis]
        self.y_val = self.y_val[np.newaxis]
        self.y_test = self.y_test[np.newaxis]
        self.train_mask = self.train_mask[np.newaxis]
        self.val_mask = self.val_mask[np.newaxis]
        self.test_mask = self.test_mask[np.newaxis]
        
        # バイアス行列の前処理
        if self.sparse:
            self.biases = process.preprocess_adj_bias(self.adj)
        else:
            adj_dense = self.adj.todense()
            adj_dense = adj_dense[np.newaxis]
            self.biases = process.adj_to_bias(adj_dense, [self.nb_nodes], nhood=1)
            
    def build_model(self):
        """モデルの構築"""
        logger.info("Building model...")
        
        # プレースホルダーの定義
        with tf.name_scope('input'):
            self.ftr_in = tf.placeholder(
                dtype=tf.float32, 
                shape=(self.batch_size, self.nb_nodes, self.ft_size),
                name='features'
            )
            
            if self.sparse:
                self.bias_in = tf.sparse_placeholder(dtype=tf.float32, name='bias')
            else:
                self.bias_in = tf.placeholder(
                    dtype=tf.float32, 
                    shape=(self.batch_size, self.nb_nodes, self.nb_nodes),
                    name='bias'
                )
                
            self.lbl_in = tf.placeholder(
                dtype=tf.int32, 
                shape=(self.batch_size, self.nb_nodes, self.nb_classes),
                name='labels'
            )
            self.msk_in = tf.placeholder(
                dtype=tf.int32, 
                shape=(self.batch_size, self.nb_nodes),
                name='mask'
            )
            self.attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_dropout')
            self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_dropout')
            self.is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')
        
        # モデルの推論
        self.logits = self.model_type.inference(
            self.ftr_in, self.nb_classes, self.nb_nodes, self.is_train,
            self.attn_drop, self.ffd_drop,
            bias_mat=self.bias_in,
            hid_units=self.hid_units, 
            n_heads=self.n_heads,
            residual=self.residual, 
            activation=self.nonlinearity
        )
        
        # 損失関数とメトリクスの定義
        log_resh = tf.reshape(self.logits, [-1, self.nb_classes])
        lab_resh = tf.reshape(self.lbl_in, [-1, self.nb_classes])
        msk_resh = tf.reshape(self.msk_in, [-1])
        
        self.loss = self.model_type.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        self.accuracy = self.model_type.masked_accuracy(log_resh, lab_resh, msk_resh)
        
        # 最適化器
        self.train_op = self.model_type.training(self.loss, self.lr, self.l2_coef)
        
        # セーバー
        self.saver = tf.train.Saver(max_to_keep=1)
        
    def get_feed_dict(self, step: int, mode: str) -> Dict[str, Any]:
        """フィードディクショナリを取得"""
        start_idx = step * self.batch_size
        end_idx = (step + 1) * self.batch_size
        
        if self.sparse:
            bbias = self.biases
        else:
            bbias = self.biases[start_idx:end_idx]
            
        if mode == 'train':
            labels = self.y_train[start_idx:end_idx]
            mask = self.train_mask[start_idx:end_idx]
            is_training = True
            attn_drop = self.attn_drop_rate
            ffd_drop = self.ffd_drop_rate
        elif mode == 'val':
            labels = self.y_val[start_idx:end_idx]
            mask = self.val_mask[start_idx:end_idx]
            is_training = False
            attn_drop = 0.0
            ffd_drop = 0.0
        else:  # test
            labels = self.y_test[start_idx:end_idx]
            mask = self.test_mask[start_idx:end_idx]
            is_training = False
            attn_drop = 0.0
            ffd_drop = 0.0
            
        return {
            self.ftr_in: self.features[start_idx:end_idx],
            self.bias_in: bbias,
            self.lbl_in: labels,
            self.msk_in: mask,
            self.is_train: is_training,
            self.attn_drop: attn_drop,
            self.ffd_drop: ffd_drop
        }
        
    def train_epoch(self, sess: tf.Session) -> Tuple[float, float]:
        """1エポックの訓練"""
        tr_step = 0
        tr_size = self.features.shape[0]
        train_loss_total = 0.0
        train_acc_total = 0.0
        
        while tr_step * self.batch_size < tr_size:
            feed_dict = self.get_feed_dict(tr_step, 'train')
            
            _, loss_value, acc_value = sess.run(
                [self.train_op, self.loss, self.accuracy],
                feed_dict=feed_dict
            )
            
            train_loss_total += loss_value
            train_acc_total += acc_value
            tr_step += 1
            
        return train_loss_total / tr_step, train_acc_total / tr_step
        
    def validate_epoch(self, sess: tf.Session) -> Tuple[float, float]:
        """検証"""
        vl_step = 0
        vl_size = self.features.shape[0]
        val_loss_total = 0.0
        val_acc_total = 0.0
        
        while vl_step * self.batch_size < vl_size:
            feed_dict = self.get_feed_dict(vl_step, 'val')
            
            loss_value, acc_value = sess.run(
                [self.loss, self.accuracy],
                feed_dict=feed_dict
            )
            
            val_loss_total += loss_value
            val_acc_total += acc_value
            vl_step += 1
            
        return val_loss_total / vl_step, val_acc_total / vl_step
        
    def test(self, sess: tf.Session) -> Tuple[float, float]:
        """テスト"""
        ts_step = 0
        ts_size = self.features.shape[0]
        test_loss_total = 0.0
        test_acc_total = 0.0
        
        while ts_step * self.batch_size < ts_size:
            feed_dict = self.get_feed_dict(ts_step, 'test')
            
            loss_value, acc_value = sess.run(
                [self.loss, self.accuracy],
                feed_dict=feed_dict
            )
            
            test_loss_total += loss_value
            test_acc_total += acc_value
            ts_step += 1
            
        return test_loss_total / ts_step, test_acc_total / ts_step
        
    def train(self):
        """訓練のメイン関数"""
        logger.info("Starting training...")
        
        # チェックポイントディレクトリの作成
        os.makedirs(os.path.dirname(self.checkpt_file), exist_ok=True)
        
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        # 早期停止の変数
        best_val_loss = np.inf
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        with tf.Session() as sess:
            sess.run(init_op)
            
            start_time = time.time()
            
            for epoch in range(self.nb_epochs):
                epoch_start_time = time.time()
                
                # 訓練
                train_loss, train_acc = self.train_epoch(sess)
                
                # 検証
                val_loss, val_acc = self.validate_epoch(sess)
                
                epoch_time = time.time() - epoch_start_time
                
                # 結果の出力
                if epoch % 10 == 0 or epoch < 10:
                    logger.info(
                        f'Epoch {epoch:4d} | '
                        f'Train: loss={train_loss:.5f}, acc={train_acc:.5f} | '
                        f'Val: loss={val_loss:.5f}, acc={val_acc:.5f} | '
                        f'Time: {epoch_time:.2f}s'
                    )
                
                # 早期停止の判定
                improved = False
                if val_acc >= best_val_acc or val_loss <= best_val_loss:
                    if val_acc >= best_val_acc and val_loss <= best_val_loss:
                        # 両方の指標が改善した場合のみモデルを保存
                        self.saver.save(sess, self.checkpt_file)
                        logger.info(f'Model saved at epoch {epoch}')
                        improved = True
                        
                    best_val_acc = max(val_acc, best_val_acc)
                    best_val_loss = min(val_loss, best_val_loss)
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience:
                    logger.info(f'Early stopping at epoch {epoch}')
                    logger.info(f'Best validation: loss={best_val_loss:.5f}, acc={best_val_acc:.5f} at epoch {best_epoch}')
                    break
            
            # 最良のモデルを読み込んでテスト
            logger.info("Loading best model for testing...")
            self.saver.restore(sess, self.checkpt_file)
            
            test_loss, test_acc = self.test(sess)
            
            total_time = time.time() - start_time
            logger.info(f'Test results: loss={test_loss:.5f}, accuracy={test_acc:.5f}')
            logger.info(f'Total training time: {total_time:.2f}s')
            
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'best_val_loss': best_val_loss,
            'best_val_accuracy': best_val_acc,
            'total_epochs': epoch + 1,
            'training_time': total_time
        }


def create_config() -> Dict[str, Any]:
    """デフォルト設定を作成"""
    return {
        'dataset': 'cora',
        'checkpt_file': 'pre_trained/cora/mod_cora.ckpt',
        'batch_size': 1,
        'nb_epochs': 100000,
        'patience': 100,
        'lr': 0.005,
        'l2_coef': 0.0005,
        'hid_units': [8],
        'n_heads': [8, 1],
        'residual': False,
        'nonlinearity': tf.nn.elu,
        'model': SpGAT,
        'sparse': True,
        'attn_drop': 0.6,
        'ffd_drop': 0.6
    }


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='GAT Training')
    parser.add_argument('--dataset', type=str, default='cora', 
                       choices=['cora', 'citeseer', 'pubmed'],
                       help='Dataset to use')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100000, help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping')
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[8], 
                       help='Hidden units per layer')
    parser.add_argument('--heads', type=int, nargs='+', default=[8, 1], 
                       help='Number of attention heads')
    parser.add_argument('--residual', action='store_true', help='Use residual connections')
    
    args = parser.parse_args()
    
    # 設定の作成
    config = create_config()
    config.update({
        'dataset': args.dataset,
        'lr': args.lr,
        'nb_epochs': args.epochs,
        'patience': args.patience,
        'hid_units': args.hidden_units,
        'n_heads': args.heads,
        'residual': args.residual,
        'checkpt_file': f'pre_trained/{args.dataset}/mod_{args.dataset}.ckpt'
    })
    
    # 訓練の実行
    trainer = GATTrainer(config)
    trainer.load_data()
    trainer.build_model()
    
    results = trainer.train()
    
    logger.info("Training completed!")
    logger.info(f"Final results: {results}")


if __name__ == '__main__':
    main()
