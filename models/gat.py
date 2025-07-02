import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as tf_layers
from tensorflow.keras.regularizers import l2

# --- 1. GraphAttentionLayer の修正 ---
class GraphAttentionLayer(tf_layers.Layer):
    """
    Graph Attention Layer (修正版)
    - DropoutをKerasレイヤーに変更
    - バイアス処理を簡潔化
    - get_configを追加し、モデルの保存/読み込みに対応
    """
    def __init__(self, out_sz, activation=tf.nn.elu, in_drop=0.0,
                 coef_drop=0.0, residual=False, l2_reg=0.0, **kwargs):
        super().__init__(**kwargs)
        self.out_sz = out_sz
        self.activation = activation
        self.residual = residual

        # Dropoutレイヤーを初期化
        self.in_dropout = tf_layers.Dropout(in_drop)
        self.coef_dropout = tf_layers.Dropout(coef_drop)

        # 線形変換レイヤー (バイアスをここで追加)
        self.linear_transform = tf_layers.Conv1D(out_sz, 1, use_bias=False, kernel_regularizer=l2(l2_reg))
        
        # アテンション計算用のレイヤー
        self.attention_left = tf_layers.Conv1D(1, 1, kernel_regularizer=l2(l2_reg))
        self.attention_right = tf_layers.Conv1D(1, 1, kernel_regularizer=l2(l2_reg))
        
        # バイアス項
        self.bias = self.add_weight(shape=(out_sz,),
                                    initializer='zeros',
                                    trainable=True,
                                    name='bias')

        # Residual接続用の射影レイヤー (入力と出力の次元が違う場合)
        self.residual_proj = None

    def build(self, input_shape):
        if self.residual and input_shape[-1] != self.out_sz:
            self.residual_proj = tf_layers.Conv1D(self.out_sz, 1)
        super().build(input_shape)

    def call(self, inputs, bias_mat=None):
        seq = inputs

        # 1. 入力の特徴量にDropoutを適用
        seq = self.in_dropout(seq)

        # 2. 線形変換
        seq_fts = self.linear_transform(seq)

        # 3. アテンション係数の計算
        f_1 = self.attention_left(seq_fts)
        f_2 = self.attention_right(seq_fts)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        
        # 隣接行列の情報を加味
        if bias_mat is not None:
            # bias_matは0と-infで構成されていると仮定
            logits += bias_mat

        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits, alpha=0.2))

        # 4. アテンション係数にDropoutを適用
        coefs = self.coef_dropout(coefs)
        
        # 5. 特徴量に再度Dropoutを適用 (論文に忠実な実装)
        seq_fts = self.in_dropout(seq_fts)

        # 6. アテンション係数を用いて特徴量を集約
        vals = tf.matmul(coefs, seq_fts)
        vals += self.bias

        # 7. Residual接続
        if self.residual:
            if self.residual_proj:
                vals += self.residual_proj(seq)
            else:
                vals += seq

        return self.activation(vals)

    def get_config(self):
        config = super().get_config()
        config.update({
            'out_sz': self.out_sz,
            'activation': keras.activations.serialize(self.activation),
            'in_drop': self.in_dropout.rate,
            'coef_drop': self.coef_dropout.rate,
            'residual': self.residual,
            'l2_reg': self.linear_transform.kernel_regularizer.l2 if self.linear_transform.kernel_regularizer else 0.0
        })
        return config


# --- 2. MultiHeadGraphAttention の修正 ---
class MultiHeadGraphAttention(tf_layers.Layer):
    """
    Multi-head Graph Attention Layer (修正版)
    - get_configを追加し、モデルの保存/読み込みに対応
    """
    def __init__(self, out_sz, n_heads, activation=tf.nn.elu, in_drop=0.0,
                 coef_drop=0.0, residual=False, concat=True, l2_reg=0.0, **kwargs):
        super().__init__(**kwargs)
        self.out_sz = out_sz
        self.n_heads = n_heads
        self.concat = concat

        if concat:
            # 連結する場合、各ヘッドの出力次元を調整
            assert out_sz % n_heads == 0
            self.head_out_sz = out_sz // n_heads
        else:
            # 平均化する場合、各ヘッドの出力次元は同じ
            self.head_out_sz = out_sz

        self.attention_heads = [
            GraphAttentionLayer(
                out_sz=self.head_out_sz,
                activation=activation,
                in_drop=in_drop,
                coef_drop=coef_drop,
                residual=residual,
                l2_reg=l2_reg,
                name=f'attention_head_{i}'
            ) for i in range(n_heads)
        ]

    def call(self, inputs, bias_mat=None):
        head_outputs = [head(inputs, bias_mat=bias_mat) for head in self.attention_heads]

        if self.concat:
            return tf.concat(head_outputs, axis=-1)
        else:
            return tf.reduce_mean(tf.stack(head_outputs, axis=-1), axis=-1)

    def get_config(self):
        # 最初のヘッドから設定を取得（全ヘッドで共通のため）
        head_config = self.attention_heads[0].get_config()
        config = super().get_config()
        config.update({
            'out_sz': self.out_sz,
            'n_heads': self.n_heads,
            'activation': head_config['activation'],
            'in_drop': head_config['in_drop'],
            'coef_drop': head_config['coef_drop'],
            'residual': head_config['residual'],
            'concat': self.concat,
            'l2_reg': head_config['l2_reg']
        })
        return config


# --- 3. GATモデル (Functional API版) ---
def create_gat_model(nb_classes, nb_nodes, input_dim, hid_units, n_heads,
                     activation=tf.nn.elu, in_drop=0.0, attn_drop=0.0,
                     residual=False, l2_reg=0.0):
    """
    GATモデルをFunctional APIで構築 (修正・簡略化版)
    - hid_units: 隠れ層のユニット数のリスト (例: [8, 8])
    - n_heads: 各隠れ層と出力層のヘッド数のリスト (例: [8, 8, 1])
    """
    node_features = keras.Input(shape=(nb_nodes, input_dim), name='node_features')
    bias_matrix = keras.Input(shape=(nb_nodes, nb_nodes), name='bias_matrix')

    x = node_features

    # 入力Dropout
    if in_drop > 0.0:
        x = tf_layers.Dropout(in_drop)(x)

    # 隠れ層
    for i, (units, heads) in enumerate(zip(hid_units, n_heads[:-1])):
        x = MultiHeadGraphAttention(
            out_sz=units,
            n_heads=heads,
            activation=activation,
            in_drop=in_drop,
            coef_drop=attn_drop,
            residual=residual,
            concat=True,
            l2_reg=l2_reg,
            name=f'hidden_gat_layer_{i}'
        )(x, bias_mat=bias_matrix)

    # 出力層 (MultiHeadGraphAttentionで平均化)
    output_heads = n_heads[-1]
    logits = MultiHeadGraphAttention(
        out_sz=nb_classes,
        n_heads=output_heads,
        activation=lambda x: x,  # 出力層なので活性化関数はなし
        in_drop=in_drop,
        coef_drop=attn_drop,
        residual=False,
        concat=False, # 最後の層は平均化
        l2_reg=l2_reg,
        name='output_gat_layer'
    )(x, bias_mat=bias_matrix)

    model = keras.Model(inputs=[node_features, bias_matrix], outputs=logits, name='GAT')
    return model

# --- 4. 非推奨の関数の削除 ---

# Coraデータセットを想定したパラメータ例
NB_CLASSES = 7
NB_NODES = 2708
INPUT_DIM = 1433

# モデルのハイパーパラメータ
HID_UNITS = [8]  # 隠れ層のユニット数
N_HEADS = [8, 1] # 隠れ層(8ヘッド)、出力層(1ヘッド)

# L2正則化とDropoutのレート
L2_REG = 5e-4
IN_DROP = 0.6
ATTN_DROP = 0.6

# モデルの作成
model = create_gat_model(
    nb_classes=NB_CLASSES,
    nb_nodes=NB_NODES,
    input_dim=INPUT_DIM,
    hid_units=HID_UNITS,
    n_heads=N_HEADS,
    residual=False, # Coraの実験では通常False
    in_drop=IN_DROP,
    attn_drop=ATTN_DROP,
    l2_reg=L2_REG
)

model.summary()

# モデルのコンパイル
optimizer = keras.optimizers.Adam(learning_rate=0.005)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=['accuracy']
)

# ダミーデータで実行を確認
# batch_size = 1
# dummy_features = np.random.rand(1, NB_NODES, INPUT_DIM).astype(np.float32)
# dummy_adj = np.random.randint(0, 2, size=(1, NB_NODES, NB_NODES)).astype(np.float32)
# # GATでは、隣接がない部分のアテンションを無視するために-infに近い大きな負の値を入れる
# dummy_bias_mat = -1e9 * (1.0 - dummy_adj)

# predictions = model([dummy_features, dummy_bias_mat])
# print("Output shape:", predictions.shape) # (1, 2708, 7)

# トレーニング (実際のデータで実行する場合)
# history = model.fit(
#     [X_train, bias_mat_train], y_train,
#     epochs=100,
#     validation_data=([X_val, bias_mat_val], y_val)
# )