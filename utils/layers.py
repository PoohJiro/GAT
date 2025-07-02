import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras import layers

conv1d = tf.compat.v1.layers.conv1d

def attention_head(seq, out_sz, bias_mat, activation, 
                  in_drop=0.0, coef_drop=0.0, residual=False, training=True):
    """
    Graph Attention Network head implementation
    
    Args:
        seq: Input sequence tensor [batch_size, num_nodes, in_features]
        out_sz: Output feature dimension
        bias_mat: Bias matrix for masking (e.g., adjacency matrix)
        activation: Activation function
        in_drop: Input dropout rate
        coef_drop: Attention coefficient dropout rate
        residual: Whether to use residual connection
        training: Training mode flag
    
    Returns:
        Output tensor after attention mechanism
    """
    # Input dropout
    if in_drop > 0.0:
        seq = tf.nn.dropout(seq, rate=in_drop, training=training)
    
    # Linear transformation
    seq_fts = layers.Conv1D(out_sz, 1, use_bias=False)(seq)
    
    # Attention mechanism - compute attention logits
    # Use separate Conv1D layers for better parameter separation
    f_1 = layers.Conv1D(1, 1, name='attention_left')(seq_fts)
    f_2 = layers.Conv1D(1, 1, name='attention_right')(seq_fts)
    
    # Compute attention logits: e_ij = a^T [Wh_i || Wh_j]
    # This is simplified version: e_ij = f_1(h_i) + f_2(h_j)
    logits = f_1 + tf.transpose(f_2, [0, 2, 1])
    
    # Apply LeakyReLU and bias matrix (for masking)
    logits = tf.nn.leaky_relu(logits, alpha=0.2) + bias_mat
    
    # Compute attention coefficients
    coefs = tf.nn.softmax(logits, axis=-1)
    
    # Attention coefficient dropout
    if coef_drop > 0.0:
        coefs = tf.nn.dropout(coefs, rate=coef_drop, training=training)
    
    # Apply additional input dropout to features
    if in_drop > 0.0:
        seq_fts = tf.nn.dropout(seq_fts, rate=in_drop, training=training)
    
    # Compute weighted feature aggregation
    vals = tf.matmul(coefs, seq_fts)
    
    # Add bias
    vals = layers.Dense(out_sz, use_bias=True)(vals)
    
    # Residual connection
    if residual:
        if seq.shape[-1] != out_sz:
            # Project input to match output dimension
            seq_proj = layers.Conv1D(out_sz, 1)(seq)
            vals = vals + seq_proj
        else:
            vals = vals + seq
    
    return activation(vals)


def sparse_attention_head(seq, out_sz, adj_mat, activation, nb_nodes,
                         in_drop=0.0, coef_drop=0.0, residual=False, training=True):
    """
    Sparse Graph Attention Network head for large graphs
    
    Note: This implementation assumes batch_size = 1 due to sparse tensor limitations
    
    Args:
        seq: Input sequence tensor [1, num_nodes, in_features]
        out_sz: Output feature dimension
        adj_mat: Sparse adjacency matrix
        activation: Activation function
        nb_nodes: Number of nodes
        in_drop: Input dropout rate
        coef_drop: Attention coefficient dropout rate
        residual: Whether to use residual connection
        training: Training mode flag
    
    Returns:
        Output tensor after sparse attention mechanism
    """
    # Input dropout
    if in_drop > 0.0:
        seq = tf.nn.dropout(seq, rate=in_drop, training=training)
    
    # Linear transformation
    seq_fts = layers.Conv1D(out_sz, 1, use_bias=False)(seq)
    
    # Attention mechanism for sparse case
    f_1 = layers.Conv1D(1, 1, name='sparse_attention_left')(seq_fts)
    f_2 = layers.Conv1D(1, 1, name='sparse_attention_right')(seq_fts)
    
    # Reshape for sparse operations
    f_1 = tf.reshape(f_1, (nb_nodes, 1))
    f_2 = tf.reshape(f_2, (nb_nodes, 1))
    
    # Apply sparse matrix operations
    f_1_sparse = tf.sparse.SparseTensor(
        indices=adj_mat.indices,
        values=tf.gather(tf.squeeze(f_1), adj_mat.indices[:, 0]),
        dense_shape=adj_mat.dense_shape
    )
    
    f_2_sparse = tf.sparse.SparseTensor(
        indices=adj_mat.indices,
        values=tf.gather(tf.squeeze(f_2), adj_mat.indices[:, 1]),
        dense_shape=adj_mat.dense_shape
    )
    
    # Compute sparse logits
    logits = tf.sparse.add(f_1_sparse, f_2_sparse)
    
    # Apply LeakyReLU
    lrelu_values = tf.nn.leaky_relu(logits.values, alpha=0.2)
    lrelu = tf.SparseTensor(
        indices=logits.indices,
        values=lrelu_values,
        dense_shape=logits.dense_shape
    )
    
    # Sparse softmax
    coefs = tf.sparse.softmax(lrelu)
    
    # Attention coefficient dropout
    if coef_drop > 0.0:
        coefs_values = tf.nn.dropout(coefs.values, rate=coef_drop, training=training)
        coefs = tf.SparseTensor(
            indices=coefs.indices,
            values=coefs_values,
            dense_shape=coefs.dense_shape
        )
    
    # Additional input dropout
    if in_drop > 0.0:
        seq_fts = tf.nn.dropout(seq_fts, rate=in_drop, training=training)
    
    # Sparse matrix multiplication
    coefs_reshaped = tf.sparse.reshape(coefs, [nb_nodes, nb_nodes])
    seq_fts_squeezed = tf.squeeze(seq_fts, axis=0)
    
    vals = tf.sparse.sparse_dense_matmul(coefs_reshaped, seq_fts_squeezed)
    vals = tf.expand_dims(vals, axis=0)
    vals.set_shape([1, nb_nodes, out_sz])
    
    # Add bias
    vals = layers.Dense(out_sz, use_bias=True)(vals)
    
    # Residual connection
    if residual:
        if seq.shape[-1] != out_sz:
            seq_proj = layers.Conv1D(out_sz, 1)(seq)
            vals = vals + seq_proj
        else:
            vals = vals + seq
    
    return activation(vals)


class GraphAttentionLayer(layers.Layer):
    """
    Modern Keras-style Graph Attention Layer
    """
    def __init__(self, out_sz, activation=tf.nn.elu, in_drop=0.0, 
                 coef_drop=0.0, residual=False, **kwargs):
        super().__init__(**kwargs)
        self.out_sz = out_sz
        self.activation = activation
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        
        # Initialize layers
        self.linear_transform = layers.Conv1D(out_sz, 1, use_bias=False)
        self.attention_left = layers.Conv1D(1, 1)
        self.attention_right = layers.Conv1D(1, 1)
        self.bias_layer = layers.Dense(out_sz, use_bias=True)
        self.residual_proj = None
    
    def build(self, input_shape):
        super().build(input_shape)
        if self.residual and input_shape[-1] != self.out_sz:
            self.residual_proj = layers.Conv1D(self.out_sz, 1)
    
    def call(self, inputs, bias_mat=None, training=None):
        seq, bias_mat = inputs if isinstance(inputs, (list, tuple)) else (inputs, bias_mat)
        
        # Input dropout
        if self.in_drop > 0.0:
            seq = tf.nn.dropout(seq, rate=self.in_drop, training=training)
        
        # Linear transformation
        seq_fts = self.linear_transform(seq)
        
        # Attention mechanism
        f_1 = self.attention_left(seq_fts)
        f_2 = self.attention_right(seq_fts)
        
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        
        if bias_mat is not None:
            logits += bias_mat
        
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits, alpha=0.2), axis=-1)
        
        # Attention coefficient dropout
        if self.coef_drop > 0.0:
            coefs = tf.nn.dropout(coefs, rate=self.coef_drop, training=training)
        
        # Additional feature dropout
        if self.in_drop > 0.0:
            seq_fts = tf.nn.dropout(seq_fts, rate=self.in_drop, training=training)
        
        # Weighted aggregation
        vals = tf.matmul(coefs, seq_fts)
        vals = self.bias_layer(vals)
        
        # Residual connection
        if self.residual:
            if self.residual_proj is not None:
                vals = vals + self.residual_proj(seq)
            else:
                vals = vals + seq
        
        return self.activation(vals)


# Usage example:
"""
# For regular GAT
batch_size, num_nodes, in_features = 32, 100, 64
out_features = 32

# Create input
x = tf.random.normal([batch_size, num_nodes, in_features])
bias_mat = tf.zeros([batch_size, num_nodes, num_nodes])  # or adjacency matrix

# Using functional approach
output = attention_head(x, out_features, bias_mat, tf.nn.elu, 
                       in_drop=0.1, coef_drop=0.1, residual=True)

# Using Keras layer
gat_layer = GraphAttentionLayer(out_features, in_drop=0.1, coef_drop=0.1, residual=True)
output = gat_layer([x, bias_mat], training=True)
"""