import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as tf_layers


class GraphAttentionLayer(tf_layers.Layer):
    """
    Graph Attention Layer implementation
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
        self.linear_transform = tf_layers.Conv1D(out_sz, 1, use_bias=False)
        self.attention_left = tf_layers.Conv1D(1, 1)
        self.attention_right = tf_layers.Conv1D(1, 1)
        self.bias_layer = tf_layers.Dense(out_sz, use_bias=True)
        self.residual_proj = None
    
    def build(self, input_shape):
        super().build(input_shape)
        if self.residual and input_shape[-1] != self.out_sz:
            self.residual_proj = tf_layers.Conv1D(self.out_sz, 1)
    
    def call(self, inputs, bias_mat=None, training=None):
        seq = inputs
        
        # Input dropout
        if self.in_drop > 0.0:
            seq = tf.nn.dropout(seq, rate=self.in_drop, training=training)
        
        # Linear transformation
        seq_fts = self.linear_transform(seq)
        
        # Attention mechanism
        f_1 = self.attention_left(seq_fts)  # [batch, nodes, 1]
        f_2 = self.attention_right(seq_fts)  # [batch, nodes, 1]
        
        # Compute attention logits
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])  # [batch, nodes, nodes]
        
        # Apply bias matrix (adjacency matrix) if provided
        if bias_mat is not None:
            logits += bias_mat
        
        # Apply activation and softmax
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits, alpha=0.2), axis=-1)
        
        # Attention coefficient dropout
        if self.coef_drop > 0.0:
            coefs = tf.nn.dropout(coefs, rate=self.coef_drop, training=training)
        
        # Apply input dropout to features
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


class MultiHeadGraphAttention(tf_layers.Layer):
    """
    Multi-head Graph Attention Layer
    """
    def __init__(self, out_sz, n_heads, activation=tf.nn.elu, in_drop=0.0,
                 coef_drop=0.0, residual=False, concat=True, **kwargs):
        super().__init__(**kwargs)
        self.out_sz = out_sz
        self.n_heads = n_heads
        self.activation = activation
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.concat = concat
        
        # Create attention heads
        self.attention_heads = []
        head_out_sz = out_sz // n_heads if concat else out_sz
        
        for i in range(n_heads):
            self.attention_heads.append(
                GraphAttentionLayer(
                    out_sz=head_out_sz,
                    activation=activation,
                    in_drop=in_drop,
                    coef_drop=coef_drop,
                    residual=residual,
                    name=f'attention_head_{i}'
                )
            )
    
    def call(self, inputs, bias_mat=None, training=None):
        # Apply each attention head
        head_outputs = []
        for head in self.attention_heads:
            head_out = head(inputs, bias_mat=bias_mat, training=training)
            head_outputs.append(head_out)
        
        if self.concat:
            # Concatenate head outputs
            return tf.concat(head_outputs, axis=-1)
        else:
            # Average head outputs
            return tf.reduce_mean(tf.stack(head_outputs, axis=0), axis=0)


class GAT(keras.Model):
    """
    Graph Attention Network model
    """
    def __init__(self, nb_classes, nb_nodes, hid_units, n_heads, 
                 activation=tf.nn.elu, in_drop=0.0, attn_drop=0.0, 
                 ffd_drop=0.0, residual=False, **kwargs):
        super().__init__(**kwargs)
        
        self.nb_classes = nb_classes
        self.nb_nodes = nb_nodes
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.activation = activation
        self.in_drop = in_drop
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.residual = residual
        
        # Input dropout
        self.input_dropout = tf_layers.Dropout(rate=in_drop)
        
        # Build GAT layers
        self.gat_layers = []
        
        # Hidden layers
        for i, (hid_unit, n_head) in enumerate(zip(hid_units, n_heads[:-1])):
            layer = MultiHeadGraphAttention(
                out_sz=hid_unit,
                n_heads=n_head,
                activation=activation,
                in_drop=ffd_drop,
                coef_drop=attn_drop,
                residual=residual if i > 0 else False,  # No residual for first layer
                concat=True,
                name=f'gat_layer_{i}'
            )
            self.gat_layers.append(layer)
        
        # Output layer (final attention heads)
        self.output_heads = []
        for i in range(n_heads[-1]):
            head = GraphAttentionLayer(
                out_sz=nb_classes,
                activation=lambda x: x,  # No activation for output
                in_drop=ffd_drop,
                coef_drop=attn_drop,
                residual=False,
                name=f'output_head_{i}'
            )
            self.output_heads.append(head)
    
    def call(self, inputs, bias_mat=None, training=None):
        """
        Forward pass of GAT model
        
        Args:
            inputs: Input node features [batch_size, nb_nodes, feature_dim]
            bias_mat: Bias matrix (adjacency matrix) [batch_size, nb_nodes, nb_nodes]
            training: Training mode flag
        
        Returns:
            logits: Output logits [batch_size, nb_nodes, nb_classes]
        """
        x = inputs
        
        # Apply input dropout
        if self.in_drop > 0.0:
            x = self.input_dropout(x, training=training)
        
        # Forward through GAT layers
        for gat_layer in self.gat_layers:
            x = gat_layer(x, bias_mat=bias_mat, training=training)
        
        # Output layer: apply multiple heads and average
        head_outputs = []
        for head in self.output_heads:
            head_out = head(x, bias_mat=bias_mat, training=training)
            head_outputs.append(head_out)
        
        # Average the outputs from multiple heads
        if len(head_outputs) > 1:
            logits = tf.reduce_mean(tf.stack(head_outputs, axis=0), axis=0)
        else:
            logits = head_outputs[0]
        
        return logits
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'nb_classes': self.nb_classes,
            'nb_nodes': self.nb_nodes,
            'hid_units': self.hid_units,
            'n_heads': self.n_heads,
            'activation': keras.activations.serialize(self.activation),
            'in_drop': self.in_drop,
            'attn_drop': self.attn_drop,
            'ffd_drop': self.ffd_drop,
            'residual': self.residual
        })
        return config


# Functional API version for more flexibility
def create_gat_model(nb_classes, nb_nodes, hid_units, n_heads,
                    activation=tf.nn.elu, in_drop=0.0, attn_drop=0.0,
                    ffd_drop=0.0, residual=False, input_dim=None):
    """
    Create GAT model using Functional API
    
    Args:
        nb_classes: Number of output classes
        nb_nodes: Number of nodes in the graph
        hid_units: List of hidden unit sizes
        n_heads: List of number of attention heads for each layer
        activation: Activation function
        in_drop: Input dropout rate
        attn_drop: Attention dropout rate
        ffd_drop: Feature dropout rate
        residual: Whether to use residual connections
        input_dim: Input feature dimension
    
    Returns:
        Keras Model
    """
    # Input layers
    node_features = keras.Input(shape=(nb_nodes, input_dim), name='node_features')
    bias_matrix = keras.Input(shape=(nb_nodes, nb_nodes), name='bias_matrix')
    
    x = node_features
    
    # Input dropout
    if in_drop > 0.0:
        x = tf_layers.Dropout(in_drop)(x)
    
    # Hidden layers
    for i, (hid_unit, n_head) in enumerate(zip(hid_units, n_heads[:-1])):
        multi_head_layer = MultiHeadGraphAttention(
            out_sz=hid_unit,
            n_heads=n_head,
            activation=activation,
            in_drop=ffd_drop,
            coef_drop=attn_drop,
            residual=residual if i > 0 else False,
            concat=True,
            name=f'multi_head_gat_{i}'
        )
        x = multi_head_layer(x, bias_mat=bias_matrix)
    
    # Output layer
    output_heads = []
    for i in range(n_heads[-1]):
        head = GraphAttentionLayer(
            out_sz=nb_classes,
            activation=lambda x: x,
            in_drop=ffd_drop,
            coef_drop=attn_drop,
            residual=False,
            name=f'output_head_{i}'
        )
        head_out = head(x, bias_mat=bias_matrix)
        output_heads.append(head_out)
    
    # Average output heads
    if len(output_heads) > 1:
        logits = tf_layers.Average()(output_heads)
    else:
        logits = output_heads[0]
    
    model = keras.Model(
        inputs=[node_features, bias_matrix], 
        outputs=logits, 
        name='GAT'
    )
    
    return model


# Legacy-compatible inference function
def gat_inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                 bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
    """
    Legacy-compatible GAT inference function
    
    This function maintains compatibility with the original interface
    while using the improved implementation under the hood.
    """
    # Create GAT model
    model = GAT(
        nb_classes=nb_classes,
        nb_nodes=nb_nodes,
        hid_units=hid_units,
        n_heads=n_heads,
        activation=activation,
        attn_drop=attn_drop,
        ffd_drop=ffd_drop,
        residual=residual
    )
    
    # Run inference
    logits = model(inputs, bias_mat=bias_mat, training=training)
    
    return logits


# Usage examples
"""
# Example 1: Using the GAT class
nb_classes = 7
nb_nodes = 2708
hid_units = [8, 8]
n_heads = [8, 8, 1]  # 8 heads for each hidden layer, 1 head for output

model = GAT(
    nb_classes=nb_classes,
    nb_nodes=nb_nodes,
    hid_units=hid_units,
    n_heads=n_heads,
    activation=tf.nn.elu,
    in_drop=0.6,
    attn_drop=0.6,
    ffd_drop=0.6,
    residual=True
)

# Example 2: Using functional API
input_dim = 1433
functional_model = create_gat_model(
    nb_classes=nb_classes,
    nb_nodes=nb_nodes,
    hid_units=hid_units,
    n_heads=n_heads,
    input_dim=input_dim,
    in_drop=0.6,
    attn_drop=0.6,
    ffd_drop=0.6,
    residual=True
)

# Example 3: Training setup
optimizer = keras.optimizers.Adam(learning_rate=0.005, weight_decay=5e-4)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=['accuracy']
)

# Training
# model.fit([X, adj_matrix], y, epochs=200, validation_split=0.2)
"""