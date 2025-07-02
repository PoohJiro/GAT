import tensorflow as tf
from tensorflow import keras
import numpy as np


class BaseGAttN:
    """
    Base class for Graph Attention Networks with loss functions and metrics
    """
    
    @staticmethod
    def weighted_sparse_categorical_crossentropy(y_true, y_pred, class_weights=None):
        """
        Weighted sparse categorical crossentropy loss
        
        Args:
            y_true: True labels [batch_size, ...]
            y_pred: Predicted logits [batch_size, ..., num_classes]
            class_weights: Weight for each class [num_classes]
        
        Returns:
            Weighted crossentropy loss
        """
        if class_weights is not None:
            # Convert to tensor if needed
            class_weights = tf.convert_to_tensor(class_weights, dtype=tf.float32)
            
            # Get weights for each sample based on true labels
            sample_weights = tf.gather(class_weights, y_true)
            
            # Compute crossentropy
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_true, logits=y_pred
            )
            
            # Apply sample weights
            weighted_xentropy = tf.multiply(xentropy, sample_weights)
            return tf.reduce_mean(weighted_xentropy)
        else:
            return tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y_true, logits=y_pred
                )
            )
    
    @staticmethod
    def l2_regularization_loss(model, l2_coef=1e-4):
        """
        Compute L2 regularization loss for model weights
        
        Args:
            model: Keras model or list of variables
            l2_coef: L2 regularization coefficient
        
        Returns:
            L2 regularization loss
        """
        if hasattr(model, 'trainable_variables'):
            variables = model.trainable_variables
        else:
            variables = model
        
        # Exclude bias terms and normalization parameters
        exclude_names = ['bias', 'gamma', 'beta', 'b', 'g']
        
        l2_losses = []
        for var in variables:
            # Check if variable name contains excluded terms
            if not any(name in var.name.lower() for name in exclude_names):
                l2_losses.append(tf.nn.l2_loss(var))
        
        if l2_losses:
            return l2_coef * tf.add_n(l2_losses)
        else:
            return 0.0
    
    @staticmethod
    def create_optimizer(learning_rate=0.001, optimizer_type='adam', **kwargs):
        """
        Create optimizer with specified type and parameters
        
        Args:
            learning_rate: Learning rate
            optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
            **kwargs: Additional optimizer parameters
        
        Returns:
            TensorFlow optimizer
        """
        if optimizer_type.lower() == 'adam':
            return keras.optimizers.Adam(learning_rate=learning_rate, **kwargs)
        elif optimizer_type.lower() == 'sgd':
            return keras.optimizers.SGD(learning_rate=learning_rate, **kwargs)
        elif optimizer_type.lower() == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=learning_rate, **kwargs)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    @staticmethod
    def reshape_for_loss(logits, labels, nb_classes):
        """
        Reshape logits and labels for loss computation
        
        Args:
            logits: Model predictions [batch_size, seq_len, nb_classes]
            labels: True labels [batch_size, seq_len]
            nb_classes: Number of classes
        
        Returns:
            Reshaped logits and labels
        """
        # Flatten all dimensions except the last one for logits
        log_shape = tf.shape(logits)
        log_resh = tf.reshape(logits, [-1, nb_classes])
        
        # Flatten all dimensions for labels
        lab_resh = tf.reshape(labels, [-1])
        
        return log_resh, lab_resh
    
    @staticmethod
    def confusion_matrix(y_true, y_pred, num_classes=None):
        """
        Compute confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted logits or probabilities
            num_classes: Number of classes
        
        Returns:
            Confusion matrix
        """
        if len(y_pred.shape) > 1:
            preds = tf.argmax(y_pred, axis=-1)
        else:
            preds = y_pred
        
        return tf.math.confusion_matrix(
            y_true, preds, num_classes=num_classes, dtype=tf.int32
        )


class MaskedMetrics:
    """
    Metrics with masking support for node classification tasks
    """
    
    @staticmethod
    def masked_softmax_cross_entropy(y_true, y_pred, mask):
        """
        Softmax cross-entropy loss with masking
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted logits
            mask: Mask tensor (1 for valid, 0 for masked)
        
        Returns:
            Masked cross-entropy loss
        """
        # Compute cross-entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        
        # Apply mask
        mask = tf.cast(mask, dtype=tf.float32)
        # Normalize mask to maintain proper scale
        mask_sum = tf.reduce_sum(mask)
        mask_normalized = mask * tf.cast(tf.size(mask), tf.float32) / mask_sum
        
        # Apply normalized mask
        masked_loss = loss * mask_normalized
        return tf.reduce_mean(masked_loss)
    
    @staticmethod
    def masked_sparse_categorical_crossentropy(y_true, y_pred, mask):
        """
        Sparse categorical cross-entropy loss with masking
        
        Args:
            y_true: True labels (sparse)
            y_pred: Predicted logits
            mask: Mask tensor
        
        Returns:
            Masked sparse categorical cross-entropy loss
        """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        
        mask = tf.cast(mask, dtype=tf.float32)
        mask_sum = tf.reduce_sum(mask)
        mask_normalized = mask * tf.cast(tf.size(mask), tf.float32) / mask_sum
        
        masked_loss = loss * mask_normalized
        return tf.reduce_mean(masked_loss)
    
    @staticmethod
    def masked_binary_crossentropy(y_true, y_pred, mask):
        """
        Binary cross-entropy loss with masking (for multi-label)
        
        Args:
            y_true: True labels
            y_pred: Predicted logits
            mask: Mask tensor
        
        Returns:
            Masked binary cross-entropy loss
        """
        y_true = tf.cast(y_true, dtype=tf.float32)
        
        # Compute binary cross-entropy
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        
        # Reduce over label dimensions
        loss = tf.reduce_mean(loss, axis=-1)
        
        # Apply mask
        mask = tf.cast(mask, dtype=tf.float32)
        mask_sum = tf.reduce_sum(mask)
        mask_normalized = mask * tf.cast(tf.size(mask), tf.float32) / mask_sum
        
        masked_loss = loss * mask_normalized
        return tf.reduce_mean(masked_loss)
    
    @staticmethod
    def masked_accuracy(y_true, y_pred, mask):
        """
        Accuracy with masking
        
        Args:
            y_true: True labels (one-hot or sparse)
            y_pred: Predicted logits
            mask: Mask tensor
        
        Returns:
            Masked accuracy
        """
        # Handle both one-hot and sparse labels
        if len(y_true.shape) == len(y_pred.shape):
            # One-hot encoded
            true_classes = tf.argmax(y_true, axis=-1)
        else:
            # Sparse labels
            true_classes = y_true
        
        pred_classes = tf.argmax(y_pred, axis=-1)
        
        # Compute accuracy
        correct = tf.equal(true_classes, pred_classes)
        correct = tf.cast(correct, tf.float32)
        
        # Apply mask
        mask = tf.cast(mask, dtype=tf.float32)
        mask_sum = tf.reduce_sum(mask)
        
        if mask_sum > 0:
            masked_correct = correct * mask
            accuracy = tf.reduce_sum(masked_correct) / mask_sum
        else:
            accuracy = 0.0
        
        return accuracy
    
    @staticmethod
    def masked_f1_score(y_true, y_pred, mask, average='micro'):
        """
        F1 score with masking (for multi-label classification)
        
        Args:
            y_true: True labels
            y_pred: Predicted logits
            mask: Mask tensor
            average: Averaging method ('micro', 'macro')
        
        Returns:
            Masked F1 score
        """
        # Convert predictions to binary
        predicted = tf.nn.sigmoid(y_pred)
        predicted = tf.cast(predicted > 0.5, tf.int32)
        
        # Convert labels to int32
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.cast(mask, tf.int32)
        
        # Expand mask for broadcasting
        if len(mask.shape) < len(predicted.shape):
            mask = tf.expand_dims(mask, -1)
        
        # Calculate TP, TN, FP, FN with masking
        tp = tf.reduce_sum(predicted * y_true * mask)
        tn = tf.reduce_sum((1 - predicted) * (1 - y_true) * mask)
        fp = tf.reduce_sum(predicted * (1 - y_true) * mask)
        fn = tf.reduce_sum((1 - predicted) * y_true * mask)
        
        # Calculate precision, recall, and F1
        precision = tf.cond(
            tp + fp > 0,
            lambda: tf.cast(tp, tf.float32) / tf.cast(tp + fp, tf.float32),
            lambda: 0.0
        )
        
        recall = tf.cond(
            tp + fn > 0,
            lambda: tf.cast(tp, tf.float32) / tf.cast(tp + fn, tf.float32),
            lambda: 0.0
        )
        
        f1 = tf.cond(
            precision + recall > 0,
            lambda: 2.0 * precision * recall / (precision + recall),
            lambda: 0.0
        )
        
        return f1


class GraphMetrics(keras.metrics.Metric):
    """
    Custom Keras metric for graph-based tasks
    """
    
    def __init__(self, metric_fn, name='graph_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.metric_fn = metric_fn
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        metric_value = self.metric_fn(y_true, y_pred, sample_weight)
        self.total.assign_add(metric_value)
        self.count.assign_add(1.0)
    
    def result(self):
        return tf.cond(
            self.count > 0,
            lambda: self.total / self.count,
            lambda: 0.0
        )
    
    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


# Usage examples and helper functions
def create_gat_loss_and_metrics(num_classes, class_weights=None, l2_coef=1e-4):
    """
    Create loss function and metrics for GAT training
    
    Args:
        num_classes: Number of classes
        class_weights: Optional class weights
        l2_coef: L2 regularization coefficient
    
    Returns:
        Dictionary with loss function and metrics
    """
    
    def loss_fn(y_true, y_pred, model=None):
        # Main loss
        main_loss = BaseGAttN.weighted_sparse_categorical_crossentropy(
            y_true, y_pred, class_weights
        )
        
        # L2 regularization
        if model is not None:
            l2_loss = BaseGAttN.l2_regularization_loss(model, l2_coef)
            return main_loss + l2_loss
        
        return main_loss
    
    def masked_loss_fn(y_true, y_pred, mask):
        return MaskedMetrics.masked_sparse_categorical_crossentropy(
            y_true, y_pred, mask
        )
    
    def masked_accuracy_fn(y_true, y_pred, mask):
        return MaskedMetrics.masked_accuracy(y_true, y_pred, mask)
    
    return {
        'loss': loss_fn,
        'masked_loss': masked_loss_fn,
        'masked_accuracy': masked_accuracy_fn,
        'confusion_matrix': BaseGAttN.confusion_matrix,
        'masked_f1': MaskedMetrics.masked_f1_score
    }


# Example usage:
"""
# Create GAT model and training setup
num_classes = 7
class_weights = np.array([1.0, 2.0, 1.5, 1.0, 3.0, 2.5, 1.0])

# Get loss and metrics
loss_metrics = create_gat_loss_and_metrics(num_classes, class_weights)

# Create optimizer
optimizer = BaseGAttN.create_optimizer(learning_rate=0.005, optimizer_type='adam')

# In training loop:
with tf.GradientTape() as tape:
    logits = model(x, training=True)
    loss = loss_metrics['loss'](y_true, logits, model)

gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# For masked evaluation:
mask = tf.ones_like(y_true)  # or actual mask
accuracy = loss_metrics['masked_accuracy'](y_true, logits, mask)
f1 = loss_metrics['masked_f1'](y_true, logits, mask)
"""