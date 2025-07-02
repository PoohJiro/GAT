import time
import numpy as np
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior() 
import logging
from pathlib import Path

from models import GAT
from utils import process

# Configuration
CONFIG = {
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
    'attn_drop_rate': 0.6,
    'ffd_drop_rate': 0.6,
    'early_stop_threshold': 1e-6,
    'print_interval': 100
}

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def print_hyperparameters(config):
    """Print model and training hyperparameters."""
    logger.info(f'Dataset: {config["dataset"]}')
    logger.info('----- Training hyperparams -----')
    logger.info(f'Learning rate: {config["lr"]}')
    logger.info(f'L2 coefficient: {config["l2_coef"]}')
    logger.info(f'Batch size: {config["batch_size"]}')
    logger.info(f'Max epochs: {config["nb_epochs"]}')
    logger.info(f'Patience: {config["patience"]}')
    logger.info('----- Architecture hyperparams -----')
    logger.info(f'Hidden layers: {len(config["hid_units"])}')
    logger.info(f'Hidden units per layer: {config["hid_units"]}')
    logger.info(f'Attention heads: {config["n_heads"]}')
    logger.info(f'Residual connections: {config["residual"]}')
    logger.info(f'Nonlinearity: {config["nonlinearity"].__name__}')
    logger.info(f'Dropout rates - Attention: {config["attn_drop_rate"]}, FFD: {config["ffd_drop_rate"]}')

def load_and_preprocess_data(dataset_name):
    """Load and preprocess the dataset."""
    logger.info(f'Loading dataset: {dataset_name}')
    
    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset_name)
    features, _ = process.preprocess_features(features)
    
    # Get data dimensions
    nb_nodes = features.shape[0]
    ft_size = features.shape[1] 
    nb_classes = y_train.shape[1]
    
    logger.info(f'Dataset stats - Nodes: {nb_nodes}, Features: {ft_size}, Classes: {nb_classes}')
    
    # Convert to dense and add batch dimension
    adj = adj.todense()
    data_arrays = [features, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask]
    data_arrays = [arr[np.newaxis] for arr in data_arrays]
    
    # Generate bias matrices
    biases = process.adj_to_bias(data_arrays[1], [nb_nodes], nhood=1)
    
    return data_arrays + [biases], (nb_nodes, ft_size, nb_classes)

def create_placeholders(batch_size, nb_nodes, ft_size, nb_classes):
    """Create TensorFlow placeholders."""
    with tf.name_scope('input'):
        placeholders = {
            'ftr_in': tf.placeholder(tf.float32, (batch_size, nb_nodes, ft_size), name='features'),
            'bias_in': tf.placeholder(tf.float32, (batch_size, nb_nodes, nb_nodes), name='bias'),
            'lbl_in': tf.placeholder(tf.int32, (batch_size, nb_nodes, nb_classes), name='labels'),
            'msk_in': tf.placeholder(tf.int32, (batch_size, nb_nodes), name='mask'),
            'attn_drop': tf.placeholder(tf.float32, (), name='attention_dropout'),
            'ffd_drop': tf.placeholder(tf.float32, (), name='ffd_dropout'),
            'is_train': tf.placeholder(tf.bool, (), name='is_training')
        }
    return placeholders

def build_model(placeholders, config, nb_classes, nb_nodes):
    """Build the GAT model."""
    logits = GAT.inference(
        placeholders['ftr_in'], nb_classes, nb_nodes, placeholders['is_train'],
        placeholders['attn_drop'], placeholders['ffd_drop'],
        bias_mat=placeholders['bias_in'],
        hid_units=config['hid_units'], 
        n_heads=config['n_heads'],
        residual=config['residual'], 
        activation=config['nonlinearity']
    )
    
    # Reshape for loss computation
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(placeholders['lbl_in'], [-1, nb_classes])
    msk_resh = tf.reshape(placeholders['msk_in'], [-1])
    
    # Define loss and metrics
    loss = GAT.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = GAT.masked_accuracy(log_resh, lab_resh, msk_resh)
    
    # Training operation
    train_op = GAT.training(loss, config['lr'], config['l2_coef'])
    
    return logits, loss, accuracy, train_op

def run_epoch(sess, data, placeholders, ops, config, is_training=True):
    """Run one epoch of training or validation."""
    features, biases, labels, masks = data
    loss_op, acc_op = ops[:2]
    train_op = ops[2] if is_training else None
    
    total_loss = 0.0
    total_acc = 0.0
    step = 0
    batch_size = config['batch_size']
    data_size = features.shape[0]
    
    while step * batch_size < data_size:
        feed_dict = {
            placeholders['ftr_in']: features[step*batch_size:(step+1)*batch_size],
            placeholders['bias_in']: biases[step*batch_size:(step+1)*batch_size],
            placeholders['lbl_in']: labels[step*batch_size:(step+1)*batch_size],
            placeholders['msk_in']: masks[step*batch_size:(step+1)*batch_size],
            placeholders['is_train']: is_training,
            placeholders['attn_drop']: config['attn_drop_rate'] if is_training else 0.0,
            placeholders['ffd_drop']: config['ffd_drop_rate'] if is_training else 0.0
        }
        
        if is_training:
            _, loss_val, acc_val = sess.run([train_op, loss_op, acc_op], feed_dict=feed_dict)
        else:
            loss_val, acc_val = sess.run([loss_op, acc_op], feed_dict=feed_dict)
            
        total_loss += loss_val
        total_acc += acc_val
        step += 1
    
    return total_loss / step, total_acc / step

def train_model(sess, data_dict, placeholders, ops, config, saver):
    """Main training loop with early stopping."""
    features, _, y_train, y_val, y_test, train_mask, val_mask, test_mask, biases = data_dict
    loss_op, accuracy_op, train_op = ops
    
    # Training data
    train_data = (features, biases, y_train, train_mask)
    val_data = (features, biases, y_val, val_mask)
    
    # Early stopping variables
    best_val_loss = np.inf
    best_val_acc = 0.0
    patience_counter = 0
    
    logger.info('Starting training...')
    start_time = time.time()
    
    for epoch in range(config['nb_epochs']):
        # Training
        train_loss, train_acc = run_epoch(
            sess, train_data, placeholders, (loss_op, accuracy_op, train_op), config, is_training=True
        )
        
        # Validation  
        val_loss, val_acc = run_epoch(
            sess, val_data, placeholders, (loss_op, accuracy_op), config, is_training=False
        )
        
        # Print progress
        if epoch % config['print_interval'] == 0 or epoch < 10:
            logger.info(f'Epoch {epoch:5d}: Train Loss={train_loss:.5f}, Train Acc={train_acc:.5f} | '
                       f'Val Loss={val_loss:.5f}, Val Acc={val_acc:.5f}')
        
        # Early stopping logic
        improved = False
        if val_acc > best_val_acc or val_loss < best_val_loss:
            if val_acc > best_val_acc and val_loss < best_val_loss:
                # Both metrics improved - save model
                saver.save(sess, config['checkpt_file'])
                logger.info(f'Model saved at epoch {epoch} (Val Acc: {val_acc:.5f}, Val Loss: {val_loss:.5f})')
                improved = True
            
            best_val_acc = max(val_acc, best_val_acc)
            best_val_loss = min(val_loss, best_val_loss)
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Check early stopping
        if patience_counter >= config['patience']:
            logger.info(f'Early stopping at epoch {epoch}!')
            logger.info(f'Best validation - Loss: {best_val_loss:.5f}, Accuracy: {best_val_acc:.5f}')
            break
    
    training_time = time.time() - start_time
    logger.info(f'Training completed in {training_time:.2f} seconds')
    
    return best_val_loss, best_val_acc

def evaluate_model(sess, data_dict, placeholders, ops, config):
    """Evaluate the model on test set."""
    features, _, _, _, y_test, _, _, test_mask, biases = data_dict
    loss_op, accuracy_op = ops[:2]
    
    test_data = (features, biases, y_test, test_mask)
    test_loss, test_acc = run_epoch(
        sess, test_data, placeholders, (loss_op, accuracy_op), config, is_training=False
    )
    
    logger.info(f'Test Results - Loss: {test_loss:.5f}, Accuracy: {test_acc:.5f}')
    return test_loss, test_acc

def main():
    """Main training function."""
    global logger
    logger = setup_logging()
    
    # Print configuration
    print_hyperparameters(CONFIG)
    
    # Create checkpoint directory
    Path(CONFIG['checkpt_file']).parent.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    data_dict, (nb_nodes, ft_size, nb_classes) = load_and_preprocess_data(CONFIG['dataset'])
    
    # Build computational graph
    with tf.Graph().as_default():
        # Create placeholders
        placeholders = create_placeholders(CONFIG['batch_size'], nb_nodes, ft_size, nb_classes)
        
        # Build model
        logits, loss, accuracy, train_op = build_model(placeholders, CONFIG, nb_classes, nb_nodes)
        
        # Initialize variables and session
        saver = tf.train.Saver()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        with tf.Session() as sess:
            sess.run(init_op)
            
            # Train model
            best_val_loss, best_val_acc = train_model(
                sess, data_dict, placeholders, (loss, accuracy, train_op), CONFIG, saver
            )
            
            # Restore best model and evaluate
            saver.restore(sess, CONFIG['checkpt_file'])
            test_loss, test_acc = evaluate_model(
                sess, data_dict, placeholders, (loss, accuracy), CONFIG
            )
            
            logger.info('Training completed successfully!')

if __name__ == '__main__':
    main()