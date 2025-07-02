import numpy as np
import json
import networkx as nx
from networkx.readwrite import json_graph
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Set recursion limit for deep graphs
sys.setrecursionlimit(99999)

@dataclass
class P2PConfig:
    """Configuration for P2P dataset processing."""
    dataset_path: str = 'p2p_dataset'
    graph_file: str = 'ppi-G.json'
    id_map_file: str = 'ppi-id_map.json'
    features_file: str = 'ppi-feats.npy'
    class_map_file: str = 'ppi-class_map.json'
    min_subgraph_size: int = 3
    num_classes: int = 121
    val_split_id: int = 21
    test_split_id: int = 23
    train_split_id: int = 1

class P2PDataProcessor:
    """Processor for Protein-Protein Interaction dataset."""
    
    def __init__(self, config: P2PConfig):
        self.config = config
        self.logger = self._setup_logging()
        self._validate_dataset_path()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(self.__class__.__name__)
    
    def _validate_dataset_path(self) -> None:
        """Validate that dataset directory and files exist."""
        dataset_dir = Path(self.config.dataset_path)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        required_files = [
            self.config.graph_file,
            self.config.id_map_file, 
            self.config.features_file,
            self.config.class_map_file
        ]
        
        for file_name in required_files:
            file_path = dataset_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
    
    def _run_dfs(self, adj: sp.csr_matrix, mask: np.ndarray, node: int, 
                 component_id: int) -> None:
        """Run depth-first search to identify connected components."""
        if mask[node] == -1:
            mask[node] = component_id
            # Get neighbors efficiently from sparse matrix
            neighbors = adj[node, :].nonzero()[1]
            for neighbor in neighbors:
                self._run_dfs(adj, mask, neighbor, component_id)
    
    def _dfs_split(self, adj: sp.csr_matrix) -> np.ndarray:
        """Split graph into connected components using DFS."""
        nb_nodes = adj.shape[0]
        component_mask = np.full(nb_nodes, -1, dtype=np.int32)
        component_id = 0
        
        self.logger.info(f"Splitting graph with {nb_nodes} nodes into components...")
        
        for node in range(nb_nodes):
            if component_mask[node] == -1:
                self._run_dfs(adj, component_mask, node, component_id)
                component_id += 1
        
        self.logger.info(f"Found {component_id} connected components")
        return component_mask
    
    def _validate_component_consistency(self, adj: sp.csr_matrix, 
                                      mapping: np.ndarray) -> bool:
        """Validate that all edges are within the same component."""
        nb_nodes = adj.shape[0]
        for i in range(nb_nodes):
            neighbors = adj[i, :].nonzero()[1]
            for j in neighbors:
                if mapping[i] != mapping[j]:
                    return False
        return True
    
    def _find_component_splits(self, adj: sp.csr_matrix, mapping: np.ndarray, 
                             node_labels: List[Dict]) -> Optional[Dict[int, str]]:
        """Determine train/val/test split for each component."""
        nb_nodes = adj.shape[0]
        component_splits = {}
        
        for i in range(nb_nodes):
            neighbors = adj[i, :].nonzero()[1]
            for j in neighbors:
                component_id = mapping[i]
                
                # Skip component 0 (isolated nodes)
                if component_id == 0:
                    component_splits[0] = None
                    continue
                
                # Check if nodes are in same component
                if mapping[i] == mapping[j]:
                    # Verify consistent labels within component
                    if (node_labels[i]['val'] != node_labels[j]['val'] or 
                        node_labels[i]['test'] != node_labels[j]['test']):
                        self.logger.error("Inconsistent labels within component!")
                        return None
                    
                    # Determine split type
                    if component_id not in component_splits:
                        if node_labels[i]['test']:
                            component_splits[component_id] = 'test'
                        elif node_labels[i]['val']:
                            component_splits[component_id] = 'val'
                        else:
                            component_splits[component_id] = 'train'
                    else:
                        # Verify consistency
                        current_split = component_splits[component_id]
                        node_split = ('test' if node_labels[i]['test'] 
                                    else 'val' if node_labels[i]['val'] 
                                    else 'train')
                        
                        if current_split != node_split:
                            self.logger.error("Inconsistent splits within component!")
                            return None
        
        return component_splits
    
    def _load_data(self) -> Tuple[nx.Graph, Dict, np.ndarray, Dict]:
        """Load all required data files."""
        dataset_dir = Path(self.config.dataset_path)
        
        # Load graph
        self.logger.info("Loading graph...")
        with open(dataset_dir / self.config.graph_file) as f:
            graph_data = json.load(f)
        graph = json_graph.node_link_graph(graph_data)
        self.logger.info(f"Loaded graph with {len(graph_data)} elements")
        
        # Load ID mapping
        self.logger.info("Loading ID mapping...")
        with open(dataset_dir / self.config.id_map_file) as f:
            id_map = json.load(f)
        id_map = {int(k): [int(v)] for k, v in id_map.items()}
        self.logger.info(f"Loaded {len(id_map)} ID mappings")
        
        # Load features
        self.logger.info("Loading features...")
        features = np.load(dataset_dir / self.config.features_file)
        self.logger.info(f"Loaded features with shape {features.shape}")
        
        # Load class mapping
        self.logger.info("Loading class mapping...")
        with open(dataset_dir / self.config.class_map_file) as f:
            class_map = json.load(f)
        self.logger.info(f"Loaded {len(class_map)} class mappings")
        
        return graph, graph_data, id_map, features, class_map
    
    def _standardize_features(self, graph: nx.Graph, id_map: Dict, 
                            features: np.ndarray) -> sp.lil_matrix:
        """Standardize features using training set statistics."""
        self.logger.info("Standardizing features...")
        
        # Get training node indices
        train_ids = np.array([
            id_map[n] for n in graph.nodes() 
            if not graph.nodes[n]['val'] and not graph.nodes[n]['test']
        ])
        
        # Fit scaler on training data only
        train_features = features[train_ids[:, 0]]
        scaler = StandardScaler()
        scaler.fit(train_features)
        
        # Transform all features
        standardized_features = scaler.transform(features)
        return sp.csr_matrix(standardized_features).tolil()
    
    def _reassign_component_ids(self, splits: np.ndarray, 
                              node_labels: List[Dict]) -> np.ndarray:
        """Reassign component IDs and handle small components."""
        self.logger.info("Reassigning component IDs...")
        
        split_list = splits.tolist()
        new_group_id = 1
        
        for component_id in range(np.max(split_list) + 1):
            component_size = split_list.count(component_id)
            
            if component_size >= self.config.min_subgraph_size:
                # Large enough component - assign new ID
                splits[np.array(split_list) == component_id] = new_group_id
                new_group_id += 1
            else:
                # Small component - assign based on split type
                component_nodes = np.where(np.array(split_list) == component_id)[0]
                split_type = self._determine_small_component_split(
                    component_nodes, node_labels
                )
                
                split_id_map = {
                    'train': self.config.train_split_id,
                    'val': self.config.val_split_id,
                    'test': self.config.test_split_id
                }
                
                splits[np.array(split_list) == component_id] = split_id_map[split_type]
        
        return splits
    
    def _determine_small_component_split(self, nodes: np.ndarray, 
                                       node_labels: List[Dict]) -> str:
        """Determine split type for small components."""
        split_type = None
        
        for node in nodes:
            if node_labels[node]['test']:
                current_split = 'test'
            elif node_labels[node]['val']:
                current_split = 'val'
            else:
                current_split = 'train'
            
            if split_type is None:
                split_type = current_split
            elif split_type != current_split:
                raise ValueError(
                    f"Inconsistent splits in small component: {split_type} vs {current_split}"
                )
        
        return split_type or 'train'
    
    def _create_subgraphs(self, adj: sp.csr_matrix, features: sp.lil_matrix,
                         class_map: Dict, splits: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Create fixed-size subgraphs from components."""
        self.logger.info("Creating subgraphs...")
        
        split_list = splits.tolist()
        unique_splits = list(range(1, np.max(split_list) + 1))
        
        # Calculate nodes per graph
        nodes_per_graph = []
        for split_id in unique_splits:
            count = split_list.count(split_id)
            if count > 0:
                nodes_per_graph.append(count)
        
        max_nodes = max(nodes_per_graph) if nodes_per_graph else 0
        num_subgraphs = len(nodes_per_graph)
        
        self.logger.info(f"Creating {num_subgraphs} subgraphs with max {max_nodes} nodes")
        
        # Initialize subgraph arrays
        adj_sub = np.zeros((num_subgraphs, max_nodes, max_nodes))
        feat_sub = np.zeros((num_subgraphs, max_nodes, features.shape[1]))
        labels_sub = np.zeros((num_subgraphs, max_nodes, self.config.num_classes))
        
        # Process each subgraph
        for idx, split_id in enumerate(unique_splits):
            if split_list.count(split_id) == 0:
                continue
                
            # Get nodes for this component
            component_nodes = np.where(splits == split_id)[0]
            subgraph_adj = adj[component_nodes, :][:, component_nodes]
            
            # Create padded subgraph
            if len(component_nodes) < max_nodes:
                # Pad with identity matrix
                padded_adj = np.eye(max_nodes)
                padded_adj[:len(component_nodes), :len(component_nodes)] = subgraph_adj.toarray()
                adj_sub[idx] = padded_adj
                
                # Pad features
                feat_sub[idx, :len(component_nodes)] = features[component_nodes].toarray()
                
                # Pad labels  
                for j, node in enumerate(component_nodes):
                    labels_sub[idx, j] = np.array(class_map[str(node)])
            else:
                adj_sub[idx] = subgraph_adj.toarray()
                feat_sub[idx] = features[component_nodes].toarray()
                for j, node in enumerate(component_nodes):
                    labels_sub[idx, j] = np.array(class_map[str(node)])
        
        return adj_sub, feat_sub, labels_sub, np.array(nodes_per_graph)
    
    def _split_data(self, adj_sub: np.ndarray, feat_sub: np.ndarray,
                   labels_sub: np.ndarray, nodes_per_graph: np.ndarray,
                   component_splits: Dict) -> Tuple[np.ndarray, ...]:
        """Split subgraphs into train/val/test sets."""
        self.logger.info("Splitting data into train/val/test...")
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for component_id, split_type in component_splits.items():
            if component_id == 0 or split_type is None:
                continue
            
            # Component IDs are 1-indexed, array indices are 0-indexed
            array_idx = component_id - 1
            
            if split_type == 'train':
                train_indices.append(array_idx)
            elif split_type == 'val':
                val_indices.append(array_idx)
            elif split_type == 'test':
                test_indices.append(array_idx)
        
        # Extract subsets
        train_adj = adj_sub[train_indices] if train_indices else np.array([])
        val_adj = adj_sub[val_indices] if val_indices else np.array([])
        test_adj = adj_sub[test_indices] if test_indices else np.array([])
        
        train_feat = feat_sub[train_indices] if train_indices else np.array([])
        val_feat = feat_sub[val_indices] if val_indices else np.array([])
        test_feat = feat_sub[test_indices] if test_indices else np.array([])
        
        train_labels = labels_sub[train_indices] if train_indices else np.array([])
        val_labels = labels_sub[val_indices] if val_indices else np.array([])
        test_labels = labels_sub[test_indices] if test_indices else np.array([])
        
        train_nodes = nodes_per_graph[train_indices] if train_indices else np.array([])
        val_nodes = nodes_per_graph[val_indices] if val_indices else np.array([])
        test_nodes = nodes_per_graph[test_indices] if test_indices else np.array([])
        
        return (train_adj, val_adj, test_adj, train_feat, val_feat, test_feat,
                train_labels, val_labels, test_labels, train_nodes, val_nodes, test_nodes)
    
    def _create_masks(self, train_nodes: np.ndarray, val_nodes: np.ndarray,
                     test_nodes: np.ndarray, max_nodes: int) -> Tuple[np.ndarray, ...]:
        """Create binary masks for valid nodes in each subgraph."""
        def create_mask(node_counts: np.ndarray) -> np.ndarray:
            if len(node_counts) == 0:
                return np.array([])
            
            mask = np.zeros((len(node_counts), max_nodes))
            for i, count in enumerate(node_counts):
                mask[i, :count] = 1
            return mask
        
        train_mask = create_mask(train_nodes)
        val_mask = create_mask(val_nodes)
        test_mask = create_mask(test_nodes)
        
        return train_mask, val_mask, test_mask
    
    def process(self) -> Tuple[np.ndarray, ...]:
        """Main processing function."""
        try:
            # Load all data
            graph, graph_data, id_map, features, class_map = self._load_data()
            
            # Extract adjacency matrix
            adj = nx.adjacency_matrix(graph)
            
            # Standardize features
            features_std = self._standardize_features(graph, id_map, features)
            
            # Split graph into components
            component_splits = self._dfs_split(adj)
            
            # Reassign component IDs
            component_splits = self._reassign_component_ids(
                component_splits, graph_data['nodes']
            )
            
            # Validate component consistency
            if not self._validate_component_consistency(adj, component_splits):
                raise ValueError("Graph components are not properly isolated!")
            
            self.logger.info("Graph components are properly isolated ✓")
            
            # Find split assignments
            split_mapping = self._find_component_splits(
                adj, component_splits, graph_data['nodes']
            )
            
            if split_mapping is None:
                raise ValueError("Failed to create consistent split mapping")
            
            # Create subgraphs
            adj_sub, feat_sub, labels_sub, nodes_per_graph = self._create_subgraphs(
                adj, features_std, class_map, component_splits
            )
            
            # Split into train/val/test
            split_data = self._split_data(
                adj_sub, feat_sub, labels_sub, nodes_per_graph, split_mapping
            )
            
            # Create masks
            max_nodes = adj_sub.shape[1] if adj_sub.size > 0 else 0
            masks = self._create_masks(
                split_data[9], split_data[10], split_data[11], max_nodes
            )
            
            self.logger.info("P2P dataset processing completed successfully!")
            
            return split_data + masks
            
        except Exception as e:
            self.logger.error(f"Error processing P2P dataset: {e}")
            raise

def process_p2p(config: Optional[P2PConfig] = None) -> Tuple[np.ndarray, ...]:
    """Main function to process P2P dataset."""
    if config is None:
        config = P2PConfig()
    
    processor = P2PDataProcessor(config)
    return processor.process()

# For backward compatibility
def process_p2p_legacy() -> Tuple[np.ndarray, ...]:
    """Legacy function interface."""
    return process_p2p()

if __name__ == "__main__":
    # Example usage
    try:
        results = process_p2p()
        print("Dataset processing completed successfully!")
        print(f"Returned {len(results)} arrays")
    except Exception as e:
        print(f"Error: {e}")