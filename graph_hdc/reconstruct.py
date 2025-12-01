import copy
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from queue import PriorityQueue

import torch
import numpy as np
from rich.pretty import pprint

from graph_hdc.models import HyperNet


def cosine_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Calculate the cosine distance between two tensors.

    :param a: First tensor.
    :param b: Second tensor.
    :param eps: Small epsilon value to prevent division by zero.
    :return: Cosine distance (1 - cosine similarity).
    """
    # Normalize vectors
    a_norm = a / (a.norm() + eps)
    b_norm = b / (b.norm() + eps)

    # Compute cosine similarity
    cosine_sim = torch.dot(a_norm, b_norm).item()

    # Return cosine distance
    return 1.0 - cosine_sim


def dot_product_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Calculate the dot product distance between two tensors.
    
    :param a: First tensor.
    :param b: Second tensor.
    :return: Dot product distance (1 - dot product).
    """
    return -torch.dot(a, b).item()


def manhattan_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Calculate the Manhattan distance between two tensors.

    :param a: First tensor.
    :param b: Second tensor.
    :return: Manhattan distance.
    """
    return torch.sum(torch.abs(a - b)).item()


@dataclass(order=True)
class SearchNode:
    """
    Represents a node in the A* search tree for graph reconstruction.

    The node contains the current graph state, remaining alphabet of nodes to add,
    remaining edges from decode_order_one, similarity score to target embedding,
    and references to parent/children nodes.

    Ordering is based on distance (lower is better) for priority queue usage.
    """
    similarity: float  # Distance value for min heap (lower is better)
    graph: dict = field(compare=False)
    remaining_alphabet: list = field(default_factory=list, compare=False)
    remaining_edges: list = field(default_factory=list, compare=False)
    parent: Optional['SearchNode'] = field(default=None, compare=False)
    embedding: Optional[torch.Tensor] = field(default=None, compare=False)
    children: List['SearchNode'] = field(default_factory=list, compare=False)
    expanded: bool = field(default=False, compare=False)
    depth: int = field(default=0, compare=False)


class GraphReconstructor:

    def __init__(self,
                 encoder: HyperNet,
                 population_size: int = 10,
                 distance_func: callable = cosine_distance,
                 encoder_sim: Optional[HyperNet] = None,
                 #distance_func: callable = manhattan_distance,
                 ):

        self.encoder = encoder
        self.population_size = population_size
        self.distance_func = distance_func
        self.encoder_sim = encoder_sim if encoder_sim is not None else encoder
    
    def reconstruct(self,
                    embedding: torch.Tensor,
                    ):
        
        ## --- getting the node alphabet ---
        # As the first step we can use the level zero node decoding of the graph encoder object 
        # to obtain the list of all of the nodes that are present in the given encoding. 
        # We will then use this node list as the alphabet for the reconstructive generation of 
        # the graph.
        
        node_constraints: list = self.encoder.decode_order_zero(embedding=embedding)
        num_nodes = sum([constraint['num'] for constraint in node_constraints])
        
        node_alphabet: list = [
            constraint['src'] 
            for constraint in node_constraints
            for _ in range(constraint['num'])
        ]
        random.shuffle(node_alphabet)
        
        ## --- setting up initial population ---
        
        population: list[dict] = []
        for c, node in enumerate(node_alphabet):
            
            alphabet = copy.deepcopy(node_alphabet)
            alphabet.remove(node)
            
            info: dict = {
                'alphabet': alphabet,
                'graph': {
                    'node_indices': [0],
                    'node_details': [node],
                    'node_attributes': [0],
                    'edge_indices': [],
                    'edge_attributes': [0],
                }
            }
            population.append(info)
            
            if c >= self.population_size:
                break
            
        ## --- graph generation ---
        
        blacklist_embeddings: set[torch.Tensor] = set()
        
        for info in population:
            
            # Here we will store the information about the previous graph in order to be able 
            # to go back to it if the current branch ends up in a degenerate population.
            info_prev: dict = copy.deepcopy(info)
            
            # We will iterate until we have enough nodes in the graph to match the number 
            # of nodes that are present in the embedding according to the zero order decode.
            
            while (len(info['graph']['node_details']) < num_nodes):
            
                # We will use the alphabet and for each node in the alphabet we will 
                # add it to the current best guess of the graph and then compute the 
                # encoded embedding of that best-guess graph. Finally, we select the 
                # next best guess based on the distance to the target embedding.
                graph = info['graph']
                
                graph['node_adjacency'] = np.zeros(
                    (len(graph['node_indices']), len(graph['node_indices'])),
                    dtype=np.float32
                )
                for i, j in graph['edge_indices']:
                    graph['node_adjacency'][i, j] = 1.0
                    graph['node_adjacency'][j, i] = 1.0
                
                # Here we save the current graph as the previous graph of the next iteration
                # so we have the option to go back to that if that is needed because the 
                # current branch ends up in a degenerate population.
                #info_prev = copy.deepcopy(info)
                
                # We will store all the possible one-hop neighbor graphs that can be 
                # generated from the current graph in this one.
                neighbor_graphs: list[dict] = []
                
                # --- nodes ---
                for node in info['alphabet']:
                    
                    # This method will create all possible ways of inserting that node into 
                    # the current graph. It will return a list of new graph dictionary 
                    # objects.
                    neighbor_graphs += self.graph_add_node(
                        graph=graph,
                        node=node,
                    )
                    
                # --- edges ---
                for i in info['graph']['node_indices']:
                    for j in info['graph']['node_indices']:
                        
                        # If it is possible to add an edge between the nodes i and j, 
                        # this method will return a list of new graph dictionaries with 
                        # that edge added - otherwise returns an empty list.
                        neighbor_graphs += self.graph_add_edge(
                            graph=graph,
                            i=i,
                            j=j,
                        )
                    
                del graph['node_adjacency']
                for g in neighbor_graphs:
                    if 'node_adjacency' in g:
                        del g['node_adjacency']
                    
                neighbor_graphs = [self.expand_graph_details(g) for g in neighbor_graphs]
                    
                # --- forward pass ---
                # At this point, neighbor_graphs contains a list of all the possible 
                # one-hop extensions based on the current graph and the alphabet of 
                # remaining nodes. Now we need to compute the embeddings for all of 
                # these graphs and select the one that is closest to the target 
                # embedding.
                _results: list[dict] = self.encoder.forward_graphs(neighbor_graphs)
                results: list[dict] = []
                
                for result in _results:

                    result_embedding = torch.tensor(result['graph_embedding'])

                    in_blacklist = any(
                        torch.allclose(result_embedding, emb, atol=1e-1)
                        for emb in blacklist_embeddings
                    )
                    if in_blacklist:
                        print('Skipping due to blacklist')
                        continue

                    # Extract distance embeddings for comparison using encoder_sim
                    target_dist_emb = self.encoder_sim.extract_distance_embedding(embedding)
                    candidate_dist_emb = self.encoder_sim.extract_distance_embedding(result_embedding)
                    result['distance'] = self.encoder_sim.get_distance(
                        target_dist_emb, candidate_dist_emb
                    )
                    results.append(result)
                    
                # If there are no results we have reached a degenerate leaf in the 
                # tree of possible graphs where we will go back to the previous 
                # graph and try it again.
                if len(results) == 0:
                                       
                    graph = self.expand_graph_details(graph)
                    current_embedding = torch.tensor(self.encoder.forward_graphs([graph])[0]['graph_hv_stack'])
                    blacklist_embeddings.add(current_embedding)
                    info = copy.deepcopy(info_prev)
                    
                    print('no results, going back to original graph')
                    pprint(info_prev)

                    continue
                
                graph_best, result_best = list(sorted(
                    zip(neighbor_graphs, results),
                    key=lambda x: x[1]['distance']
                ))[0]
                
                alphabet_best = copy.deepcopy(info['alphabet'])
                
                if '_node' in graph_best:
                    node = graph_best['_node']
                    del graph_best['_node']
                    alphabet_best.remove(node)
                
                # Update the current best guess graph with the best result.
                info['graph'] = graph_best
                info['alphabet'] = alphabet_best
                print('alphabet length', len(info['alphabet']), 'blacklist size', len(blacklist_embeddings))
                
        ## --- selecting the best graph ---
        
        # Now we have a population of graphs that are all valid according to the 
        # zero order constraints. We will select the best one based on the distance 
        # to the target embedding.
        results: list[dict] = self.encoder.forward_graphs(
            [info['graph'] for info in population]
        )

        for result in results:
            # Extract distance embeddings for comparison using encoder_sim
            target_dist_emb = self.encoder_sim.extract_distance_embedding(embedding)
            result_embedding = torch.tensor(result['graph_embedding'])
            candidate_dist_emb = self.encoder_sim.extract_distance_embedding(result_embedding)
            result['distance'] = self.encoder_sim.get_distance(
                target_dist_emb, candidate_dist_emb
            )
            
        # Sort the population by the distance to the target embedding.
        population_best, result_best = list(sorted(
            zip(population, results),
            key=lambda x: x[1]['distance']
        ))[0]
                
        # We can return the best graph and its embedding.
        return {
            'graph': population_best['graph'],
            'embedding': result_best['graph_embedding'],
            'distance': result_best['distance'],
        }

    def graph_add_node(self, graph: dict, node: dict) -> list[dict]:
        
        next_index = len(graph['node_indices'])
        
        modified_graphs = []
        for index in graph['node_indices']:
            
            node_current = graph['node_details'][index]
            if 'node_degrees' in node_current:
                degree_existing = sum([int(index in edge) for edge in graph['edge_indices']])
                degree_expected = node_current['node_degrees']
                if degree_existing >= degree_expected:
                    continue
                
            modified_graph: dict = copy.deepcopy(graph)
            # insert the new node at the end of the node list
            modified_graph['node_indices'].append(next_index)
            modified_graph['node_details'].append(node)
            modified_graph['node_attributes'].append(0)

            # insert the edge connection (bidirectional for consistency)
            modified_graph['edge_indices'].append((index, next_index))
            modified_graph['edge_attributes'].append(0)
            modified_graph['edge_indices'].append((next_index, index))
            modified_graph['edge_attributes'].append(0)
            
            # expand the graph details if necessary
            modified_graph = self.expand_graph_details(modified_graph)
            modified_graph['_node'] = node
            
            modified_graphs.append(modified_graph)
            
        return modified_graphs
    
    def graph_add_edge(self, graph: dict, i: int, j: int) -> list[dict]:
        
        if i == j:
            return []
        
        if float(graph['node_adjacency'][i, j]) > 0.5:
            return []
        
        num_edges_i = sum(graph['node_adjacency'][i])
        num_edges_j = sum(graph['node_adjacency'][j])
        # num_edges_i = len([edge for edge in graph['edge_indices'] if i in edge])
        # num_edges_j = len([edge for edge in graph['edge_indices'] if j in edge])
        
        degree_i = graph['node_details'][i]['node_degrees']
        degree_j = graph['node_details'][j]['node_degrees']
        
        if num_edges_i >= degree_i or num_edges_j >= degree_j:
            #print(num_edges_i, degree_i, num_edges_j, degree_j)
            return []
        
        modified_graph: dict = copy.deepcopy(graph)

        # insert the edge connection (bidirectional for consistency)
        modified_graph['edge_indices'].append((i, j))
        modified_graph['edge_attributes'].append(0)
        modified_graph['edge_indices'].append((j, i))
        modified_graph['edge_attributes'].append(0)

        # expand the graph details if necessary
        modified_graph = self.expand_graph_details(modified_graph)

        return [modified_graph]
    
    def expand_graph_details(self, graph: dict) -> dict:
        
        detail_keys: list[str] = graph['node_details'][0].keys()
        for key in detail_keys:
            graph[key] = np.array([
                graph['node_details'][index][key]
                for index in graph['node_indices']
            ])

        return graph


class GraphReconstructorAStar:
    """
    Graph reconstruction using A* search algorithm with edge-only expansion strategy.

    This reconstructor builds graphs incrementally using A* search, exploring the most
    promising candidates first based on their similarity to the target embedding. It uses
    an edge-only expansion strategy where every graph modification must consume an edge
    from the decoded edge constraints, ensuring perfect edge accounting.

    Key Features:
    - Edge-only expansions: All graph modifications are driven by edge constraints
    - Two expansion types: (1) edge between existing nodes, (2) new node with edge
    - Perfect accounting: Each expansion consumes exactly one edge from constraints
    - Memory and time budgets: Bounded resource usage with fallback strategies
    """

    def __init__(self,
                 encoder: HyperNet,
                 memory_budget: int = 1000,
                 time_budget: float = 60.0,
                 distance_func: callable = cosine_distance,
                 batch_size: int = 100,
                 validate_constraints: bool = True,
                 epsilon: float = 1e-4,
                 decode_iterative: bool = True,
                 decode_max_iters: int = 50,
                 decode_threshold: float = 0.5,
                 decode_use_break: bool = True,
                 encoder_sim: Optional[HyperNet] = None,
                 ):
        """
        Initialize the A* graph reconstructor.

        :param encoder: HyperNet encoder instance for computing embeddings
        :param memory_budget: Maximum number of nodes to keep in search tree
        :param time_budget: Maximum reconstruction time in seconds
        :param distance_func: Distance metric for comparing embeddings
        :param batch_size: Batch size for efficient graph encoding
        :param validate_constraints: Whether to enforce structural constraints
        :param epsilon: Distance threshold for early termination (default: 1e-4)
        :param decode_iterative: Use iterative explain-away decoding for edges (default: False)
        :param decode_max_iters: Maximum iterations for iterative decoding (default: 50)
        :param decode_threshold: Score threshold for iterative decoding (default: 0.5)
        :param decode_use_break: Break after first match per iteration (default: True)
        :param encoder_sim: Optional HyperNet encoder for similarity calculations. If not provided,
            uses the main encoder. This allows using a different encoder with potentially different
            extract_distance_embedding() logic or distance function during reconstruction.
        """
        self.encoder = encoder
        self.memory_budget = memory_budget
        self.time_budget = time_budget
        self.distance_func = distance_func
        self.batch_size = batch_size
        self.validate_constraints = validate_constraints
        self.epsilon = epsilon

        # Iterative decoding parameters
        self.decode_iterative = decode_iterative
        self.decode_max_iters = decode_max_iters
        self.decode_threshold = decode_threshold
        self.decode_use_break = decode_use_break

        # Encoder for similarity calculations
        # If not provided, use the main encoder for both encoding and similarity
        self.encoder_sim = encoder_sim if encoder_sim is not None else encoder

        # Track search tree size
        self.tree_size = 0
        self.nodes_expanded = 0

    def reconstruct(self, embedding: torch.Tensor) -> Dict[str, Any]:
        """
        Reconstruct a graph from its hyperdimensional embedding using A* search.

        :param embedding: Target graph embedding to reconstruct
        :return: Dictionary with 'graph', 'embedding', and 'distance' keys
        """
        start_time = time.time()

        # Extract node constraints from embedding
        node_constraints = self.encoder.decode_order_zero(embedding=embedding)
        if not node_constraints:
            # Return empty graph if no nodes detected
            return {
                'graph': {
                    'node_indices': [],
                    'node_details': [],
                    'edge_indices': [],
                },
                'embedding': torch.zeros_like(embedding),
                'distance': float('inf'),
            }

        # Build node alphabet
        node_alphabet = self._build_node_alphabet(node_constraints)
        total_nodes = len(node_alphabet)
        print(f"Total nodes to reconstruct: {total_nodes}")

        # Extract edge constraints from embedding using decode_order_one
        # Check if encoder is an ensemble by checking for the num_models attribute
        from graph_hdc.models.ensemble import HyperNetEnsemble
        is_ensemble = isinstance(self.encoder, HyperNetEnsemble)

        # For ensembles, the embedding is already stacked (num_models, batch_size, dim) or (num_models, dim)
        # For single models with graph_hv_stack, it's (num_layers, dim) and we need first layer
        if is_ensemble:
            # Ensemble: pass the full stacked embedding
            embedding_for_edges = embedding
        else:
            # Single model: if 2D (graph_hv_stack), use the first layer for edge decoding
            embedding_for_edges = embedding[0] if embedding.dim() == 2 else embedding

        edge_constraints = self.encoder.decode_order_one(
            embedding=embedding_for_edges,
            constraints_order_zero=node_constraints,
            use_iterative=self.decode_iterative,
            max_iters=self.decode_max_iters,
            threshold=self.decode_threshold,
            use_break=self.decode_use_break,
        )
        print(f"Total edge types decoded: {len(edge_constraints)}")

        # Create a copy of edge constraints for tracking remaining edges
        # Each entry has 'src', 'dst', and 'num' fields
        remaining_edges_init = [ec.copy() for ec in edge_constraints]

        # Initialize search with single-node graphs
        priority_queue = PriorityQueue()
        best_node = None
        best_distance = float('inf')

        # Try starting from different initial nodes
        initial_graphs = self._create_initial_graphs(node_alphabet, limit=min(1, total_nodes))
        initial_results = self._batch_encode_graphs(initial_graphs)

        for i, (graph, result) in enumerate(zip(
            initial_graphs[:len(initial_results)],
            initial_results
        )):
            # Correctly compute remaining alphabet
            alphabet = node_alphabet.copy()
            # Remove the node that was used in this initial graph (only one instance)
            used_node = graph['node_details'][0]
            alphabet.remove(used_node)  # Remove only first occurrence, not all equal nodes

            # Extract distance embeddings for comparison using encoder_sim
            target_dist_emb = self.encoder_sim.extract_distance_embedding(embedding)
            result_embedding = torch.tensor(result['graph_embedding'])
            candidate_dist_emb = self.encoder_sim.extract_distance_embedding(result_embedding)
            distance = self.encoder_sim.get_distance(target_dist_emb, candidate_dist_emb)
            similarity = distance  # Store distance directly for min heap

            # Use graph_hv_stack if available, otherwise use graph_embedding
            # Ensembles with non-uniform depths only return graph_embedding
            search_embedding = result.get('graph_hv_stack', result['graph_embedding'])

            node = SearchNode(
                similarity=similarity,
                graph=graph,
                remaining_alphabet=alphabet,
                remaining_edges=remaining_edges_init.copy(),
                embedding=torch.tensor(search_embedding),
                depth=1
            )

            priority_queue.put(node)
            self.tree_size += 1

            if distance < best_distance:
                best_distance = distance
                best_node = node

        # A* search loop
        while not priority_queue.empty():
            # Check for early termination (highest priority - supersedes all other checks)
            if best_distance < self.epsilon:
                print(f"Early termination confirmed: Exiting search with near-perfect match (distance: {best_distance:.6f})")
                break

            # Check time budget
            elapsed = time.time() - start_time
            if elapsed > self.time_budget:
                print(f"Time budget exceeded ({elapsed:.1f}s), returning best result")
                break

            # Get most promising node
            current = priority_queue.get()

            # Track best node (including this one if it's better)
            if current.similarity < best_distance:
                best_distance = current.similarity
                best_node = current

                # Check for early termination with near-zero distance (supersedes all other checks)
                if best_distance < self.epsilon:
                    print(f"Early termination: Found near-perfect match (distance: {best_distance:.6f} < epsilon: {self.epsilon})")
                    print(f"Graph has {len(current.graph['node_details'])} nodes, complete: {self._is_complete_graph(current)}")
                    break

                if self._is_complete_graph(current):
                    print(f"Found complete graph with distance {best_distance}")

            # If graph is complete, no need to expand further
            if self._is_complete_graph(current):
                continue

            # Mark as expanded
            current.expanded = True
            self.nodes_expanded += 1

            # Debug output: print distance of expanded node
            print(f"\n=== Node Expansion #{self.nodes_expanded} ===")
            print(f"Expanded node distance: {current.similarity:.6f}")
            print(f"Current graph: {len(current.graph['node_details'])} nodes, {len(current.graph.get('edge_indices', []))} edges")
            print(f"Remaining alphabet: {len(current.remaining_alphabet)} nodes")

            # Check memory budget
            if self.tree_size >= self.memory_budget:
                print(f"Memory budget reached ({self.tree_size} nodes), switching to greedy")
                # Perform greedy depth-first from current best
                greedy_result = self._greedy_depth_first(
                    current, embedding, time_budget=self.time_budget - elapsed
                )
                if greedy_result and greedy_result['distance'] < best_distance:
                    return greedy_result
                continue

            # Generate expansions
            expansions = self._generate_expansions(
                current.graph,
                current.remaining_alphabet,
                current.remaining_edges
            )
            # Calculate remaining edge count (total edges, not just types)
            remaining_edge_count = sum(ec.get('num', 0) for ec in current.remaining_edges)

            # Calculate number of unique node types in remaining alphabet
            unique_node_types = set()
            for node in current.remaining_alphabet:
                node_key = tuple(sorted(node.items()))
                unique_node_types.add(node_key)

            print(f"Current graph has {len(current.graph['node_details'])} nodes, "
                  f"{len(current.remaining_alphabet)} remaining alphabet, "
                  f"{len(unique_node_types)} remaining node types, "
                  f"{len(current.remaining_edges)} remaining edge types, "
                  f"{remaining_edge_count} remaining edges")
            print(f"Generated {len(expansions)} expansions")

            if not expansions:
                # Check if this is an incomplete leaf node (dead end before using all nodes)
                # Only apply penalty if there are remaining NODES, not edges
                has_remaining_nodes = len(current.remaining_alphabet) > 0
                remaining_edge_count = sum(ec.get('num', 0) for ec in current.remaining_edges)

                if has_remaining_nodes:
                    # Incomplete leaf node - apply large penalty
                    INCOMPLETE_LEAF_PENALTY = 1000.0

                    print(f"No expansions - INCOMPLETE LEAF NODE (dead end)")
                    print(f"  Remaining alphabet: {len(current.remaining_alphabet)} nodes")
                    print(f"  Remaining edges: {remaining_edge_count} edges")

                    if remaining_edge_count == 0:
                        print(f"  CRITICAL: Cannot add remaining nodes because no edges remain!")
                        print(f"  Edge-only strategy requires edges to add nodes.")

                    print(f"  Original distance: {current.similarity:.6f}")
                    print(f"  PENALTY APPLIED: +{INCOMPLETE_LEAF_PENALTY}")

                    # If this incomplete leaf was selected as best_node, invalidate it
                    if best_node is current:
                        print(f"  WARNING: This incomplete leaf was marked as best_node - invalidating!")
                        best_distance = float('inf')
                        best_node = None
                else:
                    # All nodes placed - but check if edges remain
                    if remaining_edge_count > 0:
                        print(f"\n!!! CRITICAL: No expansions - all nodes placed BUT {remaining_edge_count} edges remain !!!")
                        print(f"  This indicates edges cannot be added due to constraints.")
                        print(f"  Running diagnostics...")

                        # Run diagnostics to understand why edges can't be added
                        self._diagnose_no_edge_expansions(
                            current.graph,
                            current.remaining_edges,
                            current.remaining_alphabet
                        )

                        # This is problematic - invalidate if it's the best node
                        if best_node is current and remaining_edge_count > 0:
                            print(f"  WARNING: Incomplete graph (missing edges) was marked as best_node - invalidating!")
                            best_distance = float('inf')
                            best_node = None
                    else:
                        # All nodes placed and all edges satisfied - truly complete
                        print(f"No expansions - all nodes and edges placed (truly complete)")

                continue

            # Batch encode expansions
            expansion_graphs = [exp['graph'] for exp in expansions]
            results = self._batch_encode_graphs(expansion_graphs)

            # Create child nodes and add to queue
            child_similarities = []
            for expansion, result in zip(expansions[:len(results)], results):
                graph_embedding = torch.tensor(result['graph_embedding'])

                # Extract distance embeddings for comparison using encoder_sim
                target_dist_emb = self.encoder_sim.extract_distance_embedding(embedding)
                candidate_dist_emb = self.encoder_sim.extract_distance_embedding(graph_embedding)
                distance = self.encoder_sim.get_distance(target_dist_emb, candidate_dist_emb)
                similarity = distance  # Store distance directly for min heap
                child_similarities.append(similarity)

                # Update alphabet
                new_alphabet = current.remaining_alphabet.copy()
                if expansion['type'] == 'add_node':
                    new_alphabet.remove(expansion['node'])
                elif expansion['type'] == 'add_node_edge_guided':
                    new_alphabet.remove(expansion['node'])

                # Update remaining edges based on expansion type
                new_remaining_edges = current.remaining_edges.copy()
                if expansion['type'] in ['add_edge_guided', 'add_node_edge_guided']:
                    # An edge from the decoded set was placed - decrement its count
                    edge_constraint = expansion['edge_constraint']

                    # Find and update this edge constraint in remaining_edges
                    for i, ec in enumerate(new_remaining_edges):
                        # Check if this is the same edge constraint (same src and dst)
                        # Handle both orientations for undirected edges
                        is_forward_match = (ec['src'] == edge_constraint['src'] and
                                          ec['dst'] == edge_constraint['dst'])
                        is_reverse_match = (ec['src'] == edge_constraint['dst'] and
                                          ec['dst'] == edge_constraint['src'])

                        if is_forward_match or is_reverse_match:
                            # Create a modified copy with decremented count
                            new_remaining_edges[i] = ec.copy()
                            new_remaining_edges[i]['num'] = ec['num'] - 1

                            # Remove if count reaches 0
                            if new_remaining_edges[i]['num'] <= 0:
                                new_remaining_edges.pop(i)
                            break

                child = SearchNode(
                    similarity=similarity,
                    graph=expansion['graph'],
                    remaining_alphabet=new_alphabet,
                    remaining_edges=new_remaining_edges,
                    parent=current,
                    embedding=graph_embedding,
                    depth=current.depth + 1
                )

                current.children.append(child)
                priority_queue.put(child)
                self.tree_size += 1

                # Track best node overall (prioritize complete graphs, but track best incomplete too)
                # Handle NaN distances
                if np.isnan(distance):
                    print(f"Warning: NaN distance for graph")
                    distance = float('inf')

                if distance < best_distance:
                    best_distance = distance
                    best_node = child

                    # Check for early termination with near-zero distance (supersedes all other checks)
                    if best_distance < self.epsilon:
                        print(f"Early termination: Found near-perfect match (distance: {best_distance:.6f} < epsilon: {self.epsilon})")
                        print(f"Graph has {len(child.graph['node_details'])} nodes, complete: {self._is_complete_graph(child)}")
                        # Break out of child processing loop - we'll exit the main loop after this
                        break

                    if self._is_complete_graph(child):
                        print(f"Found complete graph with {len(child.graph['node_details'])} nodes, distance {distance}")
                    else:
                        print(f"Found better incomplete graph with {len(child.graph['node_details'])} nodes, distance {distance}")

            # Debug output: print statistics about children
            if child_similarities:
                print(f"\nChildren statistics ({len(child_similarities)} children):")
                print(f"  Average distance: {np.mean(child_similarities):.6f}")
                print(f"  Min distance: {np.min(child_similarities):.6f} (best)")
                print(f"  Max distance: {np.max(child_similarities):.6f} (worst)")
            else:
                print(f"\nNo children generated")

        # Print summary of decoded structure at the beginning
        print("\n=== Decoding Summary ===")
        self._print_decoding_summary(node_constraints, edge_constraints)

        # Print final reconstruction statistics
        if best_node:
            self._print_reconstruction_summary(
                best_node,
                node_constraints,
                edge_constraints,
                self.nodes_expanded
            )

        # Return best result found
        if best_node:
            return {
                'graph': best_node.graph,
                'embedding': best_node.embedding,
                'distance': best_node.similarity,
            }
        else:
            return {
                'graph': initial_graphs[0] if initial_graphs else {},
                'embedding': torch.zeros_like(embedding),
                'distance': float('inf'),
            }

    def _build_node_alphabet(self, node_constraints: List[dict]) -> List[dict]:
        """
        Convert node constraints to a list of node dictionaries.

        :param node_constraints: List of constraint dicts from decode_order_zero
        :return: List of node property dictionaries
        """
        node_alphabet = []
        for constraint in node_constraints:
            for _ in range(constraint['num']):
                node_alphabet.append(constraint['src'].copy())
        return node_alphabet

    def _create_initial_graphs(self, node_alphabet: List[dict], limit: int = 10) -> List[dict]:
        """
        Create initial single-node graphs to start search from.

        Ensures graphs have consistent format with graph_dict_from_mol().

        :param node_alphabet: List of available nodes
        :param limit: Maximum number of initial graphs to create
        :return: List of initial graph dicts
        """
        graphs = []
        for i, node in enumerate(node_alphabet[:limit]):
            graph = {
                'node_indices': [0],
                'node_details': [node.copy()],  # Copy to avoid modifications
                'node_attributes': [[0]],  # 2D array for consistency
                'edge_indices': [],  # Empty list for no edges
                'edge_attributes': [],  # Empty list for no edges
            }
            # Expand details for encoding - this adds property arrays
            graph = self._expand_graph_details(graph)
            graphs.append(graph)
        return graphs

    def _node_matches_properties(self, node_detail: dict, properties: dict) -> bool:
        """
        Check if a node's properties exactly match the given property constraints.

        This is used to validate that nodes meet the requirements specified in edge
        constraints from decode_order_one. A node matches if all property values
        in the constraint dict are present and equal in the node detail dict.

        :param node_detail: Dictionary containing the node's properties
        :param properties: Dictionary of required properties from edge constraint
        :return: True if all properties match exactly, False otherwise
        """
        for key, value in properties.items():
            if node_detail.get(key) != value:
                return False
        return True

    def _generate_expansions(
        self,
        graph: dict,
        remaining_alphabet: List[dict],
        remaining_edges: List[dict]
    ) -> List[dict]:
        """
        Generate all valid single-edit expansions of the current graph.

        IMPORTANT: This method now ONLY uses edge-guided expansions. Every expansion
        must consume an edge from the decoded edge constraints to ensure proper accounting.

        Two types of expansions are possible:
        1. Place an edge between two existing nodes in the graph
        2. Add a new node from the alphabet with an edge connection

        Both types consume exactly one edge from the remaining edge constraints.

        :param graph: Current graph state
        :param remaining_alphabet: Nodes still to be added
        :param remaining_edges: Edge constraints still to be placed
        :return: List of expansion dicts with 'type', 'graph', and metadata
        """
        expansions = []

        # ONLY generate edge-guided expansions
        # Every graph modification must be driven by edge constraints
        if remaining_edges:
            edge_guided_expansions = self._generate_edge_guided_expansions(
                graph, remaining_alphabet, remaining_edges
            )
            expansions.extend(edge_guided_expansions)

        # If no edges remain, no expansions are possible
        # This is the correct behavior - we can only modify the graph by consuming edges

        return expansions

    def _generate_edge_guided_expansions(
        self,
        graph: dict,
        remaining_alphabet: List[dict],
        remaining_edges: List[dict]
    ) -> List[dict]:
        """
        Generate expansions based on decoded edge constraints (PRIMARY METHOD).

        This is the ONLY expansion method used in the edge-only strategy. Every
        expansion consumes exactly one edge from the decoded constraints, ensuring
        perfect accounting.

        Two types of expansions:
        1. Edge between existing nodes: Find two nodes in the current graph that
           match the src/dst properties of an edge constraint and connect them.
        2. New node with edge: Add a node from the remaining alphabet and connect
           it to an existing node using an edge constraint.

        Each expansion respects:
        - Edge constraint properties (node types must match src/dst)
        - Node degree constraints (nodes cannot exceed their max degree)
        - Graph validity (no duplicate edges, no self-loops)

        :param graph: Current graph state
        :param remaining_alphabet: Available nodes to add
        :param remaining_edges: Edge constraints from decode_order_one with available counts
        :return: List of expansion dicts with 'type', 'graph', 'edge_constraint', and metadata
        """
        expansions = []

        for edge_constraint in remaining_edges:
            # Skip if no more of this edge type is needed
            if edge_constraint['num'] <= 0:
                continue

            src_props = edge_constraint['src']
            dst_props = edge_constraint['dst']

            # === Type 1: Place edge between existing nodes ===
            for i in graph['node_indices']:
                node_i = graph['node_details'][i]

                # Check if node i matches src properties
                if not self._node_matches_properties(node_i, src_props):
                    continue

                # Check if node i has available degree
                if not self._has_available_degree(graph, i):
                    continue

                for j in graph['node_indices']:
                    if i >= j:  # Avoid duplicates and self-loops
                        continue

                    node_j = graph['node_details'][j]

                    # Check if node j matches dst properties
                    if not self._node_matches_properties(node_j, dst_props):
                        continue

                    # Check if node j has available degree
                    if not self._has_available_degree(graph, j):
                        continue

                    # Check if edge already exists
                    if self._edge_exists(graph, i, j):
                        continue

                    # Create expansion with this edge
                    new_graph = copy.deepcopy(graph)
                    canonical_edge = self._canonicalize_edge(i, j)
                    new_graph['edge_indices'].append(canonical_edge)
                    new_graph['edge_attributes'].append([0])

                    # Validate expansion with UPDATED edge constraints
                    # This edge will consume one from the constraint, so validate with decremented count
                    if self.validate_constraints:
                        # Create temporary updated remaining_edges for validation
                        temp_remaining_edges = []
                        for ec in remaining_edges:
                            is_forward_match = (ec['src'] == edge_constraint['src'] and
                                              ec['dst'] == edge_constraint['dst'])
                            is_reverse_match = (ec['src'] == edge_constraint['dst'] and
                                              ec['dst'] == edge_constraint['src'])

                            if is_forward_match or is_reverse_match:
                                # This edge will be consumed - decrement count for validation
                                if ec['num'] > 1:
                                    temp_ec = ec.copy()
                                    temp_ec['num'] = ec['num'] - 1
                                    temp_remaining_edges.append(temp_ec)
                                # If num becomes 0, don't add to temp list
                            else:
                                temp_remaining_edges.append(ec)

                        is_valid, reason = self._is_valid_expansion(
                            new_graph, 'add_edge_guided',
                            {'src': i, 'dst': j, 'edge_constraint': edge_constraint},
                            remaining_alphabet,
                            temp_remaining_edges  # Use updated constraints
                        )
                        if not is_valid:
                            continue

                    # Expand details for encoding
                    new_graph = self._expand_graph_details(new_graph)

                    expansions.append({
                        'type': 'add_edge_guided',
                        'graph': new_graph,
                        'src': i,
                        'dst': j,
                        'edge_constraint': edge_constraint,
                    })

            # Also check reverse direction (dst->src) since edges are undirected
            for i in graph['node_indices']:
                node_i = graph['node_details'][i]

                # Check if node i matches dst properties
                if not self._node_matches_properties(node_i, dst_props):
                    continue

                if not self._has_available_degree(graph, i):
                    continue

                for j in graph['node_indices']:
                    if i >= j:  # Avoid duplicates and self-loops
                        continue

                    node_j = graph['node_details'][j]

                    # Check if node j matches src properties
                    if not self._node_matches_properties(node_j, src_props):
                        continue

                    if not self._has_available_degree(graph, j):
                        continue

                    if self._edge_exists(graph, i, j):
                        continue

                    # Create expansion with this edge
                    new_graph = copy.deepcopy(graph)
                    canonical_edge = self._canonicalize_edge(i, j)
                    new_graph['edge_indices'].append(canonical_edge)
                    new_graph['edge_attributes'].append([0])

                    if self.validate_constraints:
                        # Create temporary updated remaining_edges for validation
                        temp_remaining_edges = []
                        for ec in remaining_edges:
                            is_forward_match = (ec['src'] == edge_constraint['src'] and
                                              ec['dst'] == edge_constraint['dst'])
                            is_reverse_match = (ec['src'] == edge_constraint['dst'] and
                                              ec['dst'] == edge_constraint['src'])

                            if is_forward_match or is_reverse_match:
                                # This edge will be consumed - decrement count for validation
                                if ec['num'] > 1:
                                    temp_ec = ec.copy()
                                    temp_ec['num'] = ec['num'] - 1
                                    temp_remaining_edges.append(temp_ec)
                                # If num becomes 0, don't add to temp list
                            else:
                                temp_remaining_edges.append(ec)

                        is_valid, reason = self._is_valid_expansion(
                            new_graph, 'add_edge_guided',
                            {'src': i, 'dst': j, 'edge_constraint': edge_constraint},
                            remaining_alphabet,
                            temp_remaining_edges  # Use updated constraints
                        )
                        if not is_valid:
                            continue

                    new_graph = self._expand_graph_details(new_graph)

                    expansions.append({
                        'type': 'add_edge_guided',
                        'graph': new_graph,
                        'src': i,
                        'dst': j,
                        'edge_constraint': edge_constraint,
                    })

            # === Type 2: Add new node + edge from decoded set ===
            if remaining_alphabet:
                next_index = len(graph['node_indices'])

                for node in remaining_alphabet:
                    # Check if this node appears in the edge constraint
                    node_matches_src = self._node_matches_properties(node, src_props)
                    node_matches_dst = self._node_matches_properties(node, dst_props)

                    if not (node_matches_src or node_matches_dst):
                        continue

                    # Case A: New node matches dst, connect to existing node matching src
                    if node_matches_dst:
                        for attach_idx in graph['node_indices']:
                            attach_node = graph['node_details'][attach_idx]

                            if not self._node_matches_properties(attach_node, src_props):
                                continue

                            if not self._has_available_degree(graph, attach_idx):
                                continue

                            # Create expansion: add node + edge
                            new_graph = copy.deepcopy(graph)
                            new_graph['node_indices'].append(next_index)
                            new_graph['node_details'].append(node)
                            new_graph['node_attributes'].append([0])

                            canonical_edge = self._canonicalize_edge(attach_idx, next_index)
                            new_graph['edge_indices'].append(canonical_edge)
                            new_graph['edge_attributes'].append([0])

                            if self.validate_constraints:
                                remaining = remaining_alphabet.copy()
                                remaining.remove(node)

                                # Create temporary updated remaining_edges for validation
                                temp_remaining_edges = []
                                for ec in remaining_edges:
                                    is_forward_match = (ec['src'] == edge_constraint['src'] and
                                                      ec['dst'] == edge_constraint['dst'])
                                    is_reverse_match = (ec['src'] == edge_constraint['dst'] and
                                                      ec['dst'] == edge_constraint['src'])

                                    if is_forward_match or is_reverse_match:
                                        # This edge will be consumed - decrement count for validation
                                        if ec['num'] > 1:
                                            temp_ec = ec.copy()
                                            temp_ec['num'] = ec['num'] - 1
                                            temp_remaining_edges.append(temp_ec)
                                        # If num becomes 0, don't add to temp list
                                    else:
                                        temp_remaining_edges.append(ec)

                                is_valid, reason = self._is_valid_expansion(
                                    new_graph, 'add_node_edge_guided',
                                    {'node': node, 'attachment': attach_idx, 'edge_constraint': edge_constraint},
                                    remaining,
                                    temp_remaining_edges  # Use updated constraints
                                )
                                if not is_valid:
                                    continue

                            new_graph = self._expand_graph_details(new_graph)
                            new_graph['_node'] = node

                            expansions.append({
                                'type': 'add_node_edge_guided',
                                'graph': new_graph,
                                'node': node,
                                'attachment': attach_idx,
                                'edge_constraint': edge_constraint,
                            })

                    # Case B: New node matches src, connect to existing node matching dst
                    if node_matches_src:
                        for attach_idx in graph['node_indices']:
                            attach_node = graph['node_details'][attach_idx]

                            if not self._node_matches_properties(attach_node, dst_props):
                                continue

                            if not self._has_available_degree(graph, attach_idx):
                                continue

                            # Create expansion: add node + edge
                            new_graph = copy.deepcopy(graph)
                            new_graph['node_indices'].append(next_index)
                            new_graph['node_details'].append(node)
                            new_graph['node_attributes'].append([0])

                            canonical_edge = self._canonicalize_edge(attach_idx, next_index)
                            new_graph['edge_indices'].append(canonical_edge)
                            new_graph['edge_attributes'].append([0])

                            if self.validate_constraints:
                                remaining = remaining_alphabet.copy()
                                remaining.remove(node)

                                # Create temporary updated remaining_edges for validation
                                temp_remaining_edges = []
                                for ec in remaining_edges:
                                    is_forward_match = (ec['src'] == edge_constraint['src'] and
                                                      ec['dst'] == edge_constraint['dst'])
                                    is_reverse_match = (ec['src'] == edge_constraint['dst'] and
                                                      ec['dst'] == edge_constraint['src'])

                                    if is_forward_match or is_reverse_match:
                                        # This edge will be consumed - decrement count for validation
                                        if ec['num'] > 1:
                                            temp_ec = ec.copy()
                                            temp_ec['num'] = ec['num'] - 1
                                            temp_remaining_edges.append(temp_ec)
                                        # If num becomes 0, don't add to temp list
                                    else:
                                        temp_remaining_edges.append(ec)

                                is_valid, reason = self._is_valid_expansion(
                                    new_graph, 'add_node_edge_guided',
                                    {'node': node, 'attachment': attach_idx, 'edge_constraint': edge_constraint},
                                    remaining,
                                    temp_remaining_edges  # Use updated constraints
                                )
                                if not is_valid:
                                    continue

                            new_graph = self._expand_graph_details(new_graph)
                            new_graph['_node'] = node

                            expansions.append({
                                'type': 'add_node_edge_guided',
                                'graph': new_graph,
                                'node': node,
                                'attachment': attach_idx,
                                'edge_constraint': edge_constraint,
                            })

        return expansions

    def _generate_valid_node_expansions(
        self, graph: dict, remaining_alphabet: List[dict], remaining_edges: List[dict] = None
    ) -> List[dict]:
        """
        [DEPRECATED - NOT USED IN EDGE-ONLY STRATEGY]

        Generate valid node addition expansions respecting constraints.

        NOTE: This method is no longer used in the edge-only expansion strategy.
        All node additions must now go through edge-guided expansions to ensure
        proper edge accounting. Keeping this method for backwards compatibility only.

        :param graph: Current graph
        :param remaining_alphabet: Available nodes to add
        :param remaining_edges: Edge constraints still to be placed (optional)
        :return: List of expansion dicts
        """
        expansions = []
        next_index = len(graph['node_indices'])

        # Try adding each unique node type (avoid duplicates for efficiency)
        unique_nodes = []
        seen = set()
        for node in remaining_alphabet:
            node_key = tuple(sorted(node.items()))
            if node_key not in seen:
                seen.add(node_key)
                unique_nodes.append(node)

        for node in unique_nodes:
            # Find valid attachment points
            for attach_idx in graph['node_indices']:
                # Check if attachment point has available degree
                if not self._has_available_degree(graph, attach_idx):
                    continue

                # Create expansion with canonical edge (unidirectional)
                new_graph = copy.deepcopy(graph)
                new_graph['node_indices'].append(next_index)
                new_graph['node_details'].append(node)
                new_graph['node_attributes'].append([0])
                # Add only canonical form - bidirectionalization happens at encoding
                canonical_edge = self._canonicalize_edge(attach_idx, next_index)
                new_graph['edge_indices'].append(canonical_edge)
                new_graph['edge_attributes'].append([0])

                # Validate expansion
                if self.validate_constraints:
                    remaining = remaining_alphabet.copy()
                    remaining.remove(node)
                    is_valid, reason = self._is_valid_expansion(
                        new_graph, 'add_node', {'node': node}, remaining, remaining_edges
                    )
                    if not is_valid:
                        print(f"Rejected node expansion: {reason}")
                        continue

                # Expand details for encoding
                new_graph = self._expand_graph_details(new_graph)

                expansions.append({
                    'type': 'add_node',
                    'graph': new_graph,
                    'node': node,
                    'attachment': attach_idx,
                })

        return expansions

    def _generate_valid_edge_expansions(
        self, graph: dict, remaining_edges: List[dict] = None, debug: bool = False
    ) -> List[dict]:
        """
        [DEPRECATED - NOT USED IN EDGE-ONLY STRATEGY]

        Generate valid edge addition expansions respecting constraints.

        NOTE: This method is no longer used in the edge-only expansion strategy.
        All edge additions must now go through edge-guided expansions to ensure
        proper edge accounting. Keeping this method for backwards compatibility only.

        Adds bidirectional edges to match the format from graph_dict_from_mol().

        :param graph: Current graph
        :param remaining_edges: Edge constraints still to be placed (optional)
        :param debug: Enable diagnostic logging
        :return: List of expansion dicts
        """
        expansions = []
        rejection_counts = {'self_or_duplicate': 0, 'exists': 0, 'degree': 0, 'validation': 0}

        # Try adding edges between existing nodes
        for i in graph['node_indices']:
            for j in graph['node_indices']:
                if i >= j:  # Avoid self-loops and duplicates
                    rejection_counts['self_or_duplicate'] += 1
                    continue

                # Check if edge already exists
                if self._edge_exists(graph, i, j):
                    rejection_counts['exists'] += 1
                    continue

                # Check degree constraints
                if not (self._has_available_degree(graph, i) and
                        self._has_available_degree(graph, j)):
                    rejection_counts['degree'] += 1
                    continue

                # Create expansion with canonical edge (unidirectional)
                new_graph = copy.deepcopy(graph)
                # Add only canonical form - bidirectionalization happens at encoding
                canonical_edge = self._canonicalize_edge(i, j)
                new_graph['edge_indices'].append(canonical_edge)
                new_graph['edge_attributes'].append([0])

                # Validate expansion
                if self.validate_constraints:
                    is_valid, reason = self._is_valid_expansion(
                        new_graph, 'add_edge', {'src': i, 'dst': j}, [], remaining_edges
                    )
                    if not is_valid:
                        rejection_counts['validation'] += 1
                        continue

                # Expand details for encoding
                new_graph = self._expand_graph_details(new_graph)

                expansions.append({
                    'type': 'add_edge',
                    'graph': new_graph,
                    'src': i,
                    'dst': j,
                })

        if debug and len(expansions) == 0:
            print(f"  Edge expansion rejections: {rejection_counts}")

        return expansions

    def _diagnose_no_edge_expansions(
        self,
        graph: dict,
        remaining_edges: List[dict],
        remaining_alphabet: List[dict]
    ) -> None:
        """
        Diagnose why no edge expansions can be generated.

        This method provides detailed information about the current graph state
        and explains why edges cannot be added, helping to debug reconstruction issues.

        :param graph: Current graph state
        :param remaining_edges: Edge constraints still to be placed
        :param remaining_alphabet: Nodes still to be added
        """
        print(f"\n  === EDGE EXPANSION DIAGNOSTICS ===")
        print(f"  Graph state:")
        print(f"    Nodes: {len(graph['node_indices'])}")
        print(f"    Edges: {len(graph.get('edge_indices', []))}")
        print(f"    Remaining alphabet: {len(remaining_alphabet)}")
        print(f"    Remaining edge constraints: {len(remaining_edges)}")

        # Print node degree information
        print(f"\n  Node degree status:")
        for idx in graph['node_indices']:
            node_detail = graph['node_details'][idx]
            available = self._get_available_degree(graph, idx)
            max_degree = node_detail.get('node_degrees', node_detail.get('node_degree', 'inf'))

            # Count current edges
            current_edges = []
            for i, j in graph.get('edge_indices', []):
                if i == idx or j == idx:
                    current_edges.append((i, j))

            print(f"    Node {idx}: degree={max_degree}, available={available}, current_edges={len(current_edges)}")

        # Print remaining edge constraints
        if remaining_edges:
            print(f"\n  Remaining edge constraints:")
            for i, ec in enumerate(remaining_edges):
                src_str = ", ".join(f"{k}={v}" for k, v in ec['src'].items())
                dst_str = ", ".join(f"{k}={v}" for k, v in ec['dst'].items())
                print(f"    {i+1}. Count: {ec['num']}")
                print(f"       Src: {{{src_str}}}")
                print(f"       Dst: {{{dst_str}}}")

                # Check if any nodes match these constraints
                matching_src = []
                matching_dst = []
                for node_idx in graph['node_indices']:
                    node = graph['node_details'][node_idx]
                    if self._node_matches_properties(node, ec['src']):
                        matching_src.append(node_idx)
                    if self._node_matches_properties(node, ec['dst']):
                        matching_dst.append(node_idx)

                print(f"       Nodes matching src: {matching_src}")
                print(f"       Nodes matching dst: {matching_dst}")

                # Check if any edges can be formed
                can_form = False
                for src_idx in matching_src:
                    for dst_idx in matching_dst:
                        if src_idx != dst_idx and not self._edge_exists(graph, src_idx, dst_idx):
                            if self._has_available_degree(graph, src_idx) and self._has_available_degree(graph, dst_idx):
                                can_form = True
                                break
                    if can_form:
                        break

                print(f"       Can form edge: {can_form}")

        # In edge-only strategy, we only use edge-guided expansions
        if remaining_edges:
            print(f"\n  Attempting edge-guided expansions (edge-only strategy):")
            edge_guided = self._generate_edge_guided_expansions(
                graph, remaining_alphabet, remaining_edges
            )
            print(f"    Generated {len(edge_guided)} edge-guided expansions")

            if len(edge_guided) == 0:
                print(f"    No valid edge placements possible - likely due to degree constraints")
        else:
            print(f"\n  No remaining edges - expansion impossible in edge-only strategy")
            if len(remaining_alphabet) > 0:
                print(f"    WARNING: {len(remaining_alphabet)} nodes cannot be added without edges!")

        print(f"  === END DIAGNOSTICS ===\n")

    def _is_valid_expansion(
        self, graph: dict, expansion_type: str,
        expansion_data: dict, remaining_alphabet: List[dict],
        remaining_edges: List[dict] = None
    ) -> Tuple[bool, str]:
        """
        Check if an expansion satisfies all structural constraints.

        :param graph: Graph after expansion
        :param expansion_type: 'add_node' or 'add_edge'
        :param expansion_data: Details about the expansion
        :param remaining_alphabet: Nodes still to be added
        :param remaining_edges: Edge constraints still to be placed (optional)
        :return: (is_valid, reason_if_invalid)
        """
        # Check degree constraints
        for idx in graph['node_indices']:
            is_valid, reason = self._check_degree_constraint(graph, idx)
            if not is_valid:
                return False, reason

        # Check connectivity preservation (for remaining nodes)
        is_valid, reason = self._check_connectivity_preservation(
            graph, remaining_alphabet
        )
        if not is_valid:
            return False, reason

        # Check edge capacity preservation (for remaining edges)
        if remaining_edges is not None:
            is_valid, reason = self._check_edge_capacity_preservation(
                graph, remaining_edges
            )
            if not is_valid:
                return False, reason

        return True, "Valid"

    def _check_degree_constraint(self, graph: dict, node_index: int) -> Tuple[bool, str]:
        """
        Check if a node respects its encoded degree constraint.

        Handles bidirectional edges: (i,j) and (j,i) count as one edge.

        :param graph: Current graph
        :param node_index: Index of node to check
        :return: (is_valid, reason)
        """
        node_detail = graph['node_details'][node_index]

        # Check if degree constraint exists
        if 'node_degrees' not in node_detail and 'node_degree' not in node_detail:
            return True, "No degree constraint"

        # Get expected degree
        if 'node_degrees' in node_detail:
            expected_degree = node_detail['node_degrees']
        else:
            expected_degree = node_detail['node_degree']

        # Count current degree (handle bidirectional edges)
        # Use a set to track unique undirected edges
        unique_edges = set()
        for i, j in graph['edge_indices']:
            if i == node_index or j == node_index:
                # Normalize edge to (min, max) to treat (i,j) and (j,i) as same edge
                edge = tuple(sorted([i, j]))
                unique_edges.add(edge)

        current_degree = len(unique_edges)

        if current_degree > expected_degree:
            return False, f"Node {node_index} exceeds degree {expected_degree}"

        return True, "OK"

    def _check_connectivity_preservation(
        self, graph: dict, remaining_alphabet: List[dict]
    ) -> Tuple[bool, str]:
        """
        Ensure graph maintains attachment points for remaining nodes.

        :param graph: Current graph
        :param remaining_alphabet: Nodes still to be added
        :return: (is_valid, reason)
        """
        if not remaining_alphabet:
            return True, "Alphabet empty"

        # Count nodes with available attachment capacity
        nodes_with_capacity = 0
        total_capacity = 0
        for idx in graph['node_indices']:
            available = self._get_available_degree(graph, idx)
            if available > 0:
                nodes_with_capacity += 1
                total_capacity += available

        # Need at least enough capacity for remaining nodes
        # Each remaining node needs at least one attachment point
        if total_capacity < 1:
            return False, f"Total capacity {total_capacity} < 1"

        return True, "OK"

    def _check_edge_capacity_preservation(
        self, graph: dict, remaining_edges: List[dict]
    ) -> Tuple[bool, str]:
        """
        Ensure graph has at least SOME degree capacity if edges remain.

        This is a WEAK check - we only verify that at least one edge could potentially
        be added (i.e., total capacity >= 2), not that all remaining edges can be added.
        This prevents completely dead-end states while allowing flexible exploration.

        :param graph: Current graph state
        :param remaining_edges: Edge constraints still to be placed
        :return: (is_valid, reason_if_invalid)
        """
        if not remaining_edges:
            return True, "No remaining edges"

        # Calculate total remaining edge count
        total_remaining_edges = sum(ec.get('num', 0) for ec in remaining_edges)

        if total_remaining_edges == 0:
            return True, "No edges to place"

        # Calculate total available degree capacity across all nodes
        total_capacity = 0
        for idx in graph['node_indices']:
            available = self._get_available_degree(graph, idx)
            total_capacity += available

        # ============================================================================
        # IMPORTANT: This is a WEAK check by design - DO NOT CHANGE!
        # ============================================================================
        # We ONLY verify that at least ONE edge could potentially be added.
        # We do NOT check if there's enough capacity for ALL remaining edges.
        #
        # Rationale:
        # - Strong checking (capacity >= 2 * remaining_edges) would be too restrictive
        # - It would prematurely reject valid exploration paths
        # - A* search should have flexibility to explore different edge placements
        # - The weak check only prevents truly dead-end states (zero capacity)
        #
        # Each edge requires 1 degree units (one from each endpoint), so we need
        # at least total_capacity >= 1 to place any edge at all.
        # ============================================================================
        if total_capacity < 1:
            return False, f"No edge capacity remaining: total capacity is {total_capacity} (need at least 2 for one edge)"

        return True, "OK"

    def _has_available_degree(self, graph: dict, node_index: int) -> bool:
        """
        Check if a node has available degree capacity.

        :param graph: Current graph
        :param node_index: Node to check
        :return: True if node can accept more edges
        """
        return self._get_available_degree(graph, node_index) > 0

    def _get_available_degree(self, graph: dict, node_index: int) -> float:
        """
        Get the number of additional edges a node can accept.

        Handles bidirectional edges: (i,j) and (j,i) count as one edge.

        :param graph: Current graph
        :param node_index: Node to check
        :return: Available degree (inf if unconstrained)
        """
        node_detail = graph['node_details'][node_index]

        # Get degree constraint if it exists
        if 'node_degrees' in node_detail:
            max_degree = node_detail['node_degrees']
        elif 'node_degree' in node_detail:
            max_degree = node_detail['node_degree']
        else:
            return float('inf')

        # Count current edges (handle bidirectional edges)
        unique_edges = set()
        for i, j in graph['edge_indices']:
            if i == node_index or j == node_index:
                # Normalize edge to (min, max) to treat (i,j) and (j,i) as same edge
                edge = tuple(sorted([i, j]))
                unique_edges.add(edge)

        current_degree = len(unique_edges)

        return max(0, max_degree - current_degree)

    def _edge_exists(self, graph: dict, i: int, j: int) -> bool:
        """
        Check if an edge already exists between two nodes.

        Since edges are stored in canonical form, we normalize the query edge
        and check for exact match.

        :param graph: Current graph
        :param i: First node index
        :param j: Second node index
        :return: True if edge exists
        """
        canonical_edge = self._canonicalize_edge(i, j)
        for src, dst in graph['edge_indices']:
            if (src, dst) == canonical_edge:
                return True
        return False

    def _batch_encode_graphs(self, graphs: List[dict]) -> List[dict]:
        """
        Efficiently encode multiple graphs in batches.

        Filters out results with NaN or inf embeddings.

        Edges are passed in canonical (unidirectional) form directly to the encoder,
        which is assumed to handle edge directionality internally.

        :param graphs: List of graph dicts to encode (with canonical edges)
        :return: List of result dicts with valid embeddings
        """
        if not graphs:
            return []

        # Process in batches for efficiency
        all_results = []
        for i in range(0, len(graphs), self.batch_size):
            batch = graphs[i:i + self.batch_size]
            results = self.encoder.forward_graphs(batch)

            # Filter out results with NaN or inf embeddings
            for result in results:
                # Try graph_hv_stack first, fall back to graph_embedding
                # Ensembles with non-uniform depths only return graph_embedding
                embedding = result.get('graph_hv_stack', result.get('graph_embedding', None))
                if embedding is not None:
                    # Convert to tensor if needed
                    if isinstance(embedding, np.ndarray):
                        embedding = torch.from_numpy(embedding)

                    # Check for NaN or inf
                    if not (np.isnan(embedding).any() or np.isinf(embedding).any()):
                        all_results.append(result)
                    else:
                        # Skip graphs with invalid embeddings
                        pass

        return all_results

    def _greedy_depth_first(
        self, start_node: SearchNode, target_embedding: torch.Tensor,
        time_budget: float = 10.0
    ) -> Optional[dict]:
        """
        Perform greedy depth-first search from a starting node.

        :param start_node: Node to start greedy search from
        :param target_embedding: Target embedding to match
        :param time_budget: Time limit for greedy search
        :return: Best result dict or None
        """
        start_time = time.time()
        current = start_node
        total_nodes = len(start_node.remaining_alphabet) + len(start_node.graph['node_details'])

        while len(current.graph['node_details']) < total_nodes:
            # Check time
            if time.time() - start_time > time_budget:
                break

            # Generate expansions
            expansions = self._generate_expansions(
                current.graph,
                current.remaining_alphabet,
                current.remaining_edges
            )

            if not expansions:
                break

            # Encode and find best
            expansion_graphs = [exp['graph'] for exp in expansions]
            results = self._batch_encode_graphs(expansion_graphs)

            if not results:
                break

            # Select best expansion
            best_idx = 0
            best_distance = float('inf')

            for idx, result in enumerate(results):
                # Extract distance embeddings for comparison using encoder_sim
                target_dist_emb = self.encoder_sim.extract_distance_embedding(target_embedding)
                result_embedding = torch.tensor(result['graph_embedding'])
                candidate_dist_emb = self.encoder_sim.extract_distance_embedding(result_embedding)
                distance = self.encoder_sim.get_distance(
                    target_dist_emb, candidate_dist_emb
                )
                if distance < best_distance:
                    best_distance = distance
                    best_idx = idx

            # Move to best expansion
            expansion = expansions[best_idx]
            result = results[best_idx]

            # Update alphabet if node was added
            new_alphabet = current.remaining_alphabet.copy()
            if expansion['type'] == 'add_node':
                new_alphabet.remove(expansion['node'])
            elif expansion['type'] == 'add_node_edge_guided':
                new_alphabet.remove(expansion['node'])

            # Update remaining edges if edge-guided expansion was used
            new_remaining_edges = current.remaining_edges.copy()
            if expansion['type'] in ['add_edge_guided', 'add_node_edge_guided']:
                edge_constraint = expansion['edge_constraint']
                for i, ec in enumerate(new_remaining_edges):
                    # Handle both orientations for undirected edges
                    is_forward_match = (ec['src'] == edge_constraint['src'] and
                                      ec['dst'] == edge_constraint['dst'])
                    is_reverse_match = (ec['src'] == edge_constraint['dst'] and
                                      ec['dst'] == edge_constraint['src'])

                    if is_forward_match or is_reverse_match:
                        new_remaining_edges[i] = ec.copy()
                        new_remaining_edges[i]['num'] = ec['num'] - 1
                        if new_remaining_edges[i]['num'] <= 0:
                            new_remaining_edges.pop(i)
                        break

            # Use graph_hv_stack if available, otherwise use graph_embedding
            # Ensembles with non-uniform depths only return graph_embedding
            search_embedding = result.get('graph_hv_stack', result['graph_embedding'])

            current = SearchNode(
                similarity=best_distance,  # Store distance directly for min heap
                graph=expansion['graph'],
                remaining_alphabet=new_alphabet,
                remaining_edges=new_remaining_edges,
                embedding=torch.tensor(search_embedding),
                parent=current,
                depth=current.depth + 1
            )

        return {
            'graph': current.graph,
            'embedding': current.embedding,
            'distance': current.similarity,
        }

    def _expand_graph_details(self, graph: dict) -> dict:
        """
        Expand node details into array format for encoding.

        :param graph: Graph dict with node_details list
        :return: Graph dict with expanded property arrays
        """
        if not graph['node_details']:
            return graph

        graph = copy.deepcopy(graph)
        detail_keys = graph['node_details'][0].keys()

        for key in detail_keys:
            graph[key] = np.array([
                graph['node_details'][idx][key]
                for idx in graph['node_indices']
            ])

        return graph

    def _canonicalize_edge(self, i: int, j: int) -> Tuple[int, int]:
        """
        Convert edge to canonical form (min, max) for consistent representation.

        This ensures undirected edges are stored uniquely, avoiding duplication.

        :param i: First node index
        :param j: Second node index
        :return: Canonical edge tuple (min, max)
        """
        return (min(i, j), max(i, j))

    def _is_complete_graph(self, node: SearchNode) -> bool:
        """
        Check if a search node represents a complete graph.

        A graph is complete only if:
        - All nodes from the alphabet have been used (remaining_alphabet is empty)
        - All edge constraints from decode_order_one have been satisfied (no remaining edges)

        This is the proper definition of completeness for the reconstruction task.
        A graph that has the correct number of nodes but unsatisfied edge constraints
        should NOT be considered complete.

        :param node: SearchNode to check for completeness
        :return: True if graph is complete (all nodes and edges used), False otherwise
        """
        # Check if all nodes from the alphabet have been used
        if len(node.remaining_alphabet) > 0:
            return False

        # Check if all edge constraints have been satisfied
        # Sum up all remaining edge counts
        total_remaining_edges = sum(ec.get('num', 0) for ec in node.remaining_edges)
        if total_remaining_edges > 0:
            return False

        return True

    def _print_decoding_summary(
        self,
        node_constraints: List[dict],
        edge_constraints: List[dict]
    ) -> None:
        """
        Print a summary of node and edge types decoded from the graph embedding.

        This method provides a formatted output showing what structural information
        was extracted from the hyperdimensional embedding during reconstruction.

        :param node_constraints: List of node type constraints from decode_order_zero
        :param edge_constraints: List of edge type constraints from decode_order_one
        """
        print("\n" + "=" * 70)
        print("DECODED GRAPH STRUCTURE FROM EMBEDDING")
        print("=" * 70)

        # Print node types summary
        print(f"\n📊 NODE TYPES DECODED: {len(node_constraints)}")
        print("-" * 70)

        if node_constraints:
            total_nodes = sum(c['num'] for c in node_constraints)
            print(f"Total nodes across all types: {total_nodes}\n")

            for i, constraint in enumerate(node_constraints, 1):
                node_props = constraint['src']
                count = constraint['num']

                # Format properties as key=value pairs
                props_str = ", ".join(f"{k}={v}" for k, v in node_props.items())
                print(f"  {i}. Count: {count:2d}  |  Properties: {{{props_str}}}")
        else:
            print("  No node types detected")

        # Print edge types summary
        print(f"\n🔗 EDGE TYPES DECODED: {len(edge_constraints)}")
        print("-" * 70)

        if edge_constraints:
            total_edges = sum(c['num'] for c in edge_constraints)
            print(f"Total edges across all types: {total_edges}\n")

            for i, constraint in enumerate(edge_constraints, 1):
                src_props = constraint['src']
                dst_props = constraint['dst']
                count = constraint['num']

                # Format source and destination properties
                src_str = ", ".join(f"{k}={v}" for k, v in src_props.items())
                dst_str = ", ".join(f"{k}={v}" for k, v in dst_props.items())

                print(f"  {i}. Count: {count:2d}")
                print(f"     Source: {{{src_str}}}")
                print(f"     Target: {{{dst_str}}}")
                if i < len(edge_constraints):
                    print()
        else:
            print("  No edge types detected")

        print("=" * 70 + "\n")

    def _print_reconstruction_summary(
        self,
        best_node: SearchNode,
        node_constraints: List[dict],
        edge_constraints: List[dict],
        expansions_performed: int
    ) -> None:
        """
        Print detailed statistics about the reconstructed graph.

        Shows the final graph structure, node types, edge types, and how well
        the reconstruction matched the decoded constraints.

        :param best_node: The best SearchNode found during reconstruction
        :param node_constraints: Original node constraints from decode_order_zero
        :param edge_constraints: Original edge constraints from decode_order_one
        :param expansions_performed: Number of node expansions during search
        """
        graph = best_node.graph

        print("\n" + "=" * 70)
        print("RECONSTRUCTION RESULTS")
        print("=" * 70)

        # Overall statistics
        print(f"\n📈 SEARCH STATISTICS:")
        print("-" * 70)
        print(f"  Nodes expanded: {expansions_performed}")
        print(f"  Final distance: {best_node.similarity:.6f}")
        print(f"  Graph depth: {best_node.depth}")
        print(f"  Completeness: {'✓ COMPLETE' if self._is_complete_graph(best_node) else '✗ INCOMPLETE'}")

        remaining_nodes = len(best_node.remaining_alphabet)
        remaining_edges_count = sum(ec.get('num', 0) for ec in best_node.remaining_edges)

        if remaining_nodes > 0 or remaining_edges_count > 0:
            print(f"  Remaining nodes: {remaining_nodes}")
            print(f"  Remaining edges: {remaining_edges_count}")

        # Graph structure
        print(f"\n🔷 GRAPH STRUCTURE:")
        print("-" * 70)
        print(f"  Total nodes: {len(graph['node_indices'])}")
        print(f"  Total edges: {len(graph.get('edge_indices', []))}")

        # Count node types in reconstructed graph
        node_type_counts = {}
        for node_detail in graph['node_details']:
            # Create a hashable key from node properties
            node_key = tuple(sorted(node_detail.items()))
            node_type_counts[node_key] = node_type_counts.get(node_key, 0) + 1

        # Count edge types in reconstructed graph
        edge_type_counts = {}
        for i, j in graph.get('edge_indices', []):
            src_node = graph['node_details'][i]
            dst_node = graph['node_details'][j]

            # Create canonical edge key (ordered to handle undirected edges)
            src_key = tuple(sorted(src_node.items()))
            dst_key = tuple(sorted(dst_node.items()))
            edge_key = tuple(sorted([src_key, dst_key]))

            edge_type_counts[edge_key] = edge_type_counts.get(edge_key, 0) + 1

        # Since edges are stored in canonical form, we need to account for that
        # Each edge appears once in the list, not twice

        # Print node types
        print(f"\n📊 NODE TYPES IN RECONSTRUCTED GRAPH:")
        print("-" * 70)

        if node_type_counts:
            for i, (node_key, count) in enumerate(sorted(node_type_counts.items()), 1):
                props_dict = dict(node_key)
                props_str = ", ".join(f"{k}={v}" for k, v in props_dict.items())

                # Find matching constraint
                expected_count = "?"
                for constraint in node_constraints:
                    constraint_key = tuple(sorted(constraint['src'].items()))
                    if constraint_key == node_key:
                        expected_count = constraint['num']
                        break

                match_symbol = "✓" if count == expected_count else "✗"
                print(f"  {i}. {match_symbol} Count: {count:2d} (expected: {expected_count:2d})  |  {{{props_str}}}")
        else:
            print("  No nodes in graph")

        # Print edge types
        print(f"\n🔗 EDGE TYPES IN RECONSTRUCTED GRAPH:")
        print("-" * 70)

        if edge_type_counts:
            for i, (edge_key, count) in enumerate(sorted(edge_type_counts.items()), 1):
                src_key, dst_key = edge_key
                src_props = dict(src_key)
                dst_props = dict(dst_key)

                src_str = ", ".join(f"{k}={v}" for k, v in src_props.items())
                dst_str = ", ".join(f"{k}={v}" for k, v in dst_props.items())

                # Find matching constraint
                expected_count = "?"
                for constraint in edge_constraints:
                    constraint_src_key = tuple(sorted(constraint['src'].items()))
                    constraint_dst_key = tuple(sorted(constraint['dst'].items()))

                    # Check both orientations
                    if ((constraint_src_key == src_key and constraint_dst_key == dst_key) or
                        (constraint_src_key == dst_key and constraint_dst_key == src_key)):
                        expected_count = constraint['num']
                        break

                match_symbol = "✓" if count == expected_count else "✗"
                print(f"  {i}. {match_symbol} Count: {count:2d} (expected: {expected_count:2d})")
                print(f"     Source: {{{src_str}}}")
                print(f"     Target: {{{dst_str}}}")
                if i < len(edge_type_counts):
                    print()
        else:
            print("  No edges in graph")

        print("=" * 70 + "\n")

    def _bidirectionalize_edges(self, graph: dict) -> dict:
        """
        Convert canonical (unidirectional) edges to bidirectional format expected by encoder.

        The encoder expects edges in both directions: (i,j) and (j,i).
        This method takes a graph with canonical edges and duplicates them.

        :param graph: Graph dict with canonical edges
        :return: Graph dict with bidirectional edges
        """
        graph = copy.deepcopy(graph)

        if not graph.get('edge_indices'):
            return graph

        # Convert to list if it's an array
        edge_indices = graph['edge_indices']
        if isinstance(edge_indices, np.ndarray):
            edge_indices = edge_indices.tolist()

        # Get edge attributes if they exist
        edge_attributes = graph.get('edge_attributes', [])
        if isinstance(edge_attributes, np.ndarray):
            edge_attributes = edge_attributes.tolist()

        # Create bidirectional edges
        bidirectional_edges = []
        bidirectional_attrs = []

        for idx, (i, j) in enumerate(edge_indices):
            # Add both directions
            bidirectional_edges.append((i, j))
            bidirectional_edges.append((j, i))

            # Duplicate attributes if they exist
            if idx < len(edge_attributes):
                attr = edge_attributes[idx]
                bidirectional_attrs.append(attr)
                bidirectional_attrs.append(attr)

        # Update graph with bidirectional edges
        graph['edge_indices'] = bidirectional_edges
        if bidirectional_attrs:
            graph['edge_attributes'] = bidirectional_attrs

        return graph

