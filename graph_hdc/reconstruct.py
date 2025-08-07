import copy
import random

import torch
import numpy as np
from rich.pretty import pprint

from graph_hdc.models import HyperNet


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Calculate the cosine distance between two tensors.

    :param a: First tensor.
    :param b: Second tensor.
    :return: Cosine distance (1 - cosine similarity).
    """
    if a.dim() == 2 and b.dim() == 2:
        # Calculate cosine similarity for each row and return the average
        a_norm = a / a.norm(dim=1, keepdim=True)
        b_norm = b / b.norm(dim=1, keepdim=True)
        cosine_sims = (a_norm * b_norm).sum(dim=1)
        avg_cosine_sim = cosine_sims.mean().item()
        return 1.0 - avg_cosine_sim
    else:
        a_norm = a / a.norm()
        b_norm = b / b.norm()
        cosine_sim = torch.dot(a_norm, b_norm).item()
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


class GraphReconstructor:
    
    def __init__(self, 
                 encoder: HyperNet,
                 population_size: int = 10,
                 distance_func: callable = cosine_distance,
                 #distance_func: callable = manhattan_distance,
                 ):
        
        self.encoder = encoder
        self.population_size = population_size
        self.distance_func = distance_func
    
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
                    
                    result_embedding = torch.tensor(result['graph_hv_stack'])
                    
                    in_blacklist = any(
                        torch.allclose(result_embedding, emb, atol=1e-1)
                        for emb in blacklist_embeddings
                    )
                    if in_blacklist:
                        print('Skipping due to blacklist')
                        continue
                    
                    result['distance'] = self.distance_func(
                        embedding, result_embedding
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
            result['distance'] = self.distance_func(
                embedding, torch.tensor(result['graph_hv_stack'])
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
            
            # insert the edge connection
            modified_graph['edge_indices'].append((index, next_index))
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
        
        # insert the edge connection
        modified_graph['edge_indices'].append((i, j))
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
        
        