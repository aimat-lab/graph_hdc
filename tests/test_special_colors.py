import os
import random

import torch
import networkx as nx
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from rich.pretty import pprint
from torch.nn.functional import normalize, sigmoid
from torch_geometric.utils import scatter
from torch_geometric.loader import DataLoader

from graph_hdc.utils import nx_random_uniform_edge_weight
from graph_hdc.special.colors import generate_random_color_nx
from graph_hdc.special.colors import graph_dict_from_color_nx
from graph_hdc.special.colors import make_color_node_encoder_map
from graph_hdc.models import HyperNet
from graph_hdc.graph import data_list_from_graph_dicts
from .utils import ARTIFACTS_PATH


class TestColorGraphs:
    
    def test_generate_random_color_nx_basically_works(self):
        """
        generate_random_color_nx should generate a random color graph structure with the given number 
        of nodes and edges as a networkx graph object.
        """
        g: nx.Graph = generate_random_color_nx(
            num_nodes=10,
            num_edges=15,
            colors=['red', 'green', 'blue'],
            seed=42,
        )
        
        assert isinstance(g, nx.Graph)
        assert g.number_of_nodes() == 10
        assert g.number_of_edges() == 15

        # ~ plotting the graph for visual inspection
        fig, ax = plt.subplots()
        pos = nx.spring_layout(g, seed=42)
        nx.draw(g, pos, ax=ax, with_labels=True, node_color=[g.nodes[n]['color'] for n in g.nodes])

        fig_path = os.path.join(ARTIFACTS_PATH, 'test_generate_random_color_nx.png')
        fig.savefig(fig_path)
        
    def test_graph_dict_from_color_nx(self):
        """
        graph_dict_from_color_nx should generate a graph dict representation of a color graph that
        is given as a networkx graph object.
        """
        g: nx.Graph = generate_random_color_nx(
            num_nodes=10,
            num_edges=15,
            colors=['red', 'green', 'blue'],
            seed=42,
        )
        
        graph: dict = graph_dict_from_color_nx(g)
        assert isinstance(graph, dict)
        assert 'node_color' in graph
        assert 'node_degree' in graph
        
    def test_hyper_net_encoding(self):
        """
        It should be possible to use the HyperNet in conjunction with a color graph and the 
        make_color_node_encoder_map function to encode a color graph specifically into the high 
        dimensional latent space.
        """
        # ~ generate data
        num_graphs = 10
        graphs = [
            graph_dict_from_color_nx(generate_random_color_nx(
                num_nodes=10,
                num_edges=15,
                colors=['red', 'green', 'blue'],
                seed=42,
            ))
            for _ in range(num_graphs)    
        ]
        
        dim = 1000
        node_encoder_map = make_color_node_encoder_map(dim)
        hyper_net = HyperNet(
            hidden_dim=dim,
            depth=3,
            node_encoder_map=node_encoder_map,
        )
        
        data_list = data_list_from_graph_dicts(graphs)
        data_loader = DataLoader(data_list, batch_size=num_graphs, shuffle=False)
        data = next(iter(data_loader))
        
        result = hyper_net.forward(data)
        embedding = result['graph_embedding']
        
        assert isinstance(result, dict)
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (num_graphs, dim)
        
    def test_graph_similarity_with_minimal_perturbation(self):
        """
        When introducing a minimal perturbation to a graph, the corresponding embeddings should still 
        have a high cosine similarity.
        """
        # Generate the original graph
        g: nx.Graph = generate_random_color_nx(
            num_nodes=10,
            num_edges=15,
            colors=['red', 'green', 'blue'],
            seed=42,
        )
        graph = graph_dict_from_color_nx(g)
        
        # Setup the encoder hyper net
        dim = 10_000
        hyper_net = HyperNet(
            hidden_dim=dim,
            depth=3,
            node_encoder_map=make_color_node_encoder_map(dim),
        )
        
        # Initial embedding
        data = next(iter(DataLoader(data_list_from_graph_dicts([graph]), batch_size=1)))
        result = hyper_net.forward(data)
        emb_original = result['graph_embedding'].detach()
        
        # Perturb the graph by changing the color of one node
        g_perturbed = g.copy()
        node_to_perturb = list(g_perturbed.nodes)[0]
        g_perturbed.nodes[node_to_perturb]['color'] = 'green'  # Change color to a new one
        graph_perturbed = graph_dict_from_color_nx(g_perturbed)
        
        # Embedding of the perturbed graph
        data_perturbed = next(iter(DataLoader(data_list_from_graph_dicts([graph_perturbed]), batch_size=1)))
        result_perturbed = hyper_net.forward(data_perturbed)
        emb_perturbed = result_perturbed['graph_embedding'].detach()
        
        # Calculate cosine similarity
        cosine_similarity = torch.nn.functional.cosine_similarity(emb_original, emb_perturbed).item()
        #cosine_similarity = (normalize(emb_original) * normalize(emb_perturbed)).sum().item()
        
        # Plot the original and perturbed graphs
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        pos = nx.spring_layout(g, seed=42)
        nx.draw(g, pos, ax=axes[0], with_labels=True, node_color=[g.nodes[n]['color'] for n in g.nodes])
        axes[0].set_title('Original Graph')
        
        pos_perturbed = nx.spring_layout(g_perturbed, seed=42)
        nx.draw(g_perturbed, pos_perturbed, ax=axes[1], with_labels=True, node_color=[g_perturbed.nodes[n]['color'] for n in g_perturbed.nodes])
        axes[1].set_title('Perturbed Graph')
        
        fig.suptitle(f'Cosine Similarity: {cosine_similarity:.4f}')
        
        fig_path = os.path.join(ARTIFACTS_PATH, 'graph_similarity_with_minimal_perturbation.png')
        fig.savefig(fig_path)
        
    def test_optimize_graph_structure_with_edge_weights(self):
        """
        It should be possible to use a gradient descent optimizer to optimize the graph structure of a graph
        by using the cont. edge_weights property and then defining a threshold to decide which edges to keep.
        """
        # setup the encoder hyper net
        seed = 42
        dim = 10_000
        colors = ['red', 'green', 'blue']
        #colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'gray',  'white']
        hyper_net = HyperNet(
            hidden_dim=dim,
            depth=3,
            node_encoder_map=make_color_node_encoder_map(dim, colors=colors),
        )
        
        # generate the graph
        g: nx.Graph = generate_random_color_nx(
            num_nodes=10,
            num_edges=18,
            colors=colors,
            seed=seed,
        )
        graph = graph_dict_from_color_nx(g)
        
        # initial embedding
        data = next(iter(DataLoader(data_list_from_graph_dicts([graph]), batch_size=1)))
    
        result: dict = hyper_net.forward(data)
        emb = result['graph_embedding'].detach()
        
        # optimize the graph structure
        # create a fully connected graph with the same nodes
        g_full = nx.Graph()
        for node, data in g.nodes(data=True):
            g_full.add_node(node, **data)
            
        for u in g_full.nodes():
            for v in g_full.nodes():
                if v > u:
                    if g.has_edge(u, v):
                        g_full.add_edge(u, v)
                    elif random.random() < 0.7:
                        g_full.add_edge(u, v)
            
        g_rand = nx_random_uniform_edge_weight(
            g_full.copy(),
            lo=0,
            hi=0,
        )
        
        # convert to graph dict
        graph_rand = graph_dict_from_color_nx(g_rand)
        data = next(iter(DataLoader(data_list_from_graph_dicts([graph_rand]), batch_size=1)))
        data.edge_weight.requires_grad = True
        
        # setup optimizer
        optimizer = torch.optim.Adam([data.edge_weight], lr=0.1)
        #optimizer = torch.optim.LBFGS([data.edge_weight], lr=0.1)
        
        # optimization loop
        for epoch in range(50):  # number of epochs
            
            def closure():
                
                optimizer.zero_grad()
                result = hyper_net.forward(data)
                embedding = result['graph_embedding']
                #loss = - (emb * embedding).sum()
                #loss = (emb - embedding).pow(2).mean()
                loss = (emb - embedding).abs().mean()
                loss.backward()
                print(
                    f' * epoch {epoch}'
                    f' - loss: {loss.item()}'
                    f' - edge weight (first 5): {data.edge_weight[:5]}'
                    f' - grad norm: {data.edge_weight.grad.norm()}'
                )
                return loss
            
            optimizer.step(closure)
        
        # check if edge weights have been updated
        optimized_edge_weights = sigmoid(data.edge_weight).detach().numpy()
        # create a new graph with edges where the optimized edge weights are above a threshold
        threshold = 0.5
        g_opt = nx.Graph()
        g_opt.add_nodes_from(g.nodes(data=True))
        
        for i, (u, v) in enumerate(g_rand.edges()):
            if optimized_edge_weights[i] > threshold:
                g_opt.add_edge(u, v)
                
        for i, data in g_opt.nodes(data=True):
            g_opt.nodes[i]['degree'] = g_opt.degree(i)
        
        # Embedding of the optimized graph
        data_optimized = next(iter(DataLoader(data_list_from_graph_dicts([graph_dict_from_color_nx(g_opt)]), batch_size=1)))
        result_optimized = hyper_net.forward(data_optimized)
        emb_optimized = result_optimized['graph_embedding'].detach()

        # Calculate cosine similarity
        cosine_similarity_optimized = torch.nn.functional.cosine_similarity(emb, emb_optimized).item()
        
        # plot the original and optimized graphs
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'ADAM-optimized Graph Structure - 50 epochs\n'
                     f'Cosine Similarity: {cosine_similarity_optimized:.4f}')
        
        pos = nx.spring_layout(g, seed=seed)
        nx.draw(g, pos, ax=axes[0], with_labels=True, node_color=[g.nodes[n]['color'] for n in g.nodes])
        axes[0].set_title('Original Graph')
        
        pos_optimized = nx.spring_layout(g_opt, seed=seed)
        nx.draw(
            g_opt, pos, 
            ax=axes[1], 
            with_labels=True, 
            node_color=[g_opt.nodes[n]['color'] for n in g_opt.nodes], 
            edge_cmap=plt.cm.Blues
        )
        axes[1].set_title('Optimized Graph')
        
        fig_path = os.path.join(ARTIFACTS_PATH, 'optimized_graph_comparison.png')
        fig.savefig(fig_path)
        
    def test_decode_color_graph(self):
        
        # Generate the original graph
        colors = ['red', 'green', 'blue']
        g: nx.Graph = generate_random_color_nx(
            num_nodes=4,
            num_edges=5,
            colors=colors,
            seed=42,
        )
        graph = graph_dict_from_color_nx(g)
        
        # Setup the encoder hyper net
        dim = 10_000
        node_encoder_map = make_color_node_encoder_map(dim, colors=colors)
        hyper_net = HyperNet(
            hidden_dim=dim,
            depth=3,
            node_encoder_map=node_encoder_map,
        )
        
        # Initial embedding
        data = next(iter(DataLoader(data_list_from_graph_dicts([graph]), batch_size=1)))
        result = hyper_net.forward(data)
        embedding = result['graph_embedding'].detach()
        
        # decoding
        constraints_order_zero = hyper_net.decode_order_zero(
            embedding=embedding,
        )
        constraints_order_one = hyper_net.decode_order_one(
            embedding=embedding,
            constraints_order_zero=constraints_order_zero,
        )
        print('order zero:')
        pprint(constraints_order_zero)
        print('order one:')
        pprint(constraints_order_one)
        

        