#!/usr/bin/env python3
"""Debug script to understand the inf distance issue."""

import torch
from rdkit import Chem

from graph_hdc.models import CompositeHyperNet, HyperNetEnsemble
from graph_hdc.special.molecules import graph_dict_from_mol, make_molecule_node_encoder_map
from graph_hdc.reconstruct import GraphReconstructorAStar

# Recreate test setup
hidden_dim = 5_000
smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'

# Create encoder
node_encoder_map = make_molecule_node_encoder_map(dim=hidden_dim, seed=41)
hyper_net1 = CompositeHyperNet(
    hidden_dim=hidden_dim,
    depth=4,
    node_encoder_map=node_encoder_map,
    bidirectional=True,
)

node_encoder_map = make_molecule_node_encoder_map(dim=hidden_dim, seed=42)
hyper_net2 = CompositeHyperNet(
    hidden_dim=hidden_dim,
    depth=3,
    node_encoder_map=node_encoder_map,
    bidirectional=True,
)

hyper_net = HyperNetEnsemble([hyper_net1, hyper_net2])

# Encode molecule
graph = graph_dict_from_mol(Chem.MolFromSmiles(smiles))
results = hyper_net.forward_graphs([graph])
result = results[0]

print("=== Forward pass result ===")
print(f"Keys: {result.keys()}")
print(f"graph_embedding shape: {result['graph_embedding'].shape}")
if 'graph_hv_stack' in result:
    print(f"graph_hv_stack shape: {result['graph_hv_stack'].shape}")
else:
    print("graph_hv_stack: NOT PRESENT")

graph_embedding = torch.tensor(result['graph_embedding'])
print(f"\ngraph_embedding tensor shape: {graph_embedding.shape}")
print(f"Contains NaN: {torch.isnan(graph_embedding).any()}")
print(f"Contains Inf: {torch.isinf(graph_embedding).any()}")

# Create A* reconstructor
reconstructor = GraphReconstructorAStar(
    encoder=hyper_net,
    encoder_sim=hyper_net,
    memory_budget=8000,
    time_budget=30.0,
    batch_size=200,
)

# Test initial graph creation and encoding
print("\n=== Testing initial graph creation ===")
node_constraints = hyper_net.decode_order_zero(embedding=graph_embedding)
print(f"Node constraints: {len(node_constraints)}")

node_alphabet = []
for constraint in node_constraints:
    for _ in range(constraint['num']):
        node_alphabet.append(constraint['src'].copy())

print(f"Node alphabet length: {len(node_alphabet)}")

# Create initial graphs
initial_graphs = reconstructor._create_initial_graphs(node_alphabet, limit=min(10, len(node_alphabet)))
print(f"Initial graphs created: {len(initial_graphs)}")
print(f"First graph: {initial_graphs[0]}")

# Try encoding them
print("\n=== Testing batch encoding ===")
try:
    initial_results = reconstructor._batch_encode_graphs(initial_graphs)
    print(f"Initial results: {len(initial_results)}")

    if len(initial_results) == 0:
        print("\nERROR: All initial graphs were filtered out!")
        print("Testing individual model forward pass on first graph...")

        # Test individual forward
        test_results = hyper_net.forward_graphs([initial_graphs[0]])
        print(f"Forward graphs returned: {len(test_results)} results")
        if test_results:
            test_result = test_results[0]
            print(f"Keys: {test_result.keys()}")
            if 'graph_embedding' in test_result:
                emb = test_result['graph_embedding']
                print(f"graph_embedding shape: {emb.shape}")
                print(f"Contains NaN: {torch.isnan(torch.tensor(emb)).any()}")
                print(f"Contains Inf: {torch.isinf(torch.tensor(emb)).any()}")
            if 'graph_hv_stack' in test_result:
                hv = test_result['graph_hv_stack']
                print(f"graph_hv_stack shape: {hv.shape}")
                print(f"Contains NaN: {torch.isnan(torch.tensor(hv)).any()}")
                print(f"Contains Inf: {torch.isinf(torch.tensor(hv)).any()}")

except Exception as e:
    print(f"ERROR during batch encoding: {e}")
    import traceback
    traceback.print_exc()

# Test extract_distance_embedding
print("\n=== Testing extract_distance_embedding ===")
try:
    dist_emb = hyper_net.extract_distance_embedding(graph_embedding)
    print(f"Distance embedding shape: {dist_emb.shape}")
    print(f"Contains NaN: {torch.isnan(dist_emb).any()}")
    print(f"Contains Inf: {torch.isinf(dist_emb).any()}")
except Exception as e:
    print(f"ERROR during extract_distance_embedding: {e}")
    import traceback
    traceback.print_exc()
