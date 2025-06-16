|made-with-python| |python-version| |version|

.. |made-with-python| image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
   :target: https://www.python.org/
.. |python-version| image:: https://img.shields.io/badge/Python-3.8.0-green.svg
   :target: https://www.python.org/
.. |version| image:: https://img.shields.io/badge/version-0.1.0-orange.svg
   :target: https://www.python.org/

=============================
Graph Hyperdimensional Coding
=============================

``graph_hdc`` implements **hyperdimensional computing** for graph structures.
Graphs are encoded using a message passing network that produces high dimensional
hypervectors which can be decoded again to recover structural information.  The
package comes with utilities for working with PyTorch Geometric ``Data``
objects, specialised encoders for node attributes and a small experiment suite
based on PyComex.

-----------------------
Installation from source
-----------------------

.. code-block:: console

   git clone https://github.com/the16thpythonist/graph_hdc
   cd graph_hdc
   python3 -m pip install .

After installation the command line interface exposes the package version and
the available experiments::

   python3 -m graph_hdc.cli --version
   python3 -m graph_hdc.cli exp list

---------
Quickstart
---------

The ``HyperNet`` encoder maps graphs to hypervectors.  The example below encodes
and decodes a small colour graph.

.. code-block:: python

   from graph_hdc.special.colors import generate_random_color_nx
   from graph_hdc.special.colors import graph_dict_from_color_nx, make_color_node_encoder_map
   from graph_hdc.models import HyperNet
   from graph_hdc.graph import data_list_from_graph_dicts

   # create a toy graph and convert it to the internal dict format
   g = generate_random_color_nx(num_nodes=4, num_edges=5)
   graph = graph_dict_from_color_nx(g)

   # build the encoder and obtain the embedding
   dim = 1000
   net = HyperNet(hidden_dim=dim, depth=2,
                  node_encoder_map=make_color_node_encoder_map(dim))
   result = net.forward_graphs([graph])[0]
   embedding = result['graph_embedding']
   print(embedding.shape)

   # decode which node types were present
   constraints = net.decode_order_zero(embedding)
   print(constraints)

-----------------------
Computational experiments
-----------------------

Experiments implemented with PyComex can be listed and executed via the CLI::

   python3 -m graph_hdc.cli exp list
   python3 -m graph_hdc.cli exp run predict_molecules__hdc

The ``experiments`` folder contains setups for comparing HDC encodings with
traditional molecular fingerprints and for first steps in graph generation.

-------
Credits
-------

The project depends heavily on `PyComex`_ for experiment management and
``torch_geometric`` for graph operations.

.. _PyComex: https://github.com/the16thpythonist/pycomex.git
