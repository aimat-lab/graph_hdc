"""
Experiment mixin for Asphericity-Weighted Graph Complexity calculation.

This mixin provides functionality to calculate a composite molecular descriptor that
combines 3D shape characteristics with 2D graph-theoretic complexity. The property
is defined as the product of two global molecular descriptors:

    AspherityComplexity = Asphericity × BertzCT

Where:
- **Asphericity**: A 3D descriptor measuring how non-spherical the molecular shape is
- **BertzCT**: The Bertz complexity index, a graph-theoretic measure of molecular complexity

This combined descriptor is specifically designed to test graph neural networks'
ability to capture both geometric (3D) and topological (2D) global features of
molecular structures.

Why this property requires large message passing radius:

1. **3D Asphericity Component**: Asphericity is computed from the eigenvalues of the
   radius of gyration tensor, which requires knowledge of the entire 3D molecular
   geometry. The tensor is calculated as:

   .. math::
      T_{ij} = \\frac{1}{N} \\sum_{k=1}^{N} (r_{ki} - r_{ci})(r_{kj} - r_{cj})

   where r_k are atomic positions and r_c is the molecular center of mass. This
   inherently depends on all atoms in the molecule.

2. **Graph Complexity Component**: The Bertz complexity index (BertzCT) is based on
   the molecular connectivity matrix and considers:
   - Total number of atoms and bonds
   - Distribution of bond types across the molecule
   - Ring systems and branching patterns
   - Overall graph connectivity

   All these factors require aggregating information from the entire molecular graph.

3. **Dual Global Dependencies**: By combining 3D shape and 2D complexity, this
   descriptor requires neural networks to simultaneously:
   - Capture the overall 3D spatial arrangement of atoms
   - Understand the topological complexity of the bonding network
   - Integrate these two distinct types of global information

4. **Non-Local Shape Features**: Asphericity is sensitive to the distribution of
   mass throughout the molecule. Rod-like (linear) molecules have high asphericity,
   while spherical molecules have low asphericity. This cannot be determined from
   local neighborhoods alone.

This property is particularly challenging for graph neural networks because it
requires reasoning about:
- Long-range spatial relationships in 3D (asphericity)
- Global connectivity patterns in the bonding network (complexity)
- The interaction between molecular shape and topological structure

Example values:
- Linear alkane chain: High asphericity (~0.4), medium complexity
- Compact fullerene: Low asphericity (~0.05), very high complexity
- Large dendrimer: Medium asphericity, very high complexity
"""
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import AllChem, Descriptors3D, GraphDescriptors
from pycomex.functional.experiment import Experiment, ExperimentMixin

# Create the mixin instance
mixin = ExperimentMixin(glob=globals())


def calculate_asphericity_complexity(mol: Chem.Mol) -> float:
    """
    Calculate the Asphericity-Weighted Graph Complexity for a molecule.

    This function computes a composite descriptor that combines:
    1. **Asphericity**: A 3D shape descriptor measuring non-sphericity
    2. **BertzCT**: A graph complexity measure based on connectivity

    The calculation process:
    1. Generate a 3D conformer of the molecule using ETKDG algorithm
    2. Compute Asphericity from the radius of gyration tensor eigenvalues
    3. Compute BertzCT from the molecular graph connectivity
    4. Return their product

    **Asphericity** is defined from the radius of gyration tensor eigenvalues
    (λ₁ ≥ λ₂ ≥ λ₃):
        Asphericity = λ₁ - (λ₂ + λ₃)/2

    It ranges from 0 (perfectly spherical) to ~0.5 (extremely rod-like).

    **BertzCT** (Complexity Index) is based on the molecular graph's adjacency
    and distance matrices, considering:
    - Number of atoms and bonds
    - Bond type diversity
    - Connectivity patterns

    The product combines spatial and topological information:
        AspherityComplexity = Asphericity × BertzCT

    Example:
        For n-octane (C8H18):
        - Asphericity ≈ 0.39 (linear, rod-like)
        - BertzCT ≈ 133 (moderately complex)
        - AspherityComplexity ≈ 52

        For adamantane (C10H16):
        - Asphericity ≈ 0.05 (compact, near-spherical)
        - BertzCT ≈ 195 (cage structure, complex)
        - AspherityComplexity ≈ 10

    :param mol: RDKit Mol object representing the molecule.

    :return: The Asphericity-Weighted Graph Complexity as a float value.
             Returns 0.0 if conformer generation fails or calculations error.
    """
    # First, calculate BertzCT (doesn't require 3D)
    try:
        bertz_ct = GraphDescriptors.BertzCT(mol)
    except Exception:
        # If BertzCT calculation fails, return 0
        return 0.0

    # Generate a 3D conformer for asphericity calculation
    # Need to add hydrogens and embed in 3D
    mol_with_h = Chem.AddHs(mol)

    try:
        # Use ETKDG method for conformer generation
        # Returns -1 if embedding fails
        result = AllChem.EmbedMolecule(mol_with_h, randomSeed=42)

        if result == -1:
            # Conformer generation failed, return 0
            return 0.0

        # Optimize the geometry (optional but recommended)
        AllChem.MMFFOptimizeMolecule(mol_with_h)

    except Exception:
        # Handle any embedding errors
        return 0.0

    # Calculate Asphericity (requires 3D coordinates)
    try:
        asphericity = Descriptors3D.Asphericity(mol_with_h)

        # Handle NaN or invalid values
        if np.isnan(asphericity) or np.isinf(asphericity):
            return 0.0

    except Exception:
        # If asphericity calculation fails, return 0
        return 0.0

    # Compute the product
    asphericity_complexity = asphericity * bertz_ct

    return asphericity_complexity


@mixin.hook('get_graph_labels', replace=True, default=False)
def get_graph_labels(e: Experiment,
                     index: int,
                     graph: dict
                     ) -> np.ndarray:
    """
    Generate regression labels based on Asphericity-Weighted Graph Complexity.

    This hook replaces the default label extraction with a custom calculation
    of a composite descriptor that combines 3D molecular shape (asphericity)
    with 2D graph complexity (Bertz index).

    The combined descriptor tests graph neural networks' ability to:
    - **Capture 3D shape**: Asphericity requires understanding the overall
      spatial arrangement of atoms in three dimensions
    - **Measure topological complexity**: BertzCT requires analyzing the entire
      molecular graph's connectivity patterns
    - **Integrate dual representations**: The product combines geometric and
      graph-theoretic information

    This property is particularly valuable for evaluating whether models can:
    1. Learn global 3D features from molecular graphs (which are primarily 2D)
    2. Compute graph-level complexity measures from local neighborhoods
    3. Combine multiple types of global molecular information

    Computational details:
    - 3D conformers are generated using ETKDG algorithm with fixed random seed
    - Geometry is optimized using MMFF force field
    - Asphericity is computed from radius of gyration tensor eigenvalues
    - BertzCT is computed from molecular connectivity matrix

    Example molecules:
        **Linear pentadecane** (C15H32):
        - High asphericity (~0.42): elongated, rod-like shape
        - Medium complexity (~215): unbranched chain
        - Product ≈ 90: high due to elongated shape

        **Buckminsterfullerene** (C60):
        - Low asphericity (~0.02): nearly perfect sphere
        - Very high complexity (~1800): complex cage structure
        - Product ≈ 36: moderate despite high complexity

        **Large dendrimer** (e.g., C100H150N10O20):
        - Medium asphericity (~0.15): somewhat branched 3D structure
        - Very high complexity (~3000+): highly branched
        - Product ≈ 450+: very high due to both factors

    :param e: The experiment instance providing logging and tracking functionality.
    :param index: The index of the current graph in the dataset.
    :param graph: Dictionary containing graph data including 'graph_repr' (SMILES string).

    :return: NumPy array of shape (1,) containing the Asphericity-Weighted
             Graph Complexity value.
    """
    smiles = str(graph['graph_repr'])
    mol = Chem.MolFromSmiles(smiles)

    # Calculate the Asphericity-Weighted Graph Complexity
    asphericity_complexity = calculate_asphericity_complexity(mol)

    # Store the calculated value as the regression target
    graph['graph_labels'] = np.array([asphericity_complexity]).astype(float)
    return graph['graph_labels']
