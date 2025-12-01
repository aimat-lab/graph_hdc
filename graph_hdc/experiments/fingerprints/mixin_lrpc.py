"""
Experiment mixin for Long-Range Polar Coupling Index (LRPC) calculation.

This mixin provides functionality to calculate the Long-Range Polar Coupling Index,
a molecular descriptor that measures interactions between polar functional groups
separated by significant distances in the molecular graph. This property is
specifically designed to test graph neural networks' ability to capture long-range
electronic and electrostatic interactions.

The LRPC is calculated by:
1. Identifying all polar atoms (N, O, F, P, S, Cl, Br, I)
2. Computing partial charges using the Gasteiger method
3. For each pair of polar atoms separated by >6 bonds:
   - Computing charge coupling (difference in partial charges)
   - Applying a distance-dependent factor: 1 / (1 + distance)
   - Adding a bonus if atoms are connected through a conjugated system
4. Summing all contributions

Why this property requires large message passing radius:

1. **Long-Range Dependencies**: By design, LRPC only considers polar atom pairs
   separated by more than 6 bonds. This forces neural networks to aggregate
   information across large neighborhoods to capture these distant interactions.

2. **Electrostatic Coupling**: The partial charge distribution across a molecule
   is influenced by the entire electronic structure, not just local bonding.
   Accurate charge prediction requires global molecular context.

3. **Conjugated Pathways**: The presence of conjugated systems connecting distant
   polar groups significantly enhances electronic coupling. Detecting these
   pathways requires tracing connections across multiple bonds through the
   molecular graph.

4. **Non-Local Electronic Effects**: Polar groups can influence each other through
   inductive effects, resonance, and through-space interactions that extend
   beyond immediate neighbors. These effects require neural networks to integrate
   information from distant parts of the molecule.

The LRPC is a synthetic descriptor designed to challenge models' ability to learn
global molecular properties that depend on both geometric (distance) and electronic
(charge, conjugation) factors distributed across the entire molecular structure.
"""
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import rdmolops, AllChem
from collections import deque
from pycomex.functional.experiment import Experiment, ExperimentMixin

# Create the mixin instance
mixin = ExperimentMixin(glob=globals())


def check_conjugated_path(mol: Chem.Mol,
                          atom_i: int,
                          atom_j: int
                          ) -> bool:
    """
    Check if two atoms are connected through a conjugated bond pathway.

    This function uses breadth-first search to determine whether there exists
    a path between two atoms that consists entirely of conjugated bonds. Such
    paths facilitate electron delocalization and enhance electronic coupling
    between distant polar groups.

    The algorithm explores the molecular graph starting from atom_i, traversing
    only conjugated bonds, and checks if atom_j can be reached.

    :param mol: RDKit Mol object representing the molecule.
    :param atom_i: Index of the first atom.
    :param atom_j: Index of the second atom.

    :return: True if atoms are connected through conjugated bonds, False otherwise.
    """
    # BFS to find if there's a conjugated path between atom_i and atom_j
    visited = set()
    queue = deque([atom_i])
    visited.add(atom_i)

    while queue:
        current_atom_idx = queue.popleft()
        current_atom = mol.GetAtomWithIdx(current_atom_idx)

        # Check all bonds of the current atom
        for bond in current_atom.GetBonds():
            if not bond.GetIsConjugated():
                continue

            # Get the neighbor atom
            neighbor_idx = bond.GetOtherAtomIdx(current_atom_idx)

            if neighbor_idx == atom_j:
                return True

            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                queue.append(neighbor_idx)

    return False


def calculate_lrpc(mol: Chem.Mol) -> float:
    """
    Calculate the Long-Range Polar Coupling Index for a molecule.

    This function computes a descriptor that quantifies the interaction between
    polar groups separated by significant distances in the molecular graph. The
    calculation combines several factors:

    1. **Polar atom identification**: Atoms with heteroatoms (N, O, F, P, S, halogens)
    2. **Partial charges**: Gasteiger method for computing atomic partial charges
    3. **Distance filtering**: Only considers atom pairs >6 bonds apart
    4. **Coupling strength**: Based on charge difference and inverse distance
    5. **Conjugation bonus**: Enhanced coupling through conjugated pathways

    The mathematical formulation is:
        LRPC = Σ(i,j: d(i,j)>6) |q(i)-q(j)| × [1/(1+d(i,j))] × (1 + 0.5×conj(i,j))

    where:
        - q(i), q(j) are partial charges
        - d(i,j) is the bond distance between atoms
        - conj(i,j) is 1 if connected through conjugated path, 0 otherwise

    Example:
        For a molecule with two polar groups (N and O) separated by 8 bonds,
        with charges -0.4 and -0.2, not conjugated:
        contribution = |(-0.4)-(-0.2)| × 1/(1+8) × (1 + 0) = 0.2 × 0.111 = 0.022

    :param mol: RDKit Mol object representing the molecule.

    :return: The Long-Range Polar Coupling Index as a float value.
    """
    # Identify all polar atoms (N, O, F, P, S, Cl, Br, I)
    polar_atom_numbers = {7, 8, 9, 15, 16, 17, 35, 53}
    polar_atoms = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in polar_atom_numbers:
            polar_atoms.append(atom.GetIdx())

    # Need at least 2 polar atoms for coupling
    if len(polar_atoms) < 2:
        return 0.0

    # Compute Gasteiger charges
    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            # Handle cases where charge computation fails
            charge = atom.GetDoubleProp('_GasteigerCharge')
            if np.isnan(charge) or np.isinf(charge):
                charge = 0.0
            charges.append(charge)
    except Exception:
        # If charge calculation fails, return 0
        return 0.0

    # Get distance matrix
    distance_matrix = rdmolops.GetDistanceMatrix(mol)

    # Calculate LRPC by summing contributions from distant polar pairs
    lrpc = 0.0

    for i in polar_atoms:
        for j in polar_atoms:
            if i < j:  # Avoid double counting
                distance = distance_matrix[i, j]

                # Only consider pairs separated by >6 bonds
                if distance > 6:
                    # Charge coupling: absolute difference in partial charges
                    charge_coupling = abs(charges[i] - charges[j])

                    # Distance factor: inversely proportional to distance
                    distance_factor = 1.0 / (1.0 + distance)

                    # Check for conjugated pathway
                    has_conjugated_path = check_conjugated_path(mol, i, j)
                    conjugation_bonus = 0.5 if has_conjugated_path else 0.0

                    # Compute contribution
                    contribution = (charge_coupling *
                                  distance_factor *
                                  (1.0 + conjugation_bonus))

                    lrpc += contribution

    return lrpc


@mixin.hook('get_graph_labels', replace=True, default=False)
def get_graph_labels(e: Experiment,
                     index: int,
                     graph: dict
                     ) -> np.ndarray:
    """
    Generate regression labels based on the Long-Range Polar Coupling Index.

    This hook replaces the default label extraction with a custom calculation
    of the LRPC descriptor, which quantifies interactions between distant polar
    groups in molecular structures.

    The LRPC is designed to test graph neural networks' ability to:
    - Capture long-range dependencies (>6 bonds)
    - Model electrostatic interactions between charged/polar groups
    - Recognize conjugated pathways that enhance electronic coupling
    - Integrate geometric and electronic information across the molecule

    This property is particularly challenging because it requires the network to:
    1. Identify polar functional groups
    2. Estimate partial charge distributions
    3. Evaluate distances in the molecular graph
    4. Detect conjugated systems
    5. Combine all these factors into a single descriptor

    Example:
        For a molecule with formula C10H12N2O2 containing two nitro groups
        separated by an aromatic ring system:
        - The aromatic system provides a conjugated pathway
        - The nitro groups carry significant partial charges
        - Distance is typically 7-10 bonds
        - LRPC would capture this long-range electronic coupling

    :param e: The experiment instance providing logging and tracking functionality.
    :param index: The index of the current graph in the dataset.
    :param graph: Dictionary containing graph data including 'graph_repr' (SMILES string).

    :return: NumPy array of shape (1,) containing the LRPC value.
    """
    smiles = str(graph['graph_repr'])
    mol = Chem.MolFromSmiles(smiles)

    # Calculate the Long-Range Polar Coupling Index
    lrpc_value = calculate_lrpc(mol)

    # Store the calculated value as the regression target
    graph['graph_labels'] = np.array([lrpc_value]).astype(float)
    return graph['graph_labels']
