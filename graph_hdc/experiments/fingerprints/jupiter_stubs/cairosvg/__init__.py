"""
Stand-in for cairosvg on the JUPITER Booster compute nodes, which do not have the system library libcairo that cairosvg
loads when it is imported (the login nodes have it; the only cairo module on JUPITER belongs to Stages/2026, whose
Python 3.13 chem_mat_database does not support). chem_mat_data imports cairosvg at import time for its molecule
drawings (chem_mat_data.visualization), which the fingerprint experiments never call. run_ex27.sbatch puts this folder
on PYTHONPATH only on nodes without libcairo; calling the drawing function then raises instead of drawing.
"""


def svg2png(*args, **kwargs):
    raise RuntimeError('cairosvg is not available on this node (no libcairo): molecule drawings are disabled')
