#!/bin/bash
# Login-node dataset pre-fetch for Experiment 13. JUPITER Booster compute nodes have no internet,
# so every dataset must be downloaded from the fileshare into the (project) chem_mat_data cache
# BEFORE any compute job runs. Run this on the JUPITER login node.
set -u
jutil env activate -p aimatchem
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
cd /e/project1/aimatchem/teufel1/graph_hdc
source .venv/bin/activate

# Unique DATASET_NAMEs behind the 19 targets (clogp reuses aqsoldb's molecules).
for name in aqsoldb freesolv lipophilicity bace_reg hopv15_exp compas_3x qm9_smiles; do
    echo "=== $name ==="
    python -c "from chem_mat_data.main import load_graph_dataset; g=load_graph_dataset('$name', folder_path='/tmp'); print('  ok:', '$name', len(g), 'graphs')" 2>&1 | tail -3
done
echo "DOWNLOAD_DONE"
