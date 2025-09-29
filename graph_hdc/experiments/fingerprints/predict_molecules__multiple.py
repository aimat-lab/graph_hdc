from typing import List
from rich.pretty import pprint
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
import statistics
import numpy as np


BASE_EXPERIMENT = 'predict_molecules__hdc__aqsoldb.py'

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MODELS = ['neural_net']

__PREFIX__ = BASE_EXPERIMENT.replace('.py', '')

experiment = Experiment(
    folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment
def experiment(e: Experiment):
    
    e.log('starting meta experiment...')

    metrics_list: List[dict] = []
    for seed in e.SEEDS:
        
        e.log(f' * Running experiment for SEED={seed}...')
        exp = Experiment.extend(
            BASE_EXPERIMENT,
            base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals()
        )
        exp.__DEBUG__ = True
        exp.__PREFIX__ = '_multiple'
        exp.MODELS = e.MODELS
        exp.SEED = seed
        exp.run()
        
        pprint(exp['metrics'])
        e.track_many(exp['metrics'])
        
        metrics_list.append(exp['metrics'])
        
    e.log('Finished experiments, aggregating statistics...')
    metrics_keys = list(metrics_list[0].keys())
    for key1 in metrics_keys:
        
        e.log(f' * {key1}')
        for key2 in metrics_list[0][key1].keys():
            
            values = [m[key1][key2] for m in metrics_list]
            mean = np.mean(values)
            stdev = np.std(values) if len(values) > 1 else 0.0
            e.log(f'   - {key2}: {mean:.4f} ± {stdev:.4f} (n={len(values)})')
    
        
        
experiment.run_if_main()