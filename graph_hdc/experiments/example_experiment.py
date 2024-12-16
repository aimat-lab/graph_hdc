"""
This module provides an example of how to define a pycomex experiment module.
"""
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# :param PATH:
#       This is the absolute string path of the experiment folder
PATH: str = pathlib.Path(__file__).parent.absolute()

# :param NUM_ITERATIONS:
#       The number of iterations to repeat in this experiment.
NUM_ITERATIONS: int = 1000

__DEBUG__: bool = True


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)
def experiment(e: Experiment):
    e.log('starting experiment...')

    for i in range(e.NUM_ITERATIONS):
        
        # Here we can provide a default implementation for this hook
        
        # :hook step_function:
        #       This is the function to use in the final
        e.apply_hook(
            'step_function', 
            parameter=1
        )
        
    e.log('experiment done!')
        


@experiment.analysis
def analysis(e: Experiment):
    e.log('starting analysis...')


experiment.run_if_main()