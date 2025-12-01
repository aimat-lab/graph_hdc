from pycomex import Experiment, folder_path, file_namespace





__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment
def experiment(e: Experiment):
    e.log('starting experiment...')
    e.log_parameters()
    
    
experiment.run_if_main()