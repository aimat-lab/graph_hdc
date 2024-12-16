=========================
Computational Experiments
=========================

All modules in this folder are scripts which implement their own computational experiments, which can be
executed to make some calculations and subsequently produce some results.

All the experiments in this folder are implemented with a special library called ``pycomex``. The core
feature of this library is the automatic management of archive folders for each of the experiments. When
an experiment is executed, this library will automatically create a new archive folder within the ``results``
folder. Inside these archive folders all the results and artifacts created by the experiments are
persistently stored. Pycomex offers various other advanced features such as decoupled analysis execution
and experiment inheritance which may or may not be important to fully understand the implementations of
all the experiments in this folder. For more information visit: https://github.com/the16hpythonist/pycomex

===================
List of Experiments
===================

The following list gives a brief overview of the experiments and their purpose:

- ``template_experiment.py``: A template which can be copied to create new ``Experiment`` modules
    - ``template_experiment_sub.py``: A template which implements a generic ``SubExperiment`` and which can
      be copied to create a new modules.