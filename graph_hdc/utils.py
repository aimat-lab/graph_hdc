import os
import shutil
import pathlib
import logging
import string
import tempfile
import random
import subprocess
from itertools import product
from typing import List, Any, Dict, Tuple, Optional, Callable

import torch
import click
import jinja2 as j2
import networkx as nx
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy


PATH = pathlib.Path(__file__).parent.absolute()
VERSION_PATH = os.path.join(PATH, 'VERSION')
EXPERIMENTS_PATH = os.path.join(PATH, 'experiments')
TEMPLATES_PATH = os.path.join(PATH, 'templates')

# Use this jinja2 environment to conveniently load the jinja templates which are defined as files within the
# "templates" folder of the package!
TEMPLATE_ENV = j2.Environment(
    loader=j2.FileSystemLoader(TEMPLATES_PATH),
    autoescape=j2.select_autoescape(),
)
TEMPLATE_ENV.globals.update(**{
    'zip': zip,
    'enumerate': enumerate
})

# This logger can be conveniently used as the default argument for any function which optionally accepts
# a logger. This logger will simply delete all the messages passed to it.
NULL_LOGGER = logging.Logger('NULL')
NULL_LOGGER.addHandler(logging.NullHandler())


class HypervectorCombinations:
    
    def __init__(self, 
                 value_hv_dicts: Dict[str, Dict[Any, torch.Tensor]],
                 bind_fn: Optional[Callable] = None,
                 ):
        self.value_hv_dicts = value_hv_dicts
        self.bind_fn = bind_fn
        
        self.combinations: Dict[tuple] = {}
        
        key_tuples_list: List[List[Tuple[str, Any]]] = []
        for name, value_hv_dict in value_hv_dicts.items():
            key_tuples: List[Tuple[str, Any]] = [(name, key) for key, _ in value_hv_dict.items()]
            key_tuples_list.append(key_tuples)
            
        key_tuple_combinations = list(product(*key_tuples_list))
        for comb in key_tuple_combinations:
            # hvs: (num_dicts, dim)
            hvs: torch.Tensor = torch.stack([self.value_hv_dicts[name][key] for (name, key) in comb], dim=0)
            # value: (num_dicts, dim)
            value: torch.Tensor = torch_pairwise_reduce(hvs, func=self.bind_fn)
            value = value.squeeze()
            
            comb_key = tuple(sorted(comb))
            self.combinations[comb_key] = value

    def get(self, query: dict) -> torch.Tensor:
        comb_key = tuple(sorted(query.items()))
        return self.combinations[comb_key]
    
    def __iter__(self):
        self.__generator__ = self.__generate__()
        return self.__generator__
            
    def __generate__(self):
        for comb_key, value in self.combinations.items():
            comb_dict = {name: key for (name, key) in comb_key}
            yield comb_dict, value

# == CLI RELATED ==

def get_version():
    """
    Returns the version of the software, as dictated by the "VERSION" file of the package.
    """
    with open(VERSION_PATH) as file:
        content = file.read()
        return content.replace(' ', '').replace('\n', '')


# https://click.palletsprojects.com/en/8.1.x/api/#click.ParamType
class CsvString(click.ParamType):

    name = 'csv_string'

    def convert(self, value, param, ctx) -> List[str]:
        if isinstance(value, list):
            return value

        else:
            return value.split(',')

# == NETWORKX UTILS ==

def nx_random_uniform_edge_weight(g: nx.Graph,
                                  lo: float = 0.0,
                                  hi: float = 1.0,
                                  ) -> nx.Graph:
    
    for u, v in g.edges():
        g[u][v]['edge_weight'] = random.uniform(lo, hi)
        
    return g


# == TORCH UTILS ==

def torch_pairwise_reduce(tens: torch.Tensor, 
                          func: callable, 
                          dim: int = 0
                          ) -> torch.Tensor:
    """
    Given a tensor ``tens`` with shape (M, N) this function will reduce the tensor along the dimension ``dim``
    (e.g. dimension 0) using the function ``func``. The function ``func`` must be a callable which accepts two 
    tensors of shape (N,) and returns a tensor of shape (N,). The result of the reduction will be a tensor of
    shape (N,).
    
    :param tens: The input tensor
    :param func: The reduction function which is a callable that maps 
        (torch.Tensor, torch.Tensor) -> torch.Tensor
    :param dim: The dimension along which the reduction should be performed. default 0
    
    :return: The reduced tensor
    """
    result = torch.index_select(tens, dim, torch.tensor(0))
    
    for i in range(1, tens.size(dim)):
        result = func(result, torch.index_select(tens, dim, torch.tensor(i)))
    
    result = torch.squeeze(result, dim=dim)
    return result


# == JSON PICKLE ==
# The "jsonpickle" library provides the utility to save and load custom objects to and from the human-readable
# json format. This section contains the additional utiltiy functions related to this saving/loading 
# functionality.

jsonpickle_numpy.register_handlers()


@jsonpickle.handlers.register(torch.Tensor, base=True)
class TorchTensorHandler(jsonpickle_numpy.NumpyNDArrayHandler):
    
    def flatten(self, obj: Any, data: dict):
        array = obj.detach().cpu().numpy()
        return super().flatten(array, data)
    
    def restore(self, data: dict):
        array = super().restore(data)
        return torch.tensor(array)



# == STRING UTILITY ==
# These are some helper functions for some common string related problems

def random_string(length: int,
                  chars: List[str] = string.ascii_letters + string.digits
                  ) -> str:
    """
    Generates a random string with ``length`` characters, which may consist of any upper and lower case
    latin characters and any digit.

    The random string will not contain any special characters and no whitespaces etc.

    :param length: How many characters the random string should have
    :param chars: A list of all characters which may be part of the random string
    :return:
    """
    return ''.join(random.choices(chars, k=length))


# == LATEX UTILITY ==
# These functions are meant to provide a starting point for custom latex rendering. That is rendering latex
# from python strings, which were (most likely) dynamically generated based on some kind of experiment data

def render_latex(kwargs: dict,
                 output_path: str,
                 template_name: str = 'article.tex.j2'
                 ) -> None:
    """
    Renders a latex template into a PDF file. The latex template to be rendered must be a valid jinja2
    template file within the "templates" folder of the package and is identified by the string file name
    `template_name`. The argument `kwargs` is a dictionary which will be passed to that template during the
    rendering process. The designated output path of the PDF is to be given as the string absolute path
    `output_path`.

    **Example**

    The default template for this function is "article.tex.j2" which defines all the necessary boilerplate
    for an article class document. It accepts only the "content" kwargs element which is a string that is
    used as the body of the latex document.

    .. code-block:: python

        import os
        output_path = os.path.join(os.getcwd(), "out.pdf")
        kwargs = {"content": "$\text{I am a math string! } \pi = 3.141$"
        render_latex(kwargs, output_path)

    :raises ChildProcessError: if there was ANY problem with the "pdflatex" command which is used in the
        background to actually render the latex

    :param kwargs:
    :param output_path:
    :param template_name:
    :return:
    """
    with tempfile.TemporaryDirectory() as temp_path:
        # First of all we need to create the latex file on which we can then later invoke "pdflatex"
        template = TEMPLATE_ENV.get_template(template_name)
        latex_string = template.render(**kwargs)
        latex_file_path = os.path.join(temp_path, 'main.tex')
        with open(latex_file_path, mode='w') as file:
            file.write(latex_string)

        # Now we invoke the system "pdflatex" command
        command = (f'pdflatex  '
                   f'-interaction=nonstopmode '
                   f'-output-format=pdf '
                   f'-output-directory={temp_path} '
                   f'{latex_file_path} ')
        proc = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise ChildProcessError(f'pdflatex command failed! Maybe pdflatex is not properly installed on '
                                    f'the system? Error: {proc.stdout.decode()}')

        # Now finally we copy the pdf file - currently in the temp folder - to the final destination
        pdf_file_path = os.path.join(temp_path, 'main.pdf')
        shutil.copy(pdf_file_path, output_path)

