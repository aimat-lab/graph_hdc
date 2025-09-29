import os
import shutil
import pathlib
import logging
import string
import tempfile
import random
import subprocess
from itertools import product
from typing import List, Any, Dict, Tuple, Optional, Callable, Union

import torch
import click
import jinja2 as j2
import numpy as np
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

# == HYPERVECTOR UTILS ==


def generate_hermitian_symmetric_vector(k, rng=None):
    """
    Generate a Hermitian symmetric vector in the Fourier domain of dimension k.
    :param k: The dimensionality of the vector to be generated.
    :returns: A torch.Tensor of shape (k,) representing the Hermitian symmetric vector.
    """
    if rng is None:
        rng = np.random.default_rng()
    # Determine number of unique frequency components
    half_k = k // 2 + 1  # +1 to include the Nyquist frequency if k is even
    # Random complex vector for positive frequencies
    # positive_freqs = np.random.randn(half_k) + 1j * np.random.randn(half_k)
    positive_freqs = rng.normal(size=half_k) + 1j * rng.normal(size=half_k)
    # Enforce real values for DC and Nyquist if they exist (self-conjugate points)
    # positive_freqs[0] = np.random.randn()  # DC component must be real
    positive_freqs[0] = rng.normal(size=1)  # DC component must be real
    if k % 2 == 0:  # Nyquist frequency only exists if k is even
        # positive_freqs[-1] = np.random.randn()
        positive_freqs[-1] = rng.normal(size=1)
    # Create full Hermitian symmetric vector
    full_spectrum = np.zeros(k, dtype=complex)
    full_spectrum[:half_k] = positive_freqs  # Positive frequencies
    full_spectrum[half_k:] = np.conj(positive_freqs[-2:0:-1])  # Negative frequencies
    # Compute the inverse Fourier transform to get the time-domain signal
    time_domain_signal = np.fft.ifft(full_spectrum)
    # Calculate the norm of the time-domain signal
    norm = np.linalg.norm(time_domain_signal)
    # Scale full_spectrum to make the norm of the inverse Fourier transform equal to 1
    full_spectrum = full_spectrum / norm
    return torch.from_numpy(full_spectrum)


class AbstractEncoder:
    """
    Abstract base class for the property encoders. An encoder class is used to encode individual properties 
    of graph elements (nodes, edges, etc.) into a high-dimensional hypervector representation.
    
    Specific subclasses should implement the ``encode`` and ``decode`` methods to encode and decode the 
    property to and from a high-dimensional hypervector representation.
    """
    
    def __init__(self,
                 dim: int,
                 seed: Optional[int] = None,
                 ):
        self.dim = dim
        self.seed = seed
    
    # Turn the input value into a unified data format
    def normalize(self, value: Any) -> Any: 
        """
        This method should be implemented by the specific subclasses and is used to normalize the property 
        value before it is encoded into a hypervector.
        """
        return value
    
    # Turns whatever property into a random tensor
    def encode(self, value: Any) -> torch.Tensor:
        """
        This method takes the property ``value`` and encodes it into a high-dimensional hypervector 
        as a torch.Tensor.
        This method should be implemented by the specific subclasses.
        """
        raise NotImplementedError()
    
    # Turns the random tensor back into whatever the original property was
    def decode(self, hv: torch.Tensor) -> Any:
        """
        This method takes the hypervector ``hv`` and decodes it back into the original property.
        This method should be implemented by the specific subclasses.
        """
        raise NotImplementedError()
    
    # Returns a dictionary representation of the encoder mapping
    def get_encoder_hv_dict(self) -> Dict[Any, torch.Tensor]:
        """
        This method should return a dictionary representation of the encoder mapping where the keys 
        are the properties that are being encoded and the values are hypervector representations 
        that are used to represent the corresponding property values.
        
        Note that the keys of the dict should be values that are returned by the "decode" method and 
        the values should be valid values returned by the "encode" method.
        """
        raise NotImplementedError()
    
    
class ContinuousEncoder(AbstractEncoder):
    """
    Fourier Holographic Reduced Representation (FHRR) encoder for continuous values.
    
    Encodes continuous scalar values into high-dimensional hypervectors using FHRR, which creates
    smooth distributed representations where similar input values produce similar hypervectors.
    The encoding preserves ordinal relationships and enables approximate decoding.
    
    :param dim: Dimensionality of the output hypervector
    :type dim: int
    :param size: Expected range of input values for normalization
    :type size: float  
    :param bandwidth: Controls encoding sensitivity - smaller values create more sensitive encodings
    :type bandwidth: float
    :param seed: Random seed for reproducible encoder generation
    :type seed: Optional[int]
    """
    def __init__(
        self,
        dim: int,
        size: float,
        bandwidth: float,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.size = size
        self.bandwidth = bandwidth
        self.seed = seed
        
        rng = np.random.default_rng(seed + 1) if seed is not None else None
        self.matrix = generate_hermitian_symmetric_vector(self.dim, rng=rng)

    def encode(self, value: Any) -> torch.Tensor:
        """
        Encode continuous value(s) into hypervector(s) using FHRR.
        
        Transforms continuous values into high-dimensional representations by raising
        the Hermitian matrix to powers based on normalized input values, then applying
        inverse FFT to obtain real-valued hypervectors.
        
        :param value: Continuous value(s) to encode, either scalar tensor or 1D batch
        :type value: torch.Tensor
        :return: Hypervector representation(s) of shape (dim,) for scalar or (batch_size, dim) for batch
        :rtype: torch.Tensor
        """
        # Handle scalar and batch inputs
        if value.dim() == 0:
            # Scalar input
            exponent = (value / self.bandwidth).unsqueeze(-1)
            exponents = self.matrix ** exponent
            code = torch.fft.ifft(exponents).real
            return code
        else:
            # Batch input
            batch_size = value.shape[0]
            exponent = value / self.bandwidth
            # Expand matrix for batch processing
            matrix_expanded = self.matrix.unsqueeze(0).expand(batch_size, -1)
            exponent_expanded = exponent.unsqueeze(-1)
            exponents = matrix_expanded ** exponent_expanded
            code = torch.fft.ifft(exponents).real
            return code
        
    def encode_batch(self, values: torch.Tensor) -> torch.Tensor:
        # Scalar input
        exponent = (values / self.bandwidth).unsqueeze(-1)
        exponents = self.matrix ** exponent
        code = torch.fft.ifft(exponents).real
        return code
    
    def decode(self, hv: torch.Tensor) -> Any:
        """
        Approximate decoding of FHRR hypervector back to continuous value.
        
        Attempts to recover the original continuous value from its hypervector representation
        using frequency domain analysis. The method extracts phase information from the FFT
        and estimates the original scaling factor using the Hermitian matrix logarithm.
        
        :param hv: Hypervector to decode of shape (dim,)
        :type hv: torch.Tensor
        :return: Approximated continuous value (note: decoding is inherently lossy)
        :rtype: torch.Tensor
        """
        # Convert to complex for FFT
        hv_complex = hv + 0j
        
        # Take FFT to get frequency domain representation
        freq_domain = torch.fft.fft(hv_complex)
        
        # Extract phase information (angle of complex numbers)
        phases = torch.angle(freq_domain)
        
        # Use the Hermitian matrix to estimate the original exponent
        # This is an approximation based on the logarithm of the frequency domain
        log_matrix = torch.log(self.matrix + 1e-10)  # Add small epsilon for numerical stability
        
        # Estimate the scaling factor from the phase information
        # This is a heuristic approach - may need refinement
        with torch.no_grad():
            estimated_exponent = torch.mean(phases / (log_matrix + 1e-10))
            estimated_value = estimated_exponent * self.bandwidth
            
        return estimated_value.real
    
    
class CategoricalOneHotEncoder(AbstractEncoder):
    """
    This specific encoder is used to encode categorical properties given as a one-hot encoding vector
    into a high-dimensional hypervector. This is done by creating random continuous base vectors for 
    each of the categories and then selecting the corresponding base vector by the index of the one-hot
    encoding. The decoding is done by calculating the base vector with the smallest distance to the
    the given hypervector and returning the corresponding category index.
    
    :param dim: The dimensionality of the hypervectors.
    :param num_categories: The number of categories that can be encoded.
    :param seed: The random seed to use for the generation of the base vectors. Default is None, but 
        can be set for reproducibility.
    """
    
    def __init__(self,
                 dim: int,
                 num_categories: int,
                 seed: Optional[int] = 0,
                 ):
        AbstractEncoder.__init__(self, dim, seed)
        self.num_categories = num_categories
        
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)
        self.dist = torch.distributions.Normal(0.0, 1.0 / np.sqrt(dim), generator=self.gen)
        self.embeddings = self.dist.sample((num_categories, dim))
        #random = np.random.default_rng(seed + 2)
        # self.embeddings: torch.Tensor = torch.tensor(random.normal(
        #     # This scaling is important to have normalized base vectors
        #     loc=0.0,
        #     scale=(1.0 / np.sqrt(dim)), 
        #     size=(num_categories, dim)
        # ).astype(np.float32))
    
    def encode(self, value: Any
               ) -> torch.Tensor:
        
        index = torch.argmax(value)
        return self.embeddings[index]
    
    def decode(self, 
               hv: torch.Tensor, 
               distance: str ='euclidean'
               ) -> Any:
        
        if distance == 'euclidean':
            distances = np.linalg.norm(self.embeddings - hv.numpy(), axis=1)
            
        elif distance == 'cosine':
            similarities = np.dot(self.embeddings, hv.numpy()) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(hv.numpy()))
            distances = 1 - similarities
        
        else:
            raise ValueError(f"Unsupported distance metric: {distance}")
        
        index = np.argmin(distances)
        result = tuple(1 if i == index else 0 for i in range(self.num_categories))
        return result
    
    def get_encoder_hv_dict(self) -> Dict[Any, torch.Tensor]:
        return {
            tuple(1 if i == index else 0 for i in range(self.num_categories)): hv
            for index, hv in enumerate(self.embeddings)
        }

class CategoricalIntegerEncoder(AbstractEncoder):
    
    def __init__(self,
                 dim: int,
                 num_categories: int,
                 seed: Optional[int] = 0,
                 ):
        AbstractEncoder.__init__(self, dim, seed)
        self.num_categories = num_categories
        
        torch.manual_seed(seed)
        self.dist = torch.distributions.Normal(0.0, 1.0 / np.sqrt(dim))
        self.embeddings = self.dist.sample((num_categories, dim))
        # random = np.random.default_rng(seed + 3)
        # self.embeddings: torch.Tensor = torch.tensor(random.normal(
        #     # This scaling is important to have normalized base vectors
        #     loc=0.0,
        #     scale=(1.0 / np.sqrt(dim)), 
        #     size=(num_categories, dim),
        # ).astype(np.float32))
    
    def encode(self, value: Any) -> torch.Tensor:
        return self.embeddings[int(value)]
    
    # def encode_batch(self, values: torch.Tensor) -> torch.Tensor:
    #     return self.embeddings[values.long()]
    
    def decode(self, 
               hv: torch.Tensor, 
               distance: str ='euclidean',
               ) -> Any:
        
        if distance == 'euclidean':
            distances = np.linalg.norm(self.embeddings - hv.numpy(), axis=1)
            
        elif distance == 'cosine':
            similarities = np.dot(self.embeddings, hv.numpy()) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(hv.numpy()))
            distances = 1 - similarities
        
        else:
            raise ValueError(f"Unsupported distance metric: {distance}")
        
        return int(np.argmin(distances))
    
    def get_encoder_hv_dict(self) -> Dict[Any, torch.Tensor]:
        return {
            index: hv
            for index, hv in enumerate(self.embeddings)
        }

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


def shallow_dict_equal(data_1: dict, data_2: dict) -> bool:
    """
    This function is used to compare two dictionaries with shallow values. That is, the values of the
    dictionaries are not dictionaries themselves but simple values. The function will return True if the
    two dictionaries ``data_1`` and ``data_2` have the same keys and the values are equal.
    Otherwise False is returned.
    
    :param data_1: The first dictionary
    :param data_2: The second dictionary
    
    :return: True if the two dictionaries are equal, otherwise False
    """
    # The most rudimentary condition is that both dictionaries have the same keys. If that is not the 
    # case they can already not be equal
    if list(data_1.keys()) != list(data_2.keys()):
        return False
    
    # Now we iterate over all the keys and compare the values. If we find a key where the values are not
    # equal we can return False immediately
    for key in data_1.keys():
        if data_1[key] != data_2[key]:
            return False
        
    return True


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

def latex_table_element_mean(values: List[float],
                             template_name: str = 'table_element_mean.tex.j2',
                             vertical: bool = True,
                             raw: bool = False,
                             ) -> str:
    if raw:
        mean, std = values
    else:
        mean = np.mean(values)
        std = np.std(values)

    template = TEMPLATE_ENV.get_template(template_name)
    return template.render(
        mean=mean,
        std=std,
        vertical=vertical
    )


def latex_table_element_median(values: List[float],
                               upper_quantile: float = 0.75,
                               lower_quantile: float = 0.25,
                               include_variance: bool = True,
                               template_name: str = 'table_element_median.tex.j2') -> str:
    median = np.median(values)
    upper = np.quantile(values, upper_quantile)
    lower = np.quantile(values, lower_quantile)

    template = TEMPLATE_ENV.get_template(template_name)
    return template.render(
        median=median,
        upper=upper,
        lower=lower,
        include_variance=include_variance
    )


def latex_table(column_names: List[str],
                rows: List[Union[List[float], str]],
                content_template_name: str = 'table_content.tex.j2',
                table_template_name: str = 'table.tex.j2',
                list_element_cb: Callable[[List[float]], str] = latex_table_element_mean,
                prefix_lines: List[str] = [],
                caption: str = '',
                ) -> Tuple[str, str]:

    # ~ Pre Processing the row elements into strings
    string_rows = []
    for row_index, row in enumerate(rows):
        string_row = []
        for element in row:
            if isinstance(element, str):
                string_row.append(element)
            if isinstance(element, list) or isinstance(element, np.ndarray):
                string = list_element_cb(element)
                string_row.append(string)

        string_rows.append(string_row)

    alignment = ''.join(['c' for _ in column_names])

    # ~ Rendering the latex template(s)

    content_template = TEMPLATE_ENV.get_template(content_template_name)
    content = content_template.render(rows=string_rows)

    table_template = TEMPLATE_ENV.get_template(table_template_name)
    table = table_template.render(
        alignment=alignment,
        column_names=column_names,
        content=content,
        header='\n'.join(prefix_lines),
        caption=caption,
    )

    return content, table

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

