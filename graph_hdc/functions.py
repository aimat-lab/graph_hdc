from typing import Callable, Dict, Optional, Union


# == REGISTRY ==

FUNCTION_REGISTRY: Dict[str, Callable] = {}

def register_function(name: str, 
                      description: Optional[str] = None,
                      ) -> Callable:
    """
    This decorator function can be used to register a function.
    
    **What is the purpose of registering?**
    
    Some models might provide the functionality that a function/callable object can be passed as an argument to 
    the constructor for example. In this case, passing a normal function would work in the moment but to properly 
    save and load the model to and from the disk, the function needs to be registered. This is because the function
    is not serializable and therefore cannot be saved to the disk. By registering the function, the function can
    be saved as a string identifier and later be resolved to the original function when loading the model.
    
    **Example**
    
    This function can be used as a decorator to register a binding function. The name of the function
    will be used as the identifier for the binding function.
    
    ```python
    
    @register_function('multiply_tensors')
    def example_bind(tens1: torch.Tensor, tens2: torch.Tensor) -> torch.Tensor:
        return tens1 * tens2
    
    ```
    
    :param name: The name of the binding function
    :param binding_type: The type of the binding function, either 'bind' or 'unbind'
    
    :return: The decorator function
    """
    def decorator(fn: Callable) -> Callable:
        FUNCTION_REGISTRY[name] = fn
        setattr(fn, f'__function_name__', name)
        setattr(fn, f'__function_desc__', description)
        return fn
    
    return decorator


def resolve_function(value: Union[str, Callable]
                     ) -> Callable:
    """
    Given either a registered string identifier of a registered function or the function directly 
    as the ``value`` parameter, this function makes sure to return the corresponding callable object.
    If ``value`` is already a callable object, it will be returned as is. If it is a string, the
    corresponding function will be retrieved from the ``BINDING_REGISTRY``.
    
    :param value: The string identifier or the function
    
    :returns: The function callable object
    """
    if isinstance(value, str):
        return FUNCTION_REGISTRY[value]
    else:
        return value
    

def desolve_function(value: Callable
                     ) -> str:
    """
    Given a registered binding function, this function will return the string identifier of that function.
    
    :param value: The binding function
    
    :returns: The string identifier of the binding function
    """
    assert hasattr(value, '__function_name__'), (
        'The given function is not a registered  function! It can therefore not '
        'be exported as a registered string name.'
    )
    
    return getattr(value, '__function_name__')