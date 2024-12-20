import numpy as np

from graph_hdc.functions import FUNCTION_REGISTRY
from graph_hdc.functions import register_function
from graph_hdc.functions import resolve_function, desolve_function


class TestFunctionRegistry:
    
    def test_register_function_basically_works(self):
        
        assert 'multiply' not in FUNCTION_REGISTRY
        
        @register_function('multiply')
        def multiply(a, b):
            return a * b
        
        # Only after defining the function, this should be part of the registry!
        assert 'multiply' in FUNCTION_REGISTRY
        
        # Since it is registered now, the desolving and resolving should also work!
        assert resolve_function('multiply') == multiply
        assert desolve_function(multiply) == 'multiply'
        
        