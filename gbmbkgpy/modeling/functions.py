from collections.abc import Iterable
from copy import deepcopy

import numpy as np

from astromodels import Constant


def eval_func(func, val):
    """
    eval function. needed for vectrorize wrapper
    definition
    """
    return func(val)


vec_eval_func = np.vectorize(eval_func)


def getattr_value_object(ob, name):
    return getattr(ob, name).value


vec_getattr = np.vectorize(getattr_value_object)


class AstromodelFunctionVector:

    name = "AstromodelFunctionVector"

    def __init__(self, num_x, base_function=Constant()):
        """
        Matrix of astromodel functions. all entries must be from
        the same base function and will get all evaluated at the same
        values.
        """
        self._num_x = num_x
        self._base_function = base_function
        self._vec = np.empty(num_x, dtype=type(base_function))

        # fill vector
        for x in range(self._num_x):
            self._vec[x] = deepcopy(base_function)

        # set attributes to access param values
        for name in base_function.parameters.keys():
            setattr(self, name, vec_getattr(self._vec, name))

    def add_function(self, function, idx):
        """
        add the function to a given vector position
        """
        assert isinstance(function, self._base_function)
        self._vec[idx] = function

    def __call__(self, values):
        """
        Evaluate all functions in vector at the given value
        """
        if isinstance(values, Iterable):
            res = np.zeros((*values.shape,
                            *self._vec.shape))
            for x in range(self._num_x):
                res[..., x] = self._vec[x](values)
            return res

        return vec_eval_func(self._vec, values)


    @property
    def free_parameters(self):
        """
        get all the free parameters in one dict
        """
        params = {}
        for x in range(self._num_x):
            for name, param in self.vector[x].free_parameters.items():
                params[f"{name}_{x}"] = param
        return params

    @property
    def vector(self):
        return self._vec

    @property
    def num_x(self):
        return self._num_x
