import collections
from gbmbkgpy.modeling.parameter import Parameter


class Function(object):


    def __init__(self, *parameters):

        parameter_dict = collections.OrderedDict()

        for parameter in parameters:

            parameter_dict[parameter.name] = parameter


        self._parameter_dict = parameter_dict

        for key, value in self._parameter_dict.iteritems():
            self.__dict__[key] = value


    # def __setattr__(self, name, value):
    #     raise Exception("It is read only!")
    #


    @property
    def parameter_values(self):

        return [par.value for par in self._parameter_dict.itervalues()]

    def __call__(self, x):

        return self._evaluate(x, *self.parameter_values)



    def _evaluate(self):
        pass


    @property
    def parameters(self):

        return self._parameter_dict


class ContinuumFunction(Function):
    def __init__(self, coefficient_name):
        """
        A continuum function that is parametrized by a constant multiplied by
        a an interpolated function

        :param coefficient_name: the name of the coefficient
        """

        assert isinstance(coefficient_name, str)

        # build the constant

        K = Parameter(coefficient_name, initial_value=1., min_value=0, max_value=1, delta=0.1, normalization=True)

        super(ContinuumFunction, self).__init__(K)

    def set_interpolated_function(self, interpolation):
        """
        Set the temporal interpolation that will be used for the function


        :param interpolation: a scipy interpolation function
        :return: 
        """

        self._interpolation = interpolation

    def _evaluate(self, x, K):
        K * self._interpolation(x)