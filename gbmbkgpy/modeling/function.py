import collections


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
