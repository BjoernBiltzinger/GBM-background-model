prior_parameter_needed = {'uniform': ['min_value', 'max_value'], 'log_uniform': ['min_value', 'max_value'],
                          'gaussian': ['mu', 'sigma'], 'truncated_gaussian': ['mu', 'sigma', 'min_value', 'max_value'],
                          'log_normal': ['mu', 'sigma']}


class Parameter(object):
    def __init__(self, name, initial_value=None, delta=None, min_value=None, max_value=None, mu=None, sigma=None,
                 prior='log_uniform', **kwargs):
        self._name = str(name)
        self._value = initial_value
        self._min_value = min_value
        self._max_value = max_value
        self._mu = mu
        self._sigma = sigma
        self._delta = delta
        self._prior = prior
        assert prior in prior_parameter_needed, 'Unknown prior please use one of these: ' \
                                                '{}'.format(prior_parameter_needed.keys())

        self._free = True

        self._normalization = False

        for k, v in kwargs.iteritems():

            if (k.lower() == 'normalization'):
                self._normalization = bool(v)
            elif (k.lower() == 'fixed'):
                self._free = not bool(v)

    def __eq__(self, value):

        print('here')
        self._set_value(value)

    def __repr__(self):
        if (self._free):
            ff = "free"
        else:
            ff = "fixed"

        if self._min_value == None:
            min_value = "None"
        else:
            min_value = "%g" % self._min_value

        if self._max_value == None:
            max_value = "None"
        else:
            max_value = "%g" % self._max_value

        return "%20s: %10g %10s %10s %10g %s" % (
            self._name, self._value, min_value, max_value, self._delta, ff)

    def _get_value(self):

        return self._value

    def _set_value(self, value):
        self._value = float(value)

        if (abs(self._delta) > 0.2 * abs(self._value)):
            # Fix the delta to be less than 50% of the value
            self._delta = 0.2 * self._value

    value = property(_get_value, _set_value,
                     doc="")

    def _set_delta(self, delta):
        self._delta = delta

    # Define property "free"

    def _set_free(self, value=True):
        self._free = value

    def _get_free(self):
        return self._free

    free = property(_get_free, _set_free,
                    doc="Gets or sets whether the parameter is free or not. Use booleans, like: 'p.free = True' "
                        " or 'p.free = False'. ")

    # Define property "fix"

    def _set_fix(self, value=True):

        self._free = (not value)

    def _get_fix(self):

        return not self._free

    fix = property(_get_fix, _set_fix,
                   doc="Gets or sets whether the parameter is fixed or not. Use booleans, like: 'p.fix = True' "
                       " or 'p.fix = False'. ")

    def _set_bounds(self, bounds):
        """Sets the boundaries for this parameter to min_value and max_value"""

        # Use the properties so that the checks and the handling of units are made automatically

        min_value, max_value = bounds

        # Remove old boundaries to avoid problems with the new one, if the current value was within the old boundaries
        # but is not within the new ones (it will then be adjusted automatically later)
        self._min_value = None
        self._max_value = None

        self._min_value = min_value

        self._max_value = max_value

    def _get_bounds(self):
        """Returns the current boundaries for the parameter"""

        return self._min_value, self._max_value

    bounds = property(_get_bounds, _set_bounds, doc="Gets or sets the boundaries (minimum and maximum) for this "
                                                    "parameter")

    def _set_gaussian_parameter(self, parameter):
        """Sets the boundaries for this parameter to min_value and max_value"""

        # Use the properties so that the checks and the handling of units are made automatically

        mu, sigma = parameter

        # Remove old boundaries to avoid problems with the new one, if the current value was within the old boundaries
        # but is not within the new ones (it will then be adjusted automatically later)
        self._mu = None
        self._sigma = None

        self._mu = mu

        self._sigma = sigma

    def _get_gaussian_parameter(self):
        """Returns the current boundaries for the parameter"""

        return self._mu, self._sigma

    gaussian_parameter = property(_get_gaussian_parameter, _set_gaussian_parameter, doc="Gets or sets the gaussian paramter"
                                                                                        " (mu and sigma) for this parameter")

    @property
    def normalization(self):
        return self._normalization

    @property
    def name(self):
        return self._name

    @property
    def prior(self):
        return self._prior

    @property
    def get_prior_parameter(self):
        return self._min_value, self._max_value, self._mu, self._sigma
