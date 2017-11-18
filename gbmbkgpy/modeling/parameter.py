
class Parameter(object):
    def __init__(self, name, initial_value, min_value, max_value, delta, **kwargs):
        self._name = str(name)
        self._value = initial_value
        self._min_value = min_value
        self._max_value = max_value
        self._delta = delta

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


        return "%20s: %10g %10g %10g %10g %s" % (
            self._name, self._value, self._min_value, self._max_value, self._delta, ff)


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
        self.min_value = None
        self.max_value = None

        self.min_value = min_value

        self.max_value = max_value


    def _get_bounds(self):
        """Returns the current boundaries for the parameter"""

        return self.min_value, self.max_value


    bounds = property(_get_bounds, _set_bounds, doc="Gets or sets the boundaries (minimum and maximum) for this "
                                                    "parameter")

    @property
    def normalization(self):
        return self._normalization


