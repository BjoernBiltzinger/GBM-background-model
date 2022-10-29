from astromodels import Function1D, FunctionMeta
import astropy.units as astropy_units


class SBPL(Function1D, metaclass=FunctionMeta):
    r"""
    description :
        SBPL
    latex : test
    parameters :
        K :
            desc : Differential flux at the break energy
            initial value : 1e-4
            min: 1e-50
            is_normalization : True
            transformation : log10
        alpha :
            desc : low-energy photon index
            initial value : -1.0
            min : -1.5
            max : 3
        xb :
            desc : break energy
            initial value : 500
            min : 10
            transformation : log10
        beta :
            desc : high-energy photon index
            initial value : -2.0
            min : -5.0
            max : -1.6
    """

    def _set_units(self, x_unit, y_unit):
        # The normalization has the same units as y
        self.K.unit = y_unit

        # The break point has always the same dimension as the x variable
        self.xb.unit = x_unit

        # alpha and beta are dimensionless
        self.alpha.unit = astropy_units.dimensionless_unscaled
        self.beta.unit = astropy_units.dimensionless_unscaled

    def evaluate(self, x, K, alpha, xb, beta):

        if isinstance(x, astropy_units.Quantity):
            alpha_ = alpha.value
            beta_ = beta.value
            K_ = K.value
            x_ = x.value
            xb_ = xb.value
            unit_ = self.y_unit

        else:
            unit_ = 1.0
            alpha_, beta_, K_, x_, xb__ = alpha, beta, K, x, xb

        return K/((x/xb)**alpha+(x/xb)**beta)


def fix_all_params(astro_func):
    """
    Helper function to fix all the parameters of a astromodel function
    """

    for key, param in astro_func.free_parameters.items():
        param.fix = True
