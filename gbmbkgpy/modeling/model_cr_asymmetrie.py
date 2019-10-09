import numpy as np


def cr_asymmetrie_nb_n5(east, down):
    """
    Gives asymmetrie between nb and n5 side (nb/n5) for certain
    angle between pointing and east/down direction (NED System)
    """
    A = -0.106
    B = -0.104
    C = 1.409
    alpha = 1.209
    beta = 1.445

    return A * east ** alpha + B * down ** beta + C
