"""
Example Triangles

Well-known claims triangles from the reserving literature, used in the
documentation and to check results against published figures.
"""

import numpy as np
from .triangle import Triangle

_NA = np.nan


def load_raa() -> Triangle:
    """
    RAA triangle: cumulative incurred claims of automatic facultative general
    liability business, accident years 1981-1990 (in thousands).

    Source: Reinsurance Association of America, Historical Loss Development
    Study (1991). Analysed in Mack (1993) and shipped with the R ChainLadder
    package as ``RAA``.
    """
    values = [
        [5012, 8269, 10907, 11805, 13539, 16181, 18009, 18608, 18662, 18834],
        [106, 4285, 5396, 10666, 13782, 15599, 15496, 16169, 16704, _NA],
        [3410, 8992, 13873, 16141, 18735, 22214, 22863, 23466, _NA, _NA],
        [5655, 11555, 15766, 21266, 23425, 26083, 27067, _NA, _NA, _NA],
        [1092, 9565, 15836, 22169, 25955, 26180, _NA, _NA, _NA, _NA],
        [1513, 6445, 11702, 12935, 15852, _NA, _NA, _NA, _NA, _NA],
        [557, 4020, 10946, 12314, _NA, _NA, _NA, _NA, _NA, _NA],
        [1351, 6947, 13112, _NA, _NA, _NA, _NA, _NA, _NA, _NA],
        [3133, 5395, _NA, _NA, _NA, _NA, _NA, _NA, _NA, _NA],
        [2063, _NA, _NA, _NA, _NA, _NA, _NA, _NA, _NA, _NA],
    ]
    return Triangle(values, origin=range(1981, 1991), development=range(1, 11),
                    cumulative=True, name="RAA")


def load_genins() -> Triangle:
    """
    Taylor and Ashe triangle: cumulative paid claims of a general insurance
    portfolio over ten origin years.

    Source: Taylor and Ashe (1983), "Second moments of estimates of
    outstanding claims", Journal of Econometrics 23. Used throughout the
    stochastic reserving literature, including England and Verrall (2002),
    and shipped with the R ChainLadder package as ``GenIns``.
    """
    values = [
        [357848, 1124788, 1735330, 2218270, 2745596, 3319994, 3466336, 3606286, 3833515, 3901463],
        [352118, 1236139, 2170033, 3353322, 3799067, 4120063, 4647867, 4914039, 5339085, _NA],
        [290507, 1292306, 2218525, 3235179, 3985995, 4132918, 4628910, 4909315, _NA, _NA],
        [310608, 1418858, 2195047, 3757447, 4029929, 4381982, 4588268, _NA, _NA, _NA],
        [443160, 1136350, 2128333, 2897821, 3402672, 3873311, _NA, _NA, _NA, _NA],
        [396132, 1333217, 2180715, 2985752, 3691712, _NA, _NA, _NA, _NA, _NA],
        [440832, 1288463, 2419861, 3483130, _NA, _NA, _NA, _NA, _NA, _NA],
        [359480, 1421128, 2864498, _NA, _NA, _NA, _NA, _NA, _NA, _NA],
        [376686, 1363294, _NA, _NA, _NA, _NA, _NA, _NA, _NA, _NA],
        [344014, _NA, _NA, _NA, _NA, _NA, _NA, _NA, _NA, _NA],
    ]
    return Triangle(values, origin=range(1, 11), development=range(1, 11),
                    cumulative=True, name="GenIns")


def load_mw2008() -> Triangle:
    """
    Cumulative claims triangle of Merz and Wuthrich (2008), nine accident
    years, used in their paper on the one-year claims development result
    and shipped with the R ChainLadder package as ``MW2008``.

    Source: Merz, M. and Wuthrich, M.V. (2008). Modelling the claims
    development result for solvency purposes. CAS E-Forum, Fall 2008.
    """
    values = [
        [2202584, 3210449, 3468122, 3545070, 3621627, 3644636, 3669012, 3674511, 3678633],
        [2350650, 3553023, 3783846, 3840067, 3865187, 3878744, 3898281, 3902425, _NA],
        [2321885, 3424190, 3700876, 3798198, 3854755, 3878993, 3898825, _NA, _NA],
        [2171487, 3165274, 3395841, 3466453, 3515703, 3548422, _NA, _NA, _NA],
        [2140328, 3157079, 3399262, 3500520, 3585812, _NA, _NA, _NA, _NA],
        [2290664, 3338197, 3550332, 3641036, _NA, _NA, _NA, _NA, _NA],
        [2148216, 3219775, 3428335, _NA, _NA, _NA, _NA, _NA, _NA],
        [2143728, 3158581, _NA, _NA, _NA, _NA, _NA, _NA, _NA],
        [2144738, _NA, _NA, _NA, _NA, _NA, _NA, _NA, _NA],
    ]
    return Triangle(values, origin=range(1, 10), development=range(1, 10),
                    cumulative=True, name="MW2008")


def load_mcl() -> tuple:
    """
    Paid and incurred triangles of Quarg and Mack (2004), seven accident
    years, used to illustrate the Munich chain-ladder and shipped with the
    R ChainLadder package as ``MCLpaid`` and ``MCLincurred``.

    Returns:
        ``(paid, incurred)``
    """
    paid = [
        [576, 1804, 1970, 2024, 2074, 2102, 2131],
        [866, 1948, 2162, 2232, 2284, 2348, _NA],
        [1412, 3758, 4252, 4416, 4494, _NA, _NA],
        [2286, 5292, 5724, 5850, _NA, _NA, _NA],
        [1868, 3778, 4648, _NA, _NA, _NA, _NA],
        [1442, 4010, _NA, _NA, _NA, _NA, _NA],
        [2044, _NA, _NA, _NA, _NA, _NA, _NA],
    ]
    incurred = [
        [978, 2104, 2134, 2144, 2174, 2182, 2174],
        [1844, 2552, 2466, 2480, 2508, 2454, _NA],
        [2904, 4354, 4698, 4600, 4644, _NA, _NA],
        [3502, 5958, 6070, 6142, _NA, _NA, _NA],
        [2812, 4882, 4852, _NA, _NA, _NA, _NA],
        [2642, 4406, _NA, _NA, _NA, _NA, _NA],
        [5022, _NA, _NA, _NA, _NA, _NA, _NA],
    ]
    return (Triangle(paid, name="MCL paid"), Triangle(incurred, name="MCL incurred"))
