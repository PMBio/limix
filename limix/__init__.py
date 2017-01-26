from .mtSet import MTSet
from .varDecomp import VarianceDecomposition

from pkg_resources import get_distribution as _get_distribution
from pkg_resources import DistributionNotFound as _DistributionNotFound

try:
    __version__ = _get_distribution('limix').version
except _DistributionNotFound:
    __version__ = 'unknown'


def test():
    import os
    p = __import__('limix').__path__[0]
    src_path = os.path.abspath(p)
    old_path = os.getcwd()
    os.chdir(src_path)

    try:
        return_code = __import__('pytest').main(['-q'])
    finally:
        os.chdir(old_path)

    if return_code == 0:
        print("Congratulations. All tests have passed!")

    return return_code
