try:
    from diffcp._version import version as __version__
except ImportError:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("diffcp")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

from diffcp.cone_program import solve_and_derivative, \
    solve_and_derivative_batch, \
    solve_and_derivative_internal, \
    solve_only_batch, solve_only, \
    solve_internal, SolverError
from diffcp.cones import ZERO, POS, SOC, PSD, EXP
from diffcp import utils
