"""
Build the compiled HODLR inner loop.

    python setup.py build_ext --inplace

The extension is optional: ``HodlrInverse`` falls back to its Python
implementation when the module is absent, so the package works without
a compiler and gets faster with one.
"""

import numpy
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError:                                    # pragma: no cover
    raise SystemExit(
        "Cython is required to build the extension:\n"
        "    pip install cython\n"
        "The pure-Python path works without it.")

setup(
    name="nextlmm-hodlr",
    ext_modules=cythonize(
        [Extension("_hodlr_fast", ["_hodlr_fast.pyx"],
                   include_dirs=[numpy.get_include()],
                   # -O3 over the default -O2, and let the compiler use
                   # whatever this machine's vector units offer: the
                   # inner loops are small dense products that vectorise
                   # cleanly.
                   extra_compile_args=["-O3", "-march=native",
                                       "-ffast-math"])],
        language_level=3,
        compiler_directives={"boundscheck": False, "wraparound": False,
                             "cdivision": True},
    ),
)
