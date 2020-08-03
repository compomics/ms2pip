import os
import ms2pip


def get_include():
    """
    Return the directory that contains the NumPy \\*.h header files.
    Extension modules that need to compile against NumPy should use this
    function to locate the appropriate include directory.
    Notes
    -----
    When using ``distutils``, for example in ``setup.py``.
    ::
        import numpy as np
        ...
        Extension('extension_name', ...
                include_dirs=[np.get_include()])
        ...
    """
    d = os.path.join(os.path.dirname(ms2pip.__file__), 'cython_modules')
    return d
