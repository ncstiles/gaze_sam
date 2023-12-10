from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("./integration/trt_sam_postprocess.pyx",
                          "./integration/trt_sam_preprocess.pyx"),
    #ext_modules=cythonize("./integration/trt_sam_preprocess.pyx"),
    include_dirs=[np.get_include()]

)