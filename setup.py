from distutils.core import setup
from distutils.extension import Extension

setup(name="FASTUSURF",
    ext_modules=[
        Extension("fastusurf", ["fastusurf.cpp"],
        libraries = ["boost_python"])
    ])
