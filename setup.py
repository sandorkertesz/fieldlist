import io
import os

from setuptools import find_packages, setup

ROOT_DIR = os.path.dirname(__file__)


def read(fname):
    file_path = os.path.join(ROOT_DIR, fname)
    return io.open(file_path, encoding="utf-8").read()


exec(open(os.path.join(ROOT_DIR, "fieldlist", "version.py")).read())


setup(
    name="fieldlist",
    version=__version__,
    author="European Centre for Medium-Range Weather Forecasts (ECMWF)",
    author_email="software.support@ecmwf.int",
    license="Apache 2.0",
    description="GRIB field list implementation based on ecCodes",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
)
