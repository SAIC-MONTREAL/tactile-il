from setuptools import setup, find_packages
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The transform_utils package is designed to work with Python 3.6 " \
    "and greater Please install it before proceeding."

setup(
    version='0.0.1',
    name='transform_utils',
    # py_modules=['transform_utils'],
    packages=find_packages(),
    install_requires=[
        'numpy',
        'transformations>=2021.6.6',
        'numpy-quaternion',
        'std-msgs',
        'geometry-msgs',
        # 'open3d>=0.13.0',
    ],
    description="Common transformation functionalities that are used across repos.",
    author="SAIC-Montreal",
    include_package_data=True
)
