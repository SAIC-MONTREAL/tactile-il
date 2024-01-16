from setuptools import setup, find_packages

setup(
    name='pysts',
    version='0.1',
    description='Package for using an STS without requiring ROS.',
    author='Trevor Ablett',
    packages=find_packages(),
    install_requires=[
        'sts',
        'opencv-python',
    ],
    include_package_data=True
)