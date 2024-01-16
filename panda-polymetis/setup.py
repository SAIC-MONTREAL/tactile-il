from setuptools import setup, find_packages

setup(
    name='panda_polymetis',
    version='0.0.1',
    description='Package for accessings sensors and controlling the panda with polymetis.',
    author='Trevor Ablett',
    packages=find_packages(),
    install_requires=[
        'polymetis',
    ],
    include_package_data=True
)