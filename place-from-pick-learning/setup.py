from setuptools import setup, find_packages

setup(name='place_from_pick_learning',
      version='1.0',
      packages=[package for package in find_packages()
                if package.startswith('place_from_pick_learning')],
      install_requires=[]
      )
