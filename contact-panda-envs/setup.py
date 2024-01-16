from setuptools import setup, find_packages

setup(
    name='contact_panda_envs',
    version='0.0.1',
    description='Gym style envs for panda arms combined with contact sensors.',
    author='Trevor Ablett',
    packages=find_packages(),
    install_requires=[
        'gym',
        'rospkg',
        'opencv-python',
        'scikit-image',
        'pyassimp'
    ],
    include_package_data=True
)