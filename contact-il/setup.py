from setuptools import setup, find_packages

setup(
    name='contact_il',
    version='0.0.1',
    description='Imitation learning with contact sensing.',
    author='Trevor Ablett',
    packages=find_packages(),
    install_requires=[
        'inputs @ git+https://github.com/trevorablett/inputs.git@non-blocking-gamepad#egg=inputs',
        'transform_utils',  # must be manually installed for the time being
        'PyYAML',
        #'contact_panda_envs',
        #'hydra-core',
        #'tensorboard<=2.11.2',
        #'place_from_pick_learning',
        'numpy',
        'matplotlib',
        #'rospkg',
        'scipy',
        'pandas',
        #'scikit-image',
        #'simple-pid',
        #'transforms3d'
    ],
    include_package_data=True
)