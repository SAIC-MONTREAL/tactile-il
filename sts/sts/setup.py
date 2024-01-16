import os
from glob import glob
from setuptools import setup

package_name = 'sts'

setup(
    name=package_name,
    version='1.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),

    ],
    install_requires=['setuptools', 'filterpy'],
    zip_safe=True,
    maintainer='jenkin',
    maintainer_email='michaeljenkin@me.com',
    description='STS image processing pipeline and associated tools',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'calibration = sts.ros.calibration_node:main',
            'marker_detector = sts.ros.marker_detection_node:main',
            'flow_detector = sts.ros.flow_detection_node:main',
            'slip_detector = sts.ros.slip_detection_end_to_end_node:main',
            'end_to_end_slip_detector = sts.ros.slip_end_to_end_node:main',
            'depth_estimator = sts.ros.depth_estimator_node:main',
            'set_sts_mode_node = sts.ros.set_sts_mode_node:main',
            'remote_sts_bringdown = sts.ros.remote_sts_bringdown:main',
            'contact_detection = sts.ros.contact_detection_node:main'
        ],
    },
)
