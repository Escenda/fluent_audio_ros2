from setuptools import setup
from glob import glob
import os

package_name = 'fluent_vision_system'

def files_in(dirpath):
    return [f for f in glob(os.path.join(dirpath, '*')) if os.path.isfile(f)]

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', files_in('launch')),
        ('share/' + package_name + '/config', files_in('config')),
        ('lib/' + package_name, ['scripts/fv_tf_debugger_node']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maintainer',
    maintainer_email='noreply@example.com',
    description='YAML駆動のローンチパッケージ',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fv_tf_debugger_node = fluent_vision_system.tf_debugger_node:main',
        ],
    },
)


