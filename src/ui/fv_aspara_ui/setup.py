from setuptools import setup

package_name = 'fv_aspara_ui'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Fluent Vision',
    maintainer_email='dev@fluent-vision.local',
    description='Asparagus UI overlay node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fv_aspara_ui_node = fv_aspara_ui.ui_node:main',
        ],
    },
)

