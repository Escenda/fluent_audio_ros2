from glob import glob
import os

from setuptools import setup


package_name = "fluent_audio_system"


def files_in(dirpath):
    return [path for path in glob(os.path.join(dirpath, "*")) if os.path.isfile(path)]


def files_in_tree(dirpath):
    return [
        path
        for path in glob(os.path.join(dirpath, "**", "*"), recursive=True)
        if os.path.isfile(path)
    ]


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", files_in("launch")),
        ("share/" + package_name + "/config", files_in("config")),
        ("share/" + package_name + "/config/profiles", files_in_tree("config/profiles")),
    ],
    install_requires=["setuptools", "pydantic>=2"],
    zip_safe=True,
    maintainer="FluentAudio",
    maintainer_email="maintainer@example.com",
    description="YAML-driven launch package for FluentAudio node groups.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "list_required_packages = fluent_audio_system.list_required_packages:main",
        ]
    },
)
