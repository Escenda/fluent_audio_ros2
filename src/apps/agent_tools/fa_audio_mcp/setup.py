from setuptools import setup


package_name = "fa_audio_mcp"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "mcp>=1"],
    zip_safe=True,
    maintainer="FluentAudio",
    maintainer_email="maintainer@example.com",
    description="MCP adapter exposing FluentAudio timeline services as tools.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "fa_audio_mcp_server = fa_audio_mcp.server:main",
        ]
    },
)
