from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2]


def read_package_file(relative_path: str) -> str:
    return (PACKAGE_ROOT / relative_path).read_text(encoding="utf-8")


def test_colcon_registers_package_pytest_suite() -> None:
    cmake = read_package_file("CMakeLists.txt")
    package = read_package_file("package.xml")

    assert "if(BUILD_TESTING)" in cmake
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package
    assert "<test_depend>python3-pytest</test_depend>" in package
    assert "<test_depend>python3-yaml</test_depend>" in package
