from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2]


def read_package_file(relative_path: str) -> str:
    return (PACKAGE_ROOT / relative_path).read_text(encoding="utf-8")


def test_router_accepts_only_canonical_text_command_schema() -> None:
    source = read_package_file("fa_voice_command_router_py/router_node.py")

    assert 'cmd == "start"' in source
    assert 'cmd == "stop"' in source
    assert 'cmd == "mode" and len(args) == 1' in source
    assert 'cmd == "status"' in source

    assert '"on"' not in source
    assert '"enable"' not in source
    assert '"off"' not in source
    assert '"disable"' not in source
    assert '"set_mode"' not in source
    assert 'cmd.startswith("mode:")' not in source
    assert 'cmd.split(":", 1)' not in source
    assert 'cmd in ("status", "state")' not in source
