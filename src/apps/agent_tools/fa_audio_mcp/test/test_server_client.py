import importlib
import inspect
import sys
from collections.abc import Callable
from types import ModuleType

import pytest

from fa_audio_mcp.config import ServerConfig
from fa_audio_mcp.errors import AudioToolError
from fa_audio_mcp.json_types import JsonValue
from fa_audio_mcp.requests import (
    ArchiveAudioRequestValues,
    TranscribeAudioRequestValues,
)
from fa_audio_mcp.scopes import AudioScopeConfig
from fa_audio_mcp.time_range import NumericTimeRange

ToolFunction = Callable[..., dict[str, JsonValue]]


class _FakeFastMCP:
    def __init__(self, name: str) -> None:
        self.name = name
        self.tools: dict[str, ToolFunction] = {}

    def tool(self):
        def _decorator(func: ToolFunction) -> ToolFunction:
            self.tools[func.__name__] = func
            return func

        return _decorator


class _FakeRequest:
    pass


class _FakeArchiveAudioWindow:
    Request = _FakeRequest


class _FakeTranscribeAudio:
    Request = _FakeRequest


class _FakeNodeBase:
    pass


class _FakeResponse:
    pass


class _FakeFuture:
    def __init__(self, response: _FakeResponse | None) -> None:
        self._response = response

    def result(self) -> _FakeResponse | None:
        return self._response


class _FakeClient:
    def __init__(
        self,
        *,
        available: bool,
        response: _FakeResponse | None = None,
    ) -> None:
        self.available = available
        self.response = response
        self.wait_timeout_sec = 0.0
        self.request = _FakeRequest()

    def wait_for_service(self, *, timeout_sec: float) -> bool:
        self.wait_timeout_sec = timeout_sec
        return self.available

    def call_async(self, request: _FakeRequest) -> _FakeFuture:
        self.request = request
        return _FakeFuture(self.response)


class _FakeNode:
    def __init__(self, archive_client: _FakeClient, transcribe_client: _FakeClient) -> None:
        self._archive_client = archive_client
        self._transcribe_client = transcribe_client
        self.created_service_names: list[str] = []

    def create_client(self, service_type, service_name: str) -> _FakeClient:
        self.created_service_names.append(service_name)
        if service_type is _FakeArchiveAudioWindow:
            return self._archive_client
        if service_type is _FakeTranscribeAudio:
            return self._transcribe_client
        raise RuntimeError("unexpected service type")


def _install_fake_server_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    mcp_module = ModuleType("mcp")
    mcp_server_module = ModuleType("mcp.server")
    fastmcp_module = ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = _FakeFastMCP

    rclpy_module = ModuleType("rclpy")
    rclpy_module.spin_until_future_complete = _spin_until_future_complete
    node_module = ModuleType("rclpy.node")
    node_module.Node = _FakeNodeBase

    fa_interfaces_module = ModuleType("fa_interfaces")
    srv_module = ModuleType("fa_interfaces.srv")
    srv_module.ArchiveAudioWindow = _FakeArchiveAudioWindow
    srv_module.TranscribeAudio = _FakeTranscribeAudio

    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", mcp_server_module)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_module)
    monkeypatch.setitem(sys.modules, "rclpy", rclpy_module)
    monkeypatch.setitem(sys.modules, "rclpy.node", node_module)
    monkeypatch.setitem(sys.modules, "fa_interfaces", fa_interfaces_module)
    monkeypatch.setitem(sys.modules, "fa_interfaces.srv", srv_module)
    sys.modules.pop("fa_audio_mcp.server", None)


def _spin_until_future_complete(
    node: _FakeNode,
    future: _FakeFuture,
    *,
    timeout_sec: float,
) -> None:
    node.spin_timeout_sec = timeout_sec


def _import_server(monkeypatch: pytest.MonkeyPatch):
    _install_fake_server_dependencies(monkeypatch)
    return importlib.import_module("fa_audio_mcp.server")


def _server_config() -> ServerConfig:
    return ServerConfig(
        transport="stdio",
        host="0.0.0.0",
        port=9110,
        archive_service_name="archive_audio_window",
        transcribe_service_name="transcribe_audio",
        service_timeout_sec=3.5,
        archive_scope_config=AudioScopeConfig(mic="mic"),
        transcribe_scope_config=AudioScopeConfig(mic="audio/high_pass/mic"),
    )


def _archive_values() -> ArchiveAudioRequestValues:
    return ArchiveAudioRequestValues(
        time_range=NumericTimeRange(10, 20),
        time_range_spec="10..20",
        audio_scope="mic",
        reason="evidence",
        related_artifact_ids=["artifact-1"],
        codec="pcm_s16le",
        container="wav",
        payload_format="audio/wav",
    )


def _transcribe_values() -> TranscribeAudioRequestValues:
    return TranscribeAudioRequestValues(
        time_range=NumericTimeRange(10, 20),
        time_range_spec="10..20",
        audio_scope="audio/high_pass/mic",
    )


def test_ros_client_raises_service_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _import_server(monkeypatch)
    archive_client = _FakeClient(available=False)
    node = _FakeNode(
        archive_client,
        _FakeClient(available=True, response=_FakeResponse()),
    )
    client = server.RosAudioTimelineClient(node, _server_config())

    with pytest.raises(AudioToolError) as exc_info:
        client.archive_audio_window(_archive_values())

    assert exc_info.value.error_code == "service_unavailable"
    assert archive_client.wait_timeout_sec == 3.5


def test_ros_client_raises_service_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _import_server(monkeypatch)
    archive_client = _FakeClient(available=True, response=None)
    node = _FakeNode(
        archive_client,
        _FakeClient(available=True, response=_FakeResponse()),
    )
    client = server.RosAudioTimelineClient(node, _server_config())

    with pytest.raises(AudioToolError) as exc_info:
        client.archive_audio_window(_archive_values())

    assert exc_info.value.error_code == "service_timeout"
    assert node.spin_timeout_sec == 3.5


def test_ros_client_returns_response_and_populates_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _import_server(monkeypatch)
    response = _FakeResponse()
    transcribe_client = _FakeClient(available=True, response=response)
    node = _FakeNode(
        _FakeClient(available=True, response=_FakeResponse()),
        transcribe_client,
    )
    client = server.RosAudioTimelineClient(node, _server_config())

    result = client.transcribe_audio(_transcribe_values())

    assert result is response
    assert transcribe_client.request.time_range_spec == "10..20"
    assert transcribe_client.request.audio_scope == "audio/high_pass/mic"
    assert node.created_service_names == ["archive_audio_window", "transcribe_audio"]


def test_ros_client_populates_archive_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _import_server(monkeypatch)
    response = _FakeResponse()
    archive_client = _FakeClient(available=True, response=response)
    node = _FakeNode(
        archive_client,
        _FakeClient(available=True, response=_FakeResponse()),
    )
    client = server.RosAudioTimelineClient(node, _server_config())

    result = client.archive_audio_window(_archive_values())

    assert result is response
    assert archive_client.request.time_range_spec == "10..20"
    assert archive_client.request.audio_scope == "mic"
    assert archive_client.request.reason == "evidence"
    assert archive_client.request.related_artifact_ids == ["artifact-1"]
    assert archive_client.request.codec == "pcm_s16le"
    assert archive_client.request.container == "wav"
    assert archive_client.request.payload_format == "audio/wav"


def test_mcp_tools_make_audio_scope_omittable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _import_server(monkeypatch)

    mcp = server.build_mcp_server(
        _server_config(),
        server.RosAudioTimelineClient(
            _FakeNode(
                _FakeClient(available=True, response=_FakeResponse()),
                _FakeClient(available=True, response=_FakeResponse()),
            ),
            _server_config(),
        ),
        server.AudioScopeResolver(AudioScopeConfig(mic="mic", default_scope_key="mic")),
        server.AudioScopeResolver(
            AudioScopeConfig(
                mic="audio/high_pass/mic",
                default_scope_key="mic",
            )
        ),
    )

    archive_signature = inspect.signature(mcp.tools["archive_audio_window"])
    transcribe_signature = inspect.signature(mcp.tools["transcribe_audio"])

    assert archive_signature.parameters["audio_scope"].default is None
    assert transcribe_signature.parameters["audio_scope"].default is None
