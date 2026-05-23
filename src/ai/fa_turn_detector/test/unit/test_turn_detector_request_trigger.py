from fa_interfaces.msg import TurnEndRequest
from fa_turn_detector_py.turn_detector_node import FaTurnDetectorNode


class _Logger:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.warnings: list[str] = []

    def debug(self, message: str) -> None:
        self.debugs.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class _NodeProbe:
    def __init__(self) -> None:
        self._context_active = True
        self._active_session_id = "session-1"
        self._active_user_turn_id = 3
        self.detected_request_ids: list[int] = []
        self.logger = _Logger()

    def get_logger(self) -> _Logger:
        return self.logger

    def _detect_turn_end(self, *, request_id: int) -> None:
        self.detected_request_ids.append(request_id)


def _request(*, session_id: str = "session-1", turn_id: int = 3, request_id: int = 7):
    msg = TurnEndRequest()
    msg.session_id = session_id
    msg.user_turn_id = turn_id
    msg.request_id = request_id
    msg.quiet_ms = 1200
    return msg


def test_turn_end_request_triggers_detection_for_active_context() -> None:
    probe = _NodeProbe()

    FaTurnDetectorNode.on_turn_end_request(probe, _request())

    assert probe.detected_request_ids == [7]


def test_turn_end_request_ignores_stale_context_and_invalid_request_id() -> None:
    probe = _NodeProbe()

    FaTurnDetectorNode.on_turn_end_request(probe, _request(session_id="other"))
    FaTurnDetectorNode.on_turn_end_request(probe, _request(turn_id=4))
    FaTurnDetectorNode.on_turn_end_request(probe, _request(request_id=0))

    assert probe.detected_request_ids == []
    assert probe.logger.debugs == [
        "Dropping TurnEndRequest with stale session",
        "Dropping TurnEndRequest with stale turn",
    ]
    assert probe.logger.warnings == ["Dropping TurnEndRequest with invalid request_id"]
