from http.cookies import SimpleCookie
import subprocess
import sys
from unittest.mock import Mock
from urllib.parse import urlsplit

import pytest

from desktop import launcher
from spikes.desktop.shell import wait_for_worker


def test_worker_that_never_announces_readiness_times_out():
    with subprocess.Popen(
        [sys.executable, "-I", "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    ) as worker:
        try:
            with pytest.raises(RuntimeError, match="startup timed out"):
                wait_for_worker(worker, timeout=0.1)
        finally:
            worker.stdin.close()
            worker.wait(timeout=5)


WORKSPACE_URL = "http://127.0.0.1:30101/"


def picker_window():
    return Mock(
        get_current_url=Mock(return_value=WORKSPACE_URL + "?screen=tensors"),
        get_cookies=Mock(return_value=[SimpleCookie("layerdelta_session=human-session")]),
    )


def picker_connection(monkeypatch, status):
    connection = Mock()
    connection.getresponse.return_value.status = status
    opener = Mock(return_value=connection)
    monkeypatch.setattr(launcher, "HTTPConnection", opener)
    return opener, connection


@pytest.mark.parametrize("filename", ["weights.safetensors", "model.safetensors.index.json"])
def test_checkpoint_picker_returns_absolute_path_for_a_human_session(monkeypatch, tmp_path, filename):
    window = picker_window()
    path = tmp_path / filename
    window.create_file_dialog.return_value = (str(path),)
    opener, connection = picker_connection(monkeypatch, 200)

    assert launcher.pick_checkpoint_path(window, urlsplit(WORKSPACE_URL), "project") == str(path)

    opener.assert_called_once_with("127.0.0.1", 30101, timeout=3)
    connection.request.assert_called_once_with(
        "GET", "/api/v1/projects/project/whoami", headers={"Cookie": "layerdelta_session=human-session"}
    )
    connection.close.assert_called_once_with()
    window.create_file_dialog.assert_called_once_with(
        file_types=("Safetensors (*.safetensors;*.safetensors.index.json)",)
    )


@pytest.mark.parametrize("status", [401, 403])
def test_checkpoint_picker_refuses_an_unverified_session(monkeypatch, status):
    window = picker_window()
    _, connection = picker_connection(monkeypatch, status)

    assert launcher.pick_checkpoint_path(window, urlsplit(WORKSPACE_URL), "project") is None
    window.create_file_dialog.assert_not_called()
    connection.close.assert_called_once_with()


@pytest.mark.parametrize("url, cookies", [("https://example.invalid/", True), (WORKSPACE_URL, False)])
def test_checkpoint_picker_refuses_another_origin_or_missing_cookie(monkeypatch, url, cookies):
    opener, _ = picker_connection(monkeypatch, 200)
    window = picker_window()
    window.get_current_url.return_value = url
    if not cookies:
        window.get_cookies.return_value = []

    assert launcher.pick_checkpoint_path(window, urlsplit(WORKSPACE_URL), "project") is None
    opener.assert_not_called()
    window.create_file_dialog.assert_not_called()


def test_checkpoint_picker_discards_results_after_navigation(monkeypatch, tmp_path):
    picker_connection(monkeypatch, 200)
    window = picker_window()
    window.get_current_url.side_effect = [WORKSPACE_URL, "https://example.invalid/"]
    window.create_file_dialog.return_value = (str(tmp_path / "weights.safetensors"),)

    assert launcher.pick_checkpoint_path(window, urlsplit(WORKSPACE_URL), "project") is None
    window.create_file_dialog.assert_called_once()


@pytest.mark.parametrize("filename", [None, "relative.safetensors", "unsafe.pt"])
def test_checkpoint_picker_ignores_cancelled_or_unsupported_selection(monkeypatch, tmp_path, filename):
    picker_connection(monkeypatch, 200)
    window = picker_window()
    path = tmp_path / filename if filename == "unsafe.pt" else filename
    window.create_file_dialog.return_value = (str(path),) if path is not None else None

    assert launcher.pick_checkpoint_path(window, urlsplit(WORKSPACE_URL), "project") is None
