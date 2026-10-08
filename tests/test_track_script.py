import signal
import threading

from tests.helpers import load_script


def test_sigint_flag_restores_handler_on_main_thread():
    mod = load_script("03_track.py")
    before = signal.getsignal(signal.SIGINT)
    with mod._sigint_flag() as flag:
        assert flag["flag"] is False
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        assert flag["flag"] is True
    assert signal.getsignal(signal.SIGINT) is before


def test_sigint_flag_is_a_noop_off_the_main_thread():
    mod = load_script("03_track.py")
    box = {}

    def work():
        try:
            with mod._sigint_flag() as flag:
                box["flag"] = flag["flag"]
        except Exception as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=work)
    t.start()
    t.join(5)
    assert "error" not in box and box["flag"] is False
