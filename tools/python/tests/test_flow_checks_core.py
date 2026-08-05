import tempfile
import unittest
from pathlib import Path

from flow_checks.core import FlowSession


class FlowSessionTests(unittest.TestCase):
    def test_load_keeps_remote_copy_trace_in_file_order(self):
        content = "\n".join(
            (
                "[Flow] 14:21:04.000 T 1 log mode=FlowVerification",
                "AniloxRoll.Monitor.exe Warning: 0 : "
                "[RemoteCopy] remote share unavailable: TCP 445 unavailable.",
                "[Flow] 14:21:05.000 T 1 capture finalize grab=260730-142054",
                "AniloxRoll.Monitor.exe Information: 0 : "
                "[RemoteCopy] pending queued added=2 queue=2 bytes=2048",
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.log"
            path.write_text(content, encoding="utf-8")
            session = FlowSession.load(path)

        self.assertEqual(
            [
                "log mode=FlowVerification",
                "[RemoteCopy] remote share unavailable: TCP 445 unavailable.",
                "capture finalize grab=260730-142054",
                "[RemoteCopy] pending queued added=2 queue=2 bytes=2048",
            ],
            [line.message for line in session.lines],
        )
        self.assertEqual("14:21:04.000", session.lines[1].timestamp)
        self.assertEqual("14:21:05.000", session.lines[3].timestamp)

    def test_load_does_not_treat_runner_echo_as_product_trace(self):
        content = (
            "[C3] 完成 - AniloxRoll.Monitor.exe Information: 0 : "
            "[RemoteCopy] pending queued added=2 queue=2 bytes=2048"
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runner.log"
            path.write_text(content, encoding="utf-8")
            session = FlowSession.load(path)

        self.assertEqual([], session.lines)


if __name__ == "__main__":
    unittest.main()
