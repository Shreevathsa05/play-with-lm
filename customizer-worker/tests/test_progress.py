import os
import sys
import unittest
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.logcarrier import LogCarrier
from src.utils.progress import attach_progress_sink, event_phase, progress_notify


class TestProgress(unittest.TestCase):
    @patch("src.utils.progress.httpx.post")
    def test_progress_notify_sends_phase_and_event(self, post):
        post.return_value.raise_for_status = lambda: None
        progress_notify(
            "job-1",
            "RUNNING",
            phase="Dataset loaded",
            event={"event": "dataset_loaded"},
            force=True,
        )
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["status"], "RUNNING")
        self.assertEqual(payload["report"]["phase"], "Dataset loaded")
        self.assertEqual(payload["report"]["event"]["event"], "dataset_loaded")

    def test_event_phase_formats_training_step(self):
        text = event_phase({"event": "train_progress", "step": 4, "total": 10, "loss": 1.2345})
        self.assertIn("step 4/10", text)
        self.assertIn("loss 1.2345", text)

    @patch("src.utils.progress.progress_notify")
    def test_attach_progress_sink_forwards_log_events(self, notify):
        log = LogCarrier("job-2")
        attach_progress_sink(log, "job-2")
        log.info("dataset_loaded", rows=10)
        notify.assert_called()
        self.assertEqual(notify.call_args.kwargs["event"]["event"], "dataset_loaded")


if __name__ == "__main__":
    unittest.main()
