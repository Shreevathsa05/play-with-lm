import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.capabilities import probe


class CapabilityProbeTest(unittest.TestCase):
    def test_probe_never_claims_uncertified_objectives(self):
        report = probe()
        self.assertIn("hardware", report)
        self.assertIn("dpo", report["objectives"])
        self.assertFalse(any(item["certified"] for item in report["objectives"].values()))


if __name__ == "__main__":
    unittest.main()
