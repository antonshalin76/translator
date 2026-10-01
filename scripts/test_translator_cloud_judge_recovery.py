"""Supplemental judge attempts never replace the primary failure receipt."""

from __future__ import annotations

import unittest

import translator_cloud_judge_recovery as recovery


class JudgeRecoveryTests(unittest.TestCase):
    def test_only_failed_primary_judgments_are_selected(self) -> None:
        rows = [
            {
                "origin_id": f"case-{index}",
                "condition": "clean",
                "claude": {"status": "ERROR" if index < 6 else "COMPLETED"},
            }
            for index in range(24)
        ]
        self.assertEqual(len(recovery.failed_rows(rows)), 6)
        rows[6]["origin_id"] = rows[0]["origin_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            recovery.failed_rows(rows)

    def test_merge_keeps_original_rows_and_counts_supplement(self) -> None:
        original = [
            {
                "origin_id": "case-0",
                "condition": "clean",
                "status": "INCOMPLETE",
                "openai": {"status": "COMPLETED", "text": "heard"},
                "google": {"status": "COMPLETED", "text": "heard"},
                "claude": {"status": "ERROR"},
            }
        ]
        supplemental = [
            {
                "origin_id": "case-0",
                "condition": "clean",
                "status": "COMPLETED",
                "verdicts": {"A": {}},
            }
        ]
        combined = recovery.combine(original, supplemental)
        self.assertEqual(original[0]["status"], "INCOMPLETE")
        self.assertEqual(combined[0]["status"], "COMPLETED")
        self.assertEqual(combined[0]["claude"]["source"], "supplemental")


if __name__ == "__main__":
    unittest.main()
