from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TranslatorUiCompactLayoutTests(unittest.TestCase):
    def test_direction_panels_do_not_render_low_value_detail_table(self) -> None:
        source = (ROOT / "apps/translator-ui/src/main.ts").read_text()

        for label in (
            'detail("Канал"',
            'detail("Режим"',
            'detail("Latency p50"',
            'detail("Latency p95 first"',
            'detail("Latency p95 last"',
            'detail("Degradation"',
            'detail("Голос"',
        ):
            self.assertNotIn(label, source)

        self.assertIn('labeledControl("Mode", mode)', source)
        self.assertIn('labeledControl("Voice", voice)', source)


if __name__ == "__main__":
    unittest.main()
