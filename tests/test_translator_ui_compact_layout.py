from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TranslatorUiCompactLayoutTests(unittest.TestCase):
    def test_primary_surface_groups_volume_with_each_direction(self) -> None:
        source = (ROOT / "apps/translator-ui/src/main.ts").read_text()

        self.assertNotIn("function audioMixBand", source)
        self.assertIn('audioMixSlider("Оригинал", originalField', source)
        self.assertIn('audioMixSlider("Перевод", translationField', source)
        self.assertNotIn('"translator_set_latency_mode"', source)
        self.assertIn('state.snapshot.provider_id === "local"', source)
        self.assertIn('labeledControl("Голос", voice)', source)

    def test_advanced_controls_are_collapsed_and_keep_open_state(self) -> None:
        source = (ROOT / "apps/translator-ui/src/main.ts").read_text()
        self.assertIn('element("details", "advanced-settings")', source)
        self.assertIn("details.open = state.advancedOpen", source)
        self.assertIn('details.addEventListener("toggle"', source)
        self.assertIn("activeAudioMixInput && state.connected", source)


if __name__ == "__main__":
    unittest.main()
