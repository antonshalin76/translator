from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


class Task8UiControlsTests(unittest.TestCase):
    def test_sidebar_routes_and_diagnostics_are_clickable_sections(self) -> None:
        source = read("apps/translator-ui/src/main.ts")

        self.assertIn('navItem("Состояние", "status")', source)
        self.assertIn('navItem("Маршруты", "routes")', source)
        self.assertIn('navItem("Диагностика", "diagnostics")', source)
        self.assertNotIn('navItem("Маршруты", false)', source)
        self.assertNotIn('navItem("Диагностика", false)', source)
        self.assertRegex(source, r"function navItem\([^)]*section: UiSection")
        self.assertIn('item.setAttribute("href", `#${section}`)', source)
        self.assertIn("scrollIntoView", source)
        self.assertIn('section.id = "routes"', source)
        self.assertIn('section.id = "diagnostics"', source)

    def test_audio_mix_sliders_commit_changes_without_replacing_active_input(
        self,
    ) -> None:
        source = read("apps/translator-ui/src/main.ts")

        input_handler = re.search(
            r'input\.addEventListener\("input", \(\) => \{(?P<body>.*?)\n  \}\);',
            source,
            re.S,
        )
        self.assertIsNotNone(input_handler, "audio mix input handler is missing")
        body = input_handler.group("body")

        self.assertIn("valueNode.textContent", body)
        self.assertNotIn("invoke", body)
        self.assertIn("flushAudioMixChange(field, Number(input.value))", source)
        self.assertIn("function invokeAudioMixChange", source)
        self.assertIn(
            "currentAudioMixPatchIntent(next[0], next[1], state.snapshot)", source
        )
        self.assertIn("pendingAudioMix.set(field, value)", source)
        self.assertIn("pendingAudioMix.clear()", source)
        self.assertIn("drainAudioMixChanges", source)


if __name__ == "__main__":
    unittest.main()
