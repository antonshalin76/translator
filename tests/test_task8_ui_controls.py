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

    def test_audio_mix_sliders_apply_changes_from_input_events(self) -> None:
        source = read("apps/translator-ui/src/main.ts")

        input_handler = re.search(
            r'input\.addEventListener\("input", \(\) => \{(?P<body>.*?)\n  \}\);',
            source,
            re.S,
        )
        self.assertIsNotNone(input_handler, "audio mix input handler is missing")
        body = input_handler.group("body")

        self.assertIn("queueAudioMixChange(field, Number(input.value))", body)
        self.assertIn("flushAudioMixChange(field, Number(input.value))", source)
        self.assertIn("function invokeAudioMixChange", source)
        self.assertIn("state.snapshot.audio_mix", source)
        self.assertIn("audioMixTimers", source)


if __name__ == "__main__":
    unittest.main()
