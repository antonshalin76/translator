from __future__ import annotations

import argparse
import json
from pathlib import Path

from translator_sidecar.local.model_lease import VerifiedModelSource
from translator_sidecar.local.model_manifest import load_manifest
from translator_sidecar.local.mt import NllbTranslator
from translator_sidecar.provider_contract import Language, TranslationMode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()

    translator = NllbTranslator.load(
        VerifiedModelSource(load_manifest(args.manifest), args.model_id),
        device=args.device,
    )
    try:
        results = {
            "ru_en": translator.translate(
                "Доброе утро.",
                source_language=Language.RU,
                target_language=Language.EN,
                mode=TranslationMode.BALANCED,
            ),
            "en_ru": translator.translate(
                "Good morning.",
                source_language=Language.EN,
                target_language=Language.RU,
                mode=TranslationMode.BALANCED,
            ),
            "ru_en_multisentence": translator.translate(
                "Пожалуйста, относитесь к этому месту со всем достоинством, "
                "серьёзностью и уважением, которых оно заслуживает. "
                "Не шутите про Холокост и нацистов.",
                source_language=Language.RU,
                target_language=Language.EN,
                mode=TranslationMode.QUALITY_FIRST,
            ),
            "en_ru_multisentence": translator.translate(
                "Please treat the site with all of the dignity, solemnity "
                "and respect it deserves. Do not make jokes about the "
                "Holocaust or Nazis.",
                source_language=Language.EN,
                target_language=Language.RU,
                mode=TranslationMode.QUALITY_FIRST,
            ),
        }
    finally:
        translator.close()
    print(json.dumps(results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
