from __future__ import annotations

import json
import sys
from pathlib import Path


def iter_json_objects(text: str):
    decoder = json.JSONDecoder()
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        yield obj
        index = start + end


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: parse_training_log.py LOG_PATH")
    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    for obj in iter_json_objects(text):
        if "epoch" in obj and "valid" in obj:
            valid = obj["valid"]
            valid_losses = obj.get("valid_losses", {})
            print(
                "EPOCH "
                f"{obj['epoch']} "
                f"train={obj.get('train', {}).get('total')} "
                f"valid_loss={valid_losses.get('total')} "
                f"valid_mae={valid.get('mae')} "
                f"valid_corr={valid.get('corr')} "
                f"valid_acc7={valid.get('acc7')}"
            )
        elif "test_at_best" in obj:
            test = obj["test_at_best"]
            print(
                "BEST "
                f"epoch={obj.get('best_epoch')} "
                f"score={obj.get('selection_score')} "
                f"test_mae={test.get('mae')} "
                f"test_corr={test.get('corr')} "
                f"test_acc7={test.get('acc7')}"
            )
        elif "test" in obj and "best_epoch" in obj:
            test = obj["test"]
            print(
                "FINAL "
                f"best_epoch={obj.get('best_epoch')} "
                f"best_valid_loss={obj.get('best_valid_loss')} "
                f"test_mae={test.get('mae')} "
                f"test_corr={test.get('corr')} "
                f"test_acc7={test.get('acc7')}"
            )


if __name__ == "__main__":
    main()
