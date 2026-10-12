"""Download the two full source datasets via kagglehub, in the background.

Datasets (see notebook):
  * Animal-90:  iamsouravbanerjee/animal-image-dataset-90-different-animals
  * Wild Animals: whenamancodes/wild-animals-images

Writes a log and a DONE marker so an external watcher can tell when finished.
Run detached (not in the foreground) — downloads can take several minutes.
"""

import sys
import time
from pathlib import Path

import kagglehub

DATASETS = [
    ("animal90", "iamsouravbanerjee/animal-image-dataset-90-different-animals"),
    ("wild_animals", "whenamancodes/wild-animals-images"),
]

LOG = Path(__file__).resolve().parent / "_download.log"
DONE = Path(__file__).resolve().parent / "_download.done"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    if DONE.exists():
        DONE.unlink()
    LOG.write_text("", encoding="utf-8")

    results = {}
    for name, slug in DATASETS:
        log(f"START {name}: {slug}")
        try:
            path = kagglehub.dataset_download(slug)
            results[name] = path
            log(f"OK {name} -> {path}")
        except Exception as exc:  # noqa: BLE001
            log(f"FAIL {name}: {exc!r}")
            results[name] = None

    log("SUMMARY " + repr(results))
    DONE.write_text("\n".join(f"{k}={v}" for k, v in results.items()), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
