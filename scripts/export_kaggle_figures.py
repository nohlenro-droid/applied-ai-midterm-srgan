"""Run in your own Kaggle notebook to extract shareable figures/results.

This utility never uploads models; it only copies figure files and
non-sensitive CSV/JSON metrics into a zip archive in /kaggle/working.
Adjust SOURCE_ROOT to your mounted checkpoint dataset.
"""
from pathlib import Path
import shutil
import zipfile

SOURCE_ROOT = Path(
    "/kaggle/input/datasets/niklasohlenroth/"
    "applied-ai-midterm-srgan-checkpoints/project"
)
DEST = Path("/kaggle/working/midterm_public_figures_and_results")

if not SOURCE_ROOT.exists():
    raise FileNotFoundError(
        "Mount the latest private Kaggle checkpoint dataset, or change SOURCE_ROOT."
    )

if DEST.exists():
    shutil.rmtree(DEST)

count = 0
for branch in [
    "results/classifier_comparison",
    "results/srgan_quality",
    "results/training_diagnostics"
]:
    source = SOURCE_ROOT / branch
    if not source.exists():
        print("Not found:", source)
        continue
    for item in source.rglob("*"):
        if not item.is_file():
            continue
        if item.suffix.lower() not in {".pdf", ".svg", ".png", ".json", ".csv"}:
            continue
        target = DEST / item.relative_to(SOURCE_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        count += 1

assert count > 0, "No results found in source checkpoint dataset."
archive = Path("/kaggle/working/midterm_public_figures_and_results.zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for file in DEST.rglob("*"):
        if file.is_file():
            z.write(file, file.relative_to(DEST))
print("Exported", count, "figures/results to:", archive)
print("Review files before publishing any individual predictions.")
