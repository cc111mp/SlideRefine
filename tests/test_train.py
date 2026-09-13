import csv
from pathlib import Path
import numpy as np
import pytest
from tsclahe.train import Record, validate_split, read_manifest


def test_group_overlap_rejected():
    r1 = Record(Path("a"), Path("b"), Path("c"), "patient1")
    r2 = Record(Path("d"), Path("e"), Path("f"), "patient1")
    with pytest.raises(ValueError, match="group_id"):
        validate_split([r1], [r2])


def test_image_overlap_rejected():
    r1 = Record(Path("a"), Path("b"), Path("c"), "patient1")
    r2 = Record(Path("a"), Path("e"), Path("f"), "patient2")
    with pytest.raises(ValueError, match="paths"):
        validate_split([r1], [r2])


def test_empty_manifest_rejected(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("input,target,mask,group_id\n")
    with pytest.raises(ValueError, match="empty"):
        read_manifest(path)


def test_training_smoke(tmp_path, config):
    pytest.importorskip("torch")
    from tsclahe.io import write_image
    from tsclahe.train import main
    import json
    y, x = np.mgrid[:64, :64]
    for split, phase in (("train", 0), ("val", 1)):
        image = (.2 + .06 * np.sin(x / 4 + phase) * np.cos(y / 6)).astype(np.float32)
        target = image + .06 * np.sin(np.pi * image)
        mask = np.ones_like(image)
        for name, array in (("input", image), ("target", target), ("mask", mask)):
            write_image(tmp_path / f"{split}_{name}.tif", array)
        with (tmp_path / f"{split}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["input", "target", "mask", "group_id"])
            writer.writerow([f"{split}_input.tif", f"{split}_target.tif", f"{split}_mask.tif", split])
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(config.to_dict()))
    out = tmp_path / "training"
    assert main(["--train", str(tmp_path / "train.csv"), "--val", str(tmp_path / "val.csv"),
                 "--output-dir", str(out), "--epochs", "2", "--config", str(config_path),
                 "--limits", "0", "1", "--threads", "1"]) == 0
    assert (out / "best.pt").exists() and (out / "last.pt").exists()
    history = json.loads((out / "history.json").read_text())
    assert len(history) == 2 and np.isfinite(history[-1]["train_loss"])
