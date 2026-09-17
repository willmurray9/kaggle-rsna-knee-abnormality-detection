import ast
import hashlib
import json
from pathlib import Path
import sys

import pytest

from rsnaknee.feature_notebook import build_feature_notebook


def test_builder_embeds_exact_sources_and_compilable_offline_cells(tmp_path: Path) -> None:
    output = tmp_path / "build"
    build_feature_notebook(output)
    notebook = json.loads((output / "features.ipynb").read_text())
    assert [cell["cell_type"] for cell in notebook["cells"]] == ["markdown", "code", "code"]
    bootstrap = "".join(notebook["cells"][1]["source"])
    sources_node = next(node for node in ast.parse(bootstrap).body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "sources" for target in node.targets))
    sources = ast.literal_eval(sources_node.value)
    assert set(sources) == {"__init__.py", "imaging.py", "features.py"}
    package = Path(__file__).resolve().parents[1] / "src" / "rsnaknee"
    for name, source in sources.items():
        assert source == (package / name).read_text()
        compile(source, name, "exec")
    for cell in notebook["cells"][1:]:
        compile("".join(cell["source"]), "cell", "exec")
        assert cell["outputs"] == [] and cell["execution_count"] is None
    runtime = "".join(notebook["cells"][2]["source"])
    assert ".head(4)" in runtime
    assert "f'Sample {row+1}: {plane}'" in runtime
    assert "batch_size=12,workers=8,device='cuda'" in runtime
    manifest = json.loads((output / "build_manifest.json").read_text())
    assert manifest["source_sha256"] == {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
    assert manifest["notebook_sha256"] == hashlib.sha256((output / "features.ipynb").read_bytes()).hexdigest()


def test_metadata_pins_private_offline_t4_and_only_reviewed_mounts(tmp_path: Path) -> None:
    output = tmp_path / "build"
    build_feature_notebook(output, kernel_id="example/custom-feature-run")
    metadata = json.loads((output / "kernel-metadata.json").read_text())
    assert metadata["id"] == "example/custom-feature-run"
    assert metadata["is_private"] is True and metadata["enable_internet"] is False
    assert metadata["enable_gpu"] is True and metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["competition_sources"] == ["rsna-knee-abnormality-detection"]
    assert metadata["model_sources"] == ["metaresearch/dinov2/PyTorch/small/1"]
    assert metadata["dataset_sources"] == [] and metadata["kernel_sources"] == []


def test_bootstrap_materializes_sources_without_loading_model_or_running_extraction(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "build"
    build_feature_notebook(output)
    notebook = json.loads((output / "features.ipynb").read_text())
    work = tmp_path / "working"
    work.mkdir()
    bootstrap = "".join(notebook["cells"][1]["source"]).replace("/kaggle/working", str(work))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    namespace = {}
    exec(compile(bootstrap, "bootstrap", "exec"), namespace)
    assert namespace["os"].environ["HF_HUB_OFFLINE"] == "1"
    assert namespace["os"].environ["TRANSFORMERS_OFFLINE"] == "1"
    assert {path.name for path in (work / "rsnaknee").iterdir()} == {"__init__.py", "features.py", "imaging.py"}
    assert not (work / "features").exists()


def test_builder_refuses_existing_directory(tmp_path: Path) -> None:
    output = tmp_path / "build"
    output.mkdir()
    marker = output / "existing.txt"
    marker.write_text("preserve previous build")
    with pytest.raises(FileExistsError):
        build_feature_notebook(output)
    assert list(output.iterdir()) == [marker]
