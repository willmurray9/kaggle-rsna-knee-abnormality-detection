"""Build the private offline feature notebook locally; never upload or execute it.

Running the notebook requires a separate, explicitly authorized Kaggle API/CLI
upload. The builder embeds the exact current preprocessing and extraction sources.
"""

import argparse
import hashlib
import json
from pathlib import Path


RUNTIME = '''from rsnaknee.features import extract_features, prepare_study
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
roots=[Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')]
root=next(p for p in roots if (p/'train.csv').is_file())
checkpoints=[]
for directory, dirs, files in os.walk('/kaggle/input'):
    dirs[:]=[d for d in dirs if d not in ('train_series','test_series')]
    if 'config.json' in files and 'dinov2' in directory.lower():
        config=json.loads((Path(directory)/'config.json').read_text())
        if config.get('model_type')=='dinov2' and config.get('hidden_size')==384:
            checkpoints.append(Path(directory))
if len(checkpoints)!=1:
    raise RuntimeError('Expected one generic DINOv2-small checkpoint')
output=Path('/kaggle/working/features')
manifest=extract_features(root,checkpoints[0],output,batch_size=12,workers=8,device='cuda')
# Bounded preprocessing contact sheet, without report text or patient identifiers.
ids=pd.read_csv(root/'train.csv',usecols=['StudyInstanceUID']).head(4)
series=pd.read_csv(root/'train_series.csv')
canvas=Image.new('RGB',(3*224,4*248),'black')
draw=ImageDraw.Draw(canvas)
for row,study in enumerate(ids.StudyInstanceUID):
    prepared=prepare_study(root,'train',study,series.loc[series.StudyInstanceUID.eq(study)].to_dict('records'))
    for column,plane in enumerate(('Sagittal','Coronal','Axial')):
        tile=Image.fromarray((prepared['images'][column,1]*255).clip(0,255).astype('uint8')).convert('RGB')
        canvas.paste(tile,(column*224,row*248))
        draw.text((column*224+4,row*248+226),f'Sample {row+1}: {plane}',fill='white')
canvas.save(output/'preprocessing_contact_sheet.png')
print(json.dumps({'status':manifest['status'],'completed_studies':manifest['completed_studies'],'elapsed_seconds':manifest['elapsed_seconds']},indent=2))
'''


def build_feature_notebook(
    output: Path, kernel_id: str = "willmurray99/rsna-knee-frozen-image-features"
) -> None:
    """Write an unexecuted notebook, mount metadata and hashes into a new directory."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Feature notebook build directory already exists")
    package = Path(__file__).parent
    sources = {name: (package / name).read_text() for name in ("__init__.py", "imaging.py", "features.py")}
    bootstrap = (
        'import os, sys, json\nfrom pathlib import Path\n'
        'os.environ["HF_HUB_OFFLINE"]="1"\n'
        'os.environ["TRANSFORMERS_OFFLINE"]="1"\n'
        f'sources={sources!r}\n'
        'package=Path("/kaggle/working/rsnaknee")\n'
        'package.mkdir(exist_ok=True)\n'
        'for name,source in sources.items():\n'
        '    (package/name).write_text(source)\n'
        'sys.path.insert(0,"/kaggle/working")\n'
    )
    cells = []
    for kind, source in [
        ("markdown", "# Frozen DINOv2 MRI features\nPrivate offline experiment. Generic pretrained model only; "
         "no competition-trained weights, report input or label fitting. Three planes, central20%-80% three "
         "slices,224pixels. Own source embedded to make execution reproducible.\n"),
        ("code", bootstrap), ("code", RUNTIME),
    ]:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            compile(source, "feature-notebook-cell", "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {
        "cells": cells, "nbformat": 4, "nbformat_minor": 4,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    }
    metadata = {
        "id": kernel_id, "title": "RSNA Knee Frozen Image Features", "code_file": "features.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["rsna-knee-abnormality-detection"], "dataset_sources": [],
        "model_sources": ["metaresearch/dinov2/PyTorch/small/1"], "kernel_sources": [],
    }
    output.mkdir(parents=True)
    notebook_path = output / "features.ipynb"
    metadata_path = output / "kernel-metadata.json"
    notebook_path.write_text(json.dumps(notebook, indent=2) + "\n")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = {
        "source_sha256": {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()},
        "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        "kernel_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "execution": "Build only. Upload and execution require a separate authorized Kaggle API/CLI call.",
    }
    (output / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-id", default="willmurray99/rsna-knee-frozen-image-features")
    args = parser.parse_args()
    build_feature_notebook(args.output, args.kernel_id)


if __name__ == "__main__":
    main()
