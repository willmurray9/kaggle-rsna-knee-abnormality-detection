import argparse
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from rsnaknee.config import load_config
from rsnaknee.constants import TARGET_COLUMNS
from rsnaknee.data import audit_metadata, download_metadata, read_metadata, sha256
from rsnaknee.metrics import macro_auc
from rsnaknee.submission import make_constant_submission, validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="RSNA knee competition scaffold")
    parser.add_argument("command", choices=["download", "data", "baselines", "validate"])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--submission")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "download":
        download_metadata(config)
        print("Downloaded five metadata CSVs; no MRI images.")
    elif args.command == "data":
        report = audit_metadata(config)
        print(f"Metadata valid: {report['rows']['train.csv']} studies, {report['fully_labeled_studies']} fully labeled.")
        print(config["artifacts_dir"] / "reports" / "data_audit.md")
    elif args.command == "baselines":
        audit = audit_metadata(config)
        tables = read_metadata(config["raw_dir"])
        submission = make_constant_submission(tables["sample_submission.csv"])
        out = config["artifacts_dir"] / "baselines" / "constant"
        out.mkdir(parents=True, exist_ok=True)
        path = out / "submission.csv"
        submission.to_csv(path, index=False)
        truth = tables["train.csv"][TARGET_COLUMNS].dropna()
        scores = macro_auc(truth, pd.DataFrame(0.5, index=truth.index, columns=TARGET_COLUMNS))
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        manifest = {
            "purpose": "Constant 0.5 pipeline check; no learned model, no cross-validation, no Kaggle submission.",
            "created_at_utc": audit["created_at_utc"],
            "git_commit": revision.stdout.strip() or None,
            "git_dirty": bool(status.stdout.strip()),
            "input_sha256": audit["file_sha256"],
            "config_sha256": sha256(Path(args.config)),
            "submission_sha256": sha256(path),
            "python": platform.python_version(),
            "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn")},
            "labeled_studies": len(truth),
            "sanity_scores": scores,
        }
        (out / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(path)
        print(f"Constant sanity macro AUC: {scores['macro_auc']:.3f}; this is not a trained model.")
    else:
        if not args.submission:
            parser.error("validate requires --submission")
        sample = pd.read_csv(config["raw_dir"] / "sample_submission.csv")
        validate_submission(pd.read_csv(args.submission), sample)
        print("Submission schema, IDs, order and probabilities are valid.")


if __name__ == "__main__":
    main()
