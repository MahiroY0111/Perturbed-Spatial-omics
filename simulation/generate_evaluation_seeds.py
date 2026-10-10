#!/usr/bin/env python3
"""Generate the data seeds of the final evaluation (21-30) with the fixed generation rule of generation_config.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import generate_data_seed as base


FINAL_SEEDS = tuple(range(21, 31))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-seed", type=int, choices=FINAL_SEEDS, required=True)
    parser.add_argument("--template-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base.parse_args = lambda: args
    base.main()

    output_root = args.output_root.resolve()
    project_root = args.project_root.resolve()
    audit_path = output_root / "final_generation_seed_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.update(
        {
            "final_untouched_evaluation_seed": True,
            "allowed_final_seed_range": list(FINAL_SEEDS),
            "seed21_30_wrapper_source": str(Path(__file__).resolve()),
            "seed21_30_wrapper_source_sha256": base.sha256(Path(__file__).resolve()),
            "delegated_generator_source": str(Path(base.__file__).resolve()),
            "delegated_generator_source_sha256": base.sha256(Path(base.__file__).resolve()),
            "generation_rule_changed_from_seed1_20": False,
            "generation_rule_changed_from_seed1_10": False,
            "seeds_1_3_role": "development",
            "seeds_11_13_role": "posthoc sensitivity",
            "seeds_21_30_role": "previously unused final holdout evaluation",
        }
    )
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=base.json_default) + "\n",
        encoding="utf-8",
    )
    base.write_manifest(output_root, project_root)


if __name__ == "__main__":
    main()
