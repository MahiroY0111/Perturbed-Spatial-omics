#!/usr/bin/env python3
"""Generate untouched M2 data repetitions 11--20 with the frozen rule.

The implementation deliberately delegates generation and all scientific
audits to ``generate_data_seed``.  This wrapper only expands
the accepted seed range and records its own provenance before recomputing the
manifest.  It does not alter a generation parameter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import generate_data_seed as base


FINAL_SEEDS = tuple(range(11, 21))


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
    # The frozen generator is fully seed-general internally; only its historic
    # CLI range was 1--10.  Inject the already validated arguments rather than
    # duplicating or modifying its scientific generation code.
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
            "seed11_20_wrapper_source": str(Path(__file__).resolve()),
            "seed11_20_wrapper_source_sha256": base.sha256(Path(__file__).resolve()),
            "delegated_generator_source": str(Path(base.__file__).resolve()),
            "delegated_generator_source_sha256": base.sha256(Path(base.__file__).resolve()),
            "generation_rule_changed_from_seed1_10": False,
        }
    )
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=base.json_default) + "\n",
        encoding="utf-8",
    )
    # The audit was enriched after base.main() wrote its manifest.
    base.write_manifest(output_root, project_root)


if __name__ == "__main__":
    main()
