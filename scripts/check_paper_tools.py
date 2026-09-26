#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from preactir.config import load_config
from preactir.tools.paper_registry import inspect_paper_backend, paper_tool_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the paper restoration toolbox without GPU inference.")
    parser.add_argument("--config", default="configs/preactir_mio100.yaml")
    parser.add_argument("--require-assets", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    profile = str(cfg.tools.get("profile", "agenticir_default"))
    expected = paper_tool_names(profile)
    configured = list(cfg.tools.names)
    report = inspect_paper_backend(str(cfg.tools.external_root), profile)
    report["configured_tool_count"] = len(configured)
    report["missing_from_config"] = sorted(set(expected) - set(configured))
    report["unknown_in_config"] = sorted(set(configured) - set(expected))
    print(json.dumps(report, indent=2))
    invalid = report["missing_tools"] or report["missing_from_config"] or report["unknown_in_config"]
    if args.require_assets:
        invalid = invalid or report["missing_lazy_assets"]
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
