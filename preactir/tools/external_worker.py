from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--subtask", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-id", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    from executor import executor

    toolbox = executor.toolbox_router.get(args.subtask, [])
    matches = [tool for tool in toolbox if tool.tool_name == args.tool]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one tool {args.tool!r} for {args.subtask!r}; "
            f"available={[tool.tool_name for tool in toolbox]}"
        )
    try:
        matches[0](
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            silent=True,
            run_gpu_id=args.gpu_id,
        )
    except subprocess.CalledProcessError as exc:
        # The upstream executor captures a model command's combined stdout/stderr
        # in CalledProcessError.output.  Re-emit it here so the parent process can
        # distinguish deterministic input failures from transient CUDA failures.
        captured = exc.stderr or exc.stdout or exc.output or ""
        if captured:
            print(captured, file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
