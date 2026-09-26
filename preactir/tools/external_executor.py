from __future__ import annotations

import contextlib
import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from preactir.tools.base import RestorationTool
from preactir.utils.image import load_image, save_image


_EXECUTOR_CACHE: dict[Path, object] = {}


@contextlib.contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_4kagent_executor(root: Path):
    root = root.resolve()
    cached = _EXECUTOR_CACHE.get(root)
    if cached is not None:
        return cached
    if not (root / "executor" / "__init__.py").is_file():
        raise FileNotFoundError(f"Not a 4KAgent/AgenticIR checkout: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    with _working_directory(root):
        module = importlib.import_module("executor")
    backend = module.executor
    _EXECUTOR_CACHE[root] = backend
    return backend


class ExternalExecutorTool(RestorationTool):
    """Adapter for an installed AgenticIR/4KAgent restoration executor.

    The paper action name is namespaced (for example ``noise__restormer``),
    while ``backend_name`` selects the corresponding upstream implementation.
    Running the rollout builder from the 4KAgent conda environment avoids one
    extra Python startup per action. Other environments use an isolated worker.
    """

    def __init__(
        self,
        *,
        name: str,
        target: str,
        subtask: str,
        backend_name: str,
        external_root: str | Path,
        conda_env: str = "4kagent",
        execution_mode: str = "auto",
        cost_prior: float = 1.0,
        output_scale: float = 1.0,
        gpu_id: int | None = None,
    ) -> None:
        if execution_mode not in {"auto", "in_process", "subprocess"}:
            raise ValueError("execution_mode must be auto, in_process, or subprocess")
        self.name = name
        self.targets = (target,)
        self.subtask = subtask
        self.backend_name = backend_name
        self.external_root = Path(external_root).resolve()
        self.conda_env = conda_env
        self.execution_mode = execution_mode
        self.cost_prior = float(cost_prior)
        self.output_scale = float(output_scale)
        self.gpu_id = gpu_id

    def _resolved_mode(self) -> str:
        if self.execution_mode != "auto":
            return self.execution_mode
        return "in_process" if os.environ.get("CONDA_DEFAULT_ENV") == self.conda_env else "subprocess"

    def _invoke_in_process(self, input_dir: Path, output_dir: Path) -> None:
        executor = _load_4kagent_executor(self.external_root)
        toolbox = executor.toolbox_router.get(self.subtask)
        if toolbox is None:
            raise KeyError(f"4KAgent has no subtask '{self.subtask}'")
        matches = [tool for tool in toolbox if tool.tool_name == self.backend_name]
        if len(matches) != 1:
            available = [tool.tool_name for tool in toolbox]
            raise KeyError(
                f"Expected one backend tool '{self.backend_name}' for '{self.subtask}', "
                f"found {len(matches)}; available={available}"
            )
        with _working_directory(self.external_root):
            try:
                matches[0](
                    input_dir=input_dir,
                    output_dir=output_dir,
                    silent=True,
                    run_gpu_id=self.gpu_id,
                )
            except subprocess.CalledProcessError as exc:
                captured = exc.stderr or exc.stdout or ""
                details = "\n".join(captured.splitlines()[-80:])
                raise RuntimeError(
                    f"External tool {self.name} failed with exit code {exc.returncode}:\n{details}"
                ) from exc

    def _invoke_subprocess(self, input_dir: Path, output_dir: Path) -> None:
        worker = Path(__file__).with_name("external_worker.py")
        command = [
            "conda",
            "run",
            "-n",
            self.conda_env,
            "python",
            str(worker),
            "--root",
            str(self.external_root),
            "--subtask",
            self.subtask,
            "--tool",
            self.backend_name,
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ]
        if self.gpu_id is not None:
            command.extend(["--gpu-id", str(self.gpu_id)])
        completed = subprocess.run(
            command,
            cwd=self.external_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            details = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-40:])
            raise RuntimeError(
                f"External tool {self.name} failed with exit code {completed.returncode}:\n{details}"
            )

    def _apply(self, image: np.ndarray, strength: float) -> np.ndarray:
        del strength  # Pretrained checkpoints are discrete actions, not synthetic strengths.
        if not self.external_root.is_dir():
            raise FileNotFoundError(f"External tool root does not exist: {self.external_root}")
        with tempfile.TemporaryDirectory(prefix="preactir_tool_") as temporary:
            temporary_root = Path(temporary)
            input_dir = temporary_root / "input"
            output_dir = temporary_root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            save_image(input_dir / "input.png", image)
            if self._resolved_mode() == "in_process":
                self._invoke_in_process(input_dir, output_dir)
            else:
                self._invoke_subprocess(input_dir, output_dir)
            output_path = output_dir / "output.png"
            if not output_path.is_file():
                candidates = sorted(output_dir.glob("*"))
                raise RuntimeError(f"External tool {self.name} produced no output.png: {candidates}")
            return load_image(output_path, image_size=None)
