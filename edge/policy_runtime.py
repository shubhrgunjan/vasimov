"""
vasimov/edge/policy_runtime.py
Modular runtime abstraction for policy execution (ONNX, PyTorch, custom).
"""

from __future__ import annotations
import abc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

log = logging.getLogger("vasimov.edge.policy_runtime")


class BasePolicyRuntime(abc.ABC):
    """Abstract base class for policy inference runtimes."""

    @abc.abstractmethod
    def load(self, model_path: Path, options: Optional[Dict[str, Any]] = None) -> None:
        """Load model weights / graph into memory."""
        pass

    @abc.abstractmethod
    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute inference step given named input tensors."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Release runtime resources."""
        pass

    @property
    @abc.abstractmethod
    def input_names(self) -> List[str]:
        pass

    @property
    @abc.abstractmethod
    def output_names(self) -> List[str]:
        pass


class ONNXPolicyRuntime(BasePolicyRuntime):
    """ONNX Runtime implementation for Asimov 1 policy inference."""

    def __init__(self, model_path: Optional[Path] = None, options: Optional[Dict[str, Any]] = None):
        self.session = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._input_shapes: Dict[str, List[Any]] = {}
        self._output_shapes: Dict[str, List[Any]] = {}
        if model_path is not None:
            self.load(model_path, options)

    def load(self, model_path: Path, options: Optional[Dict[str, Any]] = None) -> None:
        import onnxruntime as ort

        path = Path(model_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        if options:
            for k, v in options.items():
                if hasattr(opts, k):
                    setattr(opts, k, v)

        self.session = ort.InferenceSession(
            str(path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self._input_names = [inp.name for inp in self.session.get_inputs()]
        self._output_names = [out.name for out in self.session.get_outputs()]
        self._input_shapes = {inp.name: inp.shape for inp in self.session.get_inputs()}
        self._output_shapes = {out.name: out.shape for out in self.session.get_outputs()}
        log.info("[RUNTIME] Loaded ONNX policy from %s (inputs: %s, outputs: %s)", path.name, self._input_names, self._output_names)

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.session is None:
            raise RuntimeError("ONNXPolicyRuntime not loaded. Call load() first.")

        feed = {}
        for name in self._input_names:
            if name in inputs:
                feed[name] = inputs[name]
            elif len(inputs) == 1 and len(self._input_names) == 1:
                # Fallback to single unnamed tensor
                feed[name] = next(iter(inputs.values()))
            else:
                raise KeyError(f"Missing required input tensor '{name}'. Provided: {list(inputs.keys())}")

        outputs = self.session.run(self._output_names, feed)
        return {name: arr for name, arr in zip(self._output_names, outputs)}

    def close(self) -> None:
        self.session = None

    @property
    def input_names(self) -> List[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> List[str]:
        return list(self._output_names)

    @property
    def input_shapes(self) -> Dict[str, List[Any]]:
        return dict(self._input_shapes)

    @property
    def output_shapes(self) -> Dict[str, List[Any]]:
        return dict(self._output_shapes)


def create_policy_runtime(runtime_spec: Any) -> BasePolicyRuntime:
    """Factory creating appropriate policy runtime from specification."""
    rtype = getattr(runtime_spec, "type", "onnx").lower()
    if rtype == "onnx":
        return ONNXPolicyRuntime()
    else:
        raise ValueError(f"Unsupported policy runtime type: '{rtype}'")
