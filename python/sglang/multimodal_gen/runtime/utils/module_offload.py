# Module-level offloading for multimodal pipeline models
# Supports RAM offloading at module level for low-end GPUs

from typing import Any, Dict, List, Optional, Set
import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ModuleOffloadManager:
    """
    A module-level offload manager for multimodal pipelines.

    This utility offloads entire modules (like VAE, text_encoder, transformer)
    from GPU to CPU/RAM, and loads them back when needed.
    Designed for low-end GPUs with limited VRAM.

    Typical usage:
    - Construct the manager with the pipeline and module names.
    - Call :meth:`offload_module` to move a module to CPU.
    - Call :meth:`load_module` to move a module back to GPU.
    - Use :meth:`sequential_forward` for automatic load/offload during forward pass.
    """

    def __init__(
        self,
        pin_cpu_memory: bool = True,
        sequential_mode: bool = False,
    ) -> None:
        self.pin_cpu_memory = pin_cpu_memory
        self.sequential_mode = sequential_mode
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.cpu_device = torch.device("cpu")

        # Track which modules are currently on GPU vs CPU
        self._gpu_modules: Set[str] = set()
        self._cpu_modules: Set[str] = set()

        # Store module references
        self._modules: Dict[str, nn.Module] = {}

        # Copy stream for async operations
        self.copy_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        self.enabled = torch.cuda.is_available()

    def register_module(self, name: str, module: nn.Module) -> None:
        """Register a module for offload management."""
        self._modules[name] = module
        # Assume modules start on GPU based on first parameter location
        if module is not None:
            try:
                first_param = next(module.parameters(), None)
                if first_param is not None:
                    if first_param.device.type == 'cuda':
                        self._gpu_modules.add(name)
                    else:
                        self._cpu_modules.add(name)
            except StopIteration:
                pass

    def offload_module(self, name: str, non_blocking: bool = True) -> None:
        """Offload a module from GPU to CPU."""
        if not self.enabled:
            return
        if name not in self._modules:
            return
        if name in self._cpu_modules:
            return

        module = self._modules[name]
        if module is None:
            return

        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

        try:
            # Move module to CPU
            module.to(self.cpu_device, non_blocking=non_blocking)

            # Pin memory for faster GPU transfer later
            if self.pin_cpu_memory:
                for param in module.parameters():
                    if param.data.is_pinned():
                        continue
                    try:
                        param.data = param.data.pin_memory()
                    except Exception:
                        pass  # Some tensors cannot be pinned

            self._cpu_modules.add(name)
            self._gpu_modules.discard(name)
            logger.debug(f"Offloaded module '{name}' to CPU")
        except Exception as e:
            logger.warning(f"Failed to offload module '{name}': {e}")

    def load_module(self, name: str, non_blocking: bool = True) -> None:
        """Load a module from CPU to GPU."""
        if not self.enabled:
            return
        if name not in self._modules:
            return
        if name in self._gpu_modules:
            return

        module = self._modules[name]
        if module is None:
            return

        try:
            if self.copy_stream is not None:
                with torch.cuda.stream(self.copy_stream):
                    module.to(self.device, non_blocking=non_blocking)
                if not non_blocking:
                    torch.cuda.current_stream().wait_stream(self.copy_stream)
            else:
                module.to(self.device, non_blocking=non_blocking)

            self._gpu_modules.add(name)
            self._cpu_modules.discard(name)
            logger.debug(f"Loaded module '{name}' to GPU")
        except Exception as e:
            logger.warning(f"Failed to load module '{name}': {e}")

    def sync(self) -> None:
        """Wait for all async operations to complete."""
        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

    def offload_all_except(self, keep_modules: List[str]) -> None:
        """Offload all registered modules except those in keep_modules."""
        for name in list(self._gpu_modules):
            if name not in keep_modules:
                self.offload_module(name)

    def load_all(self) -> None:
        """Load all registered modules to GPU."""
        for name in list(self._cpu_modules):
            self.load_module(name, non_blocking=False)

    def get_status(self) -> Dict[str, str]:
        """Return the current status of all managed modules."""
        status = {}
        for name in self._modules:
            if name in self._gpu_modules:
                status[name] = "GPU"
            elif name in self._cpu_modules:
                status[name] = "CPU"
            else:
                status[name] = "unknown"
        return status


class MultimodalPipelineOffloadMixin:
    """
    A mixin that provides module-level offloading capabilities for multimodal pipelines.

    Usage:
    - Inherit from this mixin in your pipeline class.
    - Call configure_module_offload() after loading all modules.
    - Use module_context() context manager for automatic load/offload.
    """

    module_offload_manager: Optional[ModuleOffloadManager] = None
    _offload_module_names: List[str] = ["vae", "text_encoder", "transformer"]

    def configure_module_offload(self, server_args: ServerArgs) -> None:
        """Configure module-level offloading based on server args."""
        if not server_args.multimodal_module_offload:
            return

        self.module_offload_manager = ModuleOffloadManager(
            pin_cpu_memory=server_args.pin_cpu_memory,
            sequential_mode=server_args.multimodal_sequential_offload,
        )

        # Register known modules
        for module_name in self._offload_module_names:
            module = getattr(self, module_name, None)
            if module is not None:
                self.module_offload_manager.register_module(module_name, module)

        logger.info(
            f"Configured module offload for {self.__class__.__name__} "
            f"with modules: {self._offload_module_names}"
        )

    def load_module_for_use(self, module_name: str) -> None:
        """Load a module to GPU before using it."""
        if self.module_offload_manager is None:
            return
        self.module_offload_manager.load_module(module_name)
        self.module_offload_manager.sync()

    def offload_module_after_use(self, module_name: str) -> None:
        """Offload a module to CPU after using it."""
        if self.module_offload_manager is None:
            return
        self.module_offload_manager.offload_module(module_name)

    def prepare_for_inference(self, required_modules: List[str]) -> None:
        """
        Prepare for inference by loading required modules and offloading others.

        Args:
            required_modules: List of module names needed for the next operation.
        """
        if self.module_offload_manager is None:
            return

        if self.module_offload_manager.sequential_mode:
            # In sequential mode, offload everything except required modules
            self.module_offload_manager.offload_all_except(required_modules)

        # Load required modules
        for module_name in required_modules:
            self.module_offload_manager.load_module(module_name)

        self.module_offload_manager.sync()


class SequentialModuleExecutor:
    """
    Executes modules sequentially with automatic offloading.
    Useful for low-end GPUs where only one module can fit in VRAM at a time.
    """

    def __init__(
        self,
        modules: Dict[str, nn.Module],
        pin_cpu_memory: bool = True,
    ) -> None:
        self.manager = ModuleOffloadManager(
            pin_cpu_memory=pin_cpu_memory,
            sequential_mode=True,
        )

        for name, module in modules.items():
            self.manager.register_module(name, module)

        # Initially offload all modules to CPU
        for name in modules:
            self.manager.offload_module(name, non_blocking=False)

    def execute(
        self,
        module_name: str,
        forward_fn,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute a forward pass on a specific module.
        Automatically loads the module to GPU, runs forward, and offloads back to CPU.

        Args:
            module_name: Name of the module to execute.
            forward_fn: The forward function to call (typically module.forward or a wrapper).
            *args, **kwargs: Arguments to pass to forward_fn.

        Returns:
            The output of forward_fn.
        """
        # Load module to GPU
        self.manager.load_module(module_name, non_blocking=False)
        self.manager.sync()

        try:
            # Run forward pass
            result = forward_fn(*args, **kwargs)
            return result
        finally:
            # Offload module back to CPU
            self.manager.offload_module(module_name, non_blocking=True)

    def cleanup(self) -> None:
        """Ensure all modules are offloaded and sync is complete."""
        for name in list(self.manager._gpu_modules):
            self.manager.offload_module(name, non_blocking=False)
        self.manager.sync()
