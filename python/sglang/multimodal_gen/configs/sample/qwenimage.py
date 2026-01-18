# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class QwenImageSamplingParams(SamplingParams):
    negative_prompt: str = " "
    num_frames: int = 1
    # Denoising stage
    guidance_scale: float = 4.0
    num_inference_steps: int = 50


@dataclass
class QwenImage2512SamplingParams(QwenImageSamplingParams):
    negative_prompt: str = (
        "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
    )


@dataclass
class QwenImageEditPlusSamplingParams(QwenImageSamplingParams):
    # Denoising stage
    guidance_scale: float = 4.0
    # true_cfg_scale: float = 4.0
    num_inference_steps: int = 40


@dataclass
class QwenImageLayeredSamplingParams(QwenImageSamplingParams):
    # num_frames corresponds to DiffSynth's layer_num parameter
    # layer_num=0 generates 1 output, layer_num=1 generates 2 outputs, etc.
    num_frames: int = 0  # Default to layer_num=0 (single image output)
    height: int = 640
    width: int = 640
    prompt: str = " "
    negative_prompt: str = " "

    guidance_scale: float = 4.0
    num_inference_steps: int = 50
    cfg_normalize: bool = True
    use_en_prompt: bool = True

    def __post_init__(self) -> None:
        # Override base class validation to allow num_frames=0 (layer_num=0)
        # For Qwen-Image-Layered, num_frames represents layer_num which can be 0
        if self.num_frames < 0:
            raise ValueError(
                f"num_frames must be >= 0 for QwenImageLayered, got {self.num_frames!r}"
            )

        if self.width is None:
            self.width_not_provided = True
        if self.height is None:
            self.height_not_provided = True

        # Skip base class _validate() for num_frames, but keep other validations
        self._validate_layered()

    def _validate_layered(self):
        """Validation specific to QwenImageLayered (allows num_frames=0)."""
        if self.prompt_path and not self.prompt_path.endswith(".txt"):
            raise ValueError(
                f"prompt_path must be a txt file, got {self.prompt_path!r}"
            )

        if (
            not isinstance(self.num_outputs_per_prompt, int)
            or self.num_outputs_per_prompt <= 0
        ):
            raise ValueError(
                f"num_outputs_per_prompt must be a positive int, got {self.num_outputs_per_prompt!r}"
            )

        if not isinstance(self.fps, int) or self.fps <= 0:
            raise ValueError(f"fps must be a positive int, got {self.fps!r}")

        if self.num_inference_steps is not None:
            if (
                not isinstance(self.num_inference_steps, int)
                or self.num_inference_steps <= 0
            ):
                raise ValueError(
                    f"num_inference_steps must be a positive int, got {self.num_inference_steps!r}"
                )

    def _adjust(self, server_args):
        """Override to allow num_frames=0 for QwenImageLayered (layer_num=0)."""
        from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

        logger = init_logger(__name__)

        pipeline_config = server_args.pipeline_config
        if not isinstance(self.prompt, str):
            raise TypeError(f"`prompt` must be a string, but got {type(self.prompt)}")

        self.data_type = server_args.pipeline_config.task_type.data_type()

        if server_args.output_path is not None:
            self.output_path = server_args.output_path
            logger.debug(
                f"Overriding output_path with server configuration: {self.output_path}"
            )

        # Process negative prompt
        if self.negative_prompt is not None and not self.negative_prompt.isspace():
            self.negative_prompt = self.negative_prompt.strip()

        # For QwenImageLayered, num_frames can be 0 (layer_num=0)
        if self.num_frames < 0:
            raise ValueError(
                f"num_frames must be >= 0 for QwenImageLayered, got "
                f"height={self.height}, width={self.width}, "
                f"num_frames={self.num_frames}"
            )

        # Validate resolution against pipeline-specific supported resolutions
        if self.height is None and self.width is None:
            if self.supported_resolutions is not None:
                self.width, self.height = self.supported_resolutions[0]
                logger.info(
                    f"Resolution unspecified, using default: {self.supported_resolutions[0]}"
                )

        if self.height is not None and self.width is not None:
            if self.supported_resolutions is not None:
                if (self.width, self.height) not in self.supported_resolutions:
                    supported_str = ", ".join(
                        [f"{w}x{h}" for w, h in self.supported_resolutions]
                    )
                    error_msg = (
                        f"Unsupported resolution: {self.width}x{self.height}, output quality may suffer. "
                        f"Supported resolutions: {supported_str}"
                    )
                    logger.warning(error_msg)
