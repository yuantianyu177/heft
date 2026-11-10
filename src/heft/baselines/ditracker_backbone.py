"""Trainable DiTracker backbone readouts with frozen VAE and LoRA attention."""

import importlib
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import resnet18

from heft.attn_hook import AttentionProcessorKind
from heft.extraction import ModelConfig

from .ditracker import DiTrackerCost, DiTrackerRefiner


class LoRALinear(nn.Module):
    """Wrap a frozen linear layer with a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int = 128):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base.requires_grad_(False)
        kwargs = {"device": base.weight.device, "dtype": torch.float32}
        self.down = nn.Linear(base.in_features, rank, bias=False, **kwargs)
        self.up = nn.Linear(rank, base.out_features, bias=False, **kwargs)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: Tensor) -> Tensor:
        frozen = self.base(value)
        update = self.up(self.down(value.to(self.down.weight.dtype)))
        return frozen + update.to(frozen.dtype)


class DiTrackerBackbone(nn.Module):
    """Read the chosen Q/K head channels at the final denoising time."""

    def __init__(
        self,
        model: ModelConfig,
        checkpoint: Path,
        *,
        device: torch.device,
        layer: int,
        heads: tuple[int, ...],
        rank: int = 128,
        chunk: int = 25,
        microbatch: int = 4,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        if (
            model.processor
            not in (AttentionProcessorKind.WAN, AttentionProcessorKind.COGVIDEOX)
            or model.temporal_patch_size != 1
            or chunk < 2
            or microbatch < 1
            or layer < 0
            or not heads
            or min(heads) < 0
            or len(set(heads)) != len(heads)
        ):
            raise ValueError("Invalid DiTracker backbone or attention readout")
        self.model, self.layer, self.chunk, self.microbatch = (
            model,
            layer,
            chunk,
            microbatch,
        )
        self.wan = model.processor == AttentionProcessorKind.WAN
        pipeline = getattr(
            importlib.import_module(model.pipeline_module), model.pipeline_class
        )
        self.pipe = pipeline.from_pretrained(
            str(checkpoint), dtype=model.dtype, local_files_only=True
        )
        self.vae = self.pipe.vae.to(device).requires_grad_(False).eval()
        self.transformer = self.pipe.transformer.to(device).requires_grad_(False).eval()
        blocks = getattr(self.transformer, model.blocks_attribute)
        if layer >= len(blocks):
            raise ValueError(f"Layer {layer} is outside the DiT")
        del blocks[layer + 1 :]
        attention = getattr(blocks[layer], model.attention_attribute)
        if max(heads) >= attention.heads:
            raise ValueError("Selected DiTracker head does not exist")
        self.register_buffer(
            "heads", torch.tensor(heads, device=device), persistent=False
        )
        for block in blocks:
            attn = getattr(block, model.attention_attribute)
            for projection in ("to_q", "to_k", "to_v"):
                setattr(attn, projection, LoRALinear(getattr(attn, projection), rank))
            attn.to_out[0] = LoRALinear(attn.to_out[0], rank)
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        self.pipe.text_encoder.to(device)
        with torch.no_grad():
            prompt, _ = self.pipe.encode_prompt(
                "",
                do_classifier_free_guidance=False,
                device=device,
                max_sequence_length=512 if self.wan else 226,
            )
            self.register_buffer(
                "prompt", prompt.to(self.transformer.dtype), persistent=False
            )
        self.pipe.text_encoder.to("cpu")
        self.pipe.scheduler.set_timesteps(model.num_inference_steps, device=device)
        self.register_buffer(
            "timestep",
            self.pipe.scheduler.timesteps[model.start_step],
            persistent=False,
        )
        self.grid = tuple(
            dim // stride
            for dim, stride in zip(
                model.resolution, model.feature_spatial_stride, strict=True
            )
        )
        if self.wan:
            mean = torch.tensor(self.vae.config.latents_mean, device=device).view(
                1, 1, -1, 1, 1
            )
            std = torch.tensor(self.vae.config.latents_std, device=device).view(
                1, 1, -1, 1, 1
            )
            self.register_buffer("mean", mean, persistent=False)
            self.register_buffer("std", std, persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        return self

    @torch.no_grad()
    def _frame_latents(self, pixels: Tensor) -> Tensor:
        batch, frames = pixels.shape[:2]
        encoded = []
        for rows in pixels.flatten(0, 1).split(self.microbatch):
            sample = self.vae.encode(
                rows[:, :, None].to(self.vae.dtype)
            ).latent_dist.mode()
            if sample.shape[2] != 1:
                raise ValueError(
                    "Independent frame encodes must produce one latent each"
                )
            encoded.append(sample[:, :, 0])
        values = torch.cat(encoded).reshape(batch, frames, *encoded[0].shape[1:])
        if self.wan:
            return ((values.float() - self.mean) / self.std).to(self.transformer.dtype)
        return values * self.pipe.vae_scaling_factor_image

    def _read(self, latents: Tensor) -> tuple[Tensor, Tensor]:
        batch, time = latents.shape[:2]
        blocks = getattr(self.transformer, self.model.blocks_attribute)
        attn = getattr(blocks[self.layer], self.model.attention_attribute)
        if getattr(attn, "_heft_capture", None) is not None:
            raise RuntimeError("The selected attention already has a capture callback")
        captured: list[tuple[Tensor, Tensor]] = []

        def readout(*, query, key, **_):
            captured.append((query, key))

        attn._heft_capture = readout
        try:
            rotary = {}
            if self.wan:
                sample = latents.transpose(1, 2).contiguous()
            else:
                sample = self.pipe.scheduler.scale_model_input(latents, self.timestep)
                if self.transformer.config.use_rotary_positional_embeddings:
                    rotary["image_rotary_emb"] = (
                        self.pipe._prepare_rotary_positional_embeddings(
                            *self.model.resolution, time, latents.device
                        )
                    )
            self.transformer(
                hidden_states=sample.to(self.transformer.dtype),
                timestep=self.timestep.expand(batch),
                encoder_hidden_states=self.prompt.expand(batch, -1, -1),
                return_dict=False,
                **rotary,
            )
        finally:
            delattr(attn, "_heft_capture")
        if len(captured) != 1:
            raise ValueError("DiT readout must return exactly one Q/K pair")
        return tuple(
            value.index_select(1, self.heads)
            .reshape(batch, len(self.heads), time, *self.grid, value.shape[-1])
            .permute(0, 2, 1, 5, 3, 4)
            for value in captured[0]
        )

    def forward(self, pixels: Tensor) -> tuple[Tensor, Tensor]:
        latents = self._frame_latents(pixels)
        frames = latents.shape[1]
        q0, k0 = self._read(latents[:, : self.chunk])
        queries, keys = [q0], [k0]
        for start in range(self.chunk, frames, self.chunk - 1):
            clip = torch.cat(
                (latents[:, :1], latents[:, start : start + self.chunk - 1]), dim=1
            )
            query, key = self._read(clip)
            queries.append(query[:, 1:])
            keys.append(key[:, 1:])
        return torch.cat(queries, dim=1), torch.cat(keys, dim=1)


class DiTracker(nn.Module):
    """Joint LoRA, framewise ResNet and iterative tracking model."""

    def __init__(self, backbone: DiTrackerBackbone):
        super().__init__()
        self.backbone = backbone
        rgb = resnet18(weights=None)
        self.resnet = nn.Sequential(
            rgb.conv1, rgb.bn1, rgb.relu, rgb.maxpool, rgb.layer1
        )
        self.refiner = DiTrackerRefiner(DiTrackerCost())

    def forward(
        self,
        rgb: Tensor,
        anchors: Tensor,
        query_times: Tensor,
        *,
        iterations: int = 6,
        query_batch: int | None = None,
    ) -> list[tuple[Tensor, Tensor, Tensor]]:
        if (
            rgb.ndim != 5
            or rgb.shape[2] != 3
            or not rgb.shape[1]
            or rgb.dtype != torch.uint8
        ):
            raise ValueError("Expected uint8 [batch, time, 3, height, width] RGB")
        if query_batch is not None and query_batch < 1:
            raise ValueError("query_batch must be positive")
        batch, time = rgb.shape[:2]
        shape = self.backbone.model.resolution
        pixels = F.interpolate(
            rgb.flatten(0, 1).float(), size=shape, mode="bilinear", align_corners=False
        )
        pixels = pixels / 127.5 - 1.0
        query, key = self.backbone(pixels.reshape(batch, time, 3, *shape))
        encoded = (
            self.resnet(pixels)
            if self.training
            else torch.cat([self.resnet(batch) for batch in pixels.split(16)])
        )
        features = encoded.reshape(batch, time, 1, *encoded.shape[1:])
        if query_batch is None or anchors.shape[1] <= query_batch:
            return self.refiner(
                query, key, features, anchors, query_times, iterations=iterations
            )
        groups = [
            self.refiner(query, key, features, xy, when, iterations=iterations)
            for xy, when in zip(
                anchors.split(query_batch, dim=1),
                query_times.split(query_batch, dim=1),
                strict=True,
            )
        ]
        return [
            tuple(torch.cat(values, dim=2) for values in zip(*step, strict=True))
            for step in zip(*groups, strict=True)
        ]

    def trainable_state(self) -> dict[str, Tensor]:
        names = {name for name, param in self.named_parameters() if param.requires_grad}
        names.update(
            name for name, _ in self.named_buffers() if name.startswith("resnet.")
        )
        return {
            name: value.detach().cpu().contiguous()
            for name, value in self.state_dict().items()
            if name in names
        }
