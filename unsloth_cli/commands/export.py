# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import platform
from pathlib import Path
from typing import List, Optional

import typer

from unsloth_cli._studio_deps import studio_backend_imports


EXPORT_FORMATS = ["merged-16bit", "merged-4bit", "gguf", "lora"]
ADAPTER_FORMATS = ["mlx", "peft"]


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


GGUF_QUANTS = ["q4_k_m", "q5_k_m", "q8_0", "f16"]


def list_checkpoints(
    outputs_dir: Path = typer.Option(
        Path("./outputs"), "--outputs-dir", help = "Directory that holds training runs."
    ),
):
    """List checkpoints detected in the outputs directory."""
    with studio_backend_imports("unsloth list-checkpoints"):
        from studio.backend.core.export import ExportBackend

    backend = ExportBackend()
    checkpoints = backend.scan_checkpoints(outputs_dir = str(outputs_dir))
    if not checkpoints:
        typer.echo("No checkpoints found.")
        raise typer.Exit()

    for model_name, ckpt_list, metadata in checkpoints:
        typer.echo(f"\n{model_name}:")
        for display, path, loss in ckpt_list:
            loss_str = f" (loss: {loss:.4f})" if loss is not None else ""
            typer.echo(f"  {display}{loss_str}: {path}")


def export(
    checkpoint: Path = typer.Argument(..., help = "Path to checkpoint directory."),
    output_dir: Path = typer.Argument(..., help = "Directory to save exported model."),
    format: str = typer.Option(
        "merged-16bit",
        "--format",
        "-f",
        help = f"Export format: {', '.join(EXPORT_FORMATS)}",
    ),
    quantization: str = typer.Option(
        "q4_k_m",
        "--quantization",
        "-q",
        help = f"GGUF quantization method: {', '.join(GGUF_QUANTS)}",
    ),
    push_to_hub: bool = typer.Option(
        False, "--push-to-hub", help = "Push exported model to HuggingFace Hub."
    ),
    repo_id: Optional[str] = typer.Option(
        None, "--repo-id", help = "HuggingFace repo ID (username/model-name)."
    ),
    hf_token: Optional[str] = typer.Option(
        None,
        "--hf-token",
        envvar = "HF_TOKEN",
        help = "HuggingFace token, for gated or private checkpoints and Hub pushes.",
    ),
    private: bool = typer.Option(False, "--private", help = "Make the HuggingFace repo private."),
    max_seq_length: int = typer.Option(2048, "--max-seq-length"),
    load_in_4bit: Optional[bool] = typer.Option(
        None,
        "--load-in-4bit/--no-load-in-4bit",
        help = "Load the checkpoint in 4-bit. Default: 16-bit for an unquantized full model "
        "(except with merged-4bit), 4-bit otherwise.",
    ),
    adapter_format: Optional[str] = typer.Option(
        None,
        "--adapter-format",
        help = "LoRA adapter on-disk format (mlx or peft); omitted writes the "
        "platform's native format. Apple-silicon hosts only.",
        # Registered everywhere so off-Mac use gets a clear error.
        hidden = not _is_apple_silicon(),
    ),
):
    """Export a checkpoint to various formats (merged, GGUF, LoRA adapter)."""
    if format not in EXPORT_FORMATS:
        typer.echo(
            f"Error: Invalid format '{format}'. Choose from: {', '.join(EXPORT_FORMATS)}",
            err = True,
        )
        raise typer.Exit(code = 2)

    if push_to_hub and not repo_id:
        typer.echo("Error: --repo-id required when using --push-to-hub", err = True)
        raise typer.Exit(code = 2)

    if adapter_format is not None:
        if adapter_format not in ADAPTER_FORMATS:
            typer.echo(
                f"Error: Invalid adapter format '{adapter_format}'. "
                f"Choose from: {', '.join(ADAPTER_FORMATS)}",
                err = True,
            )
            raise typer.Exit(code = 2)
        if format != "lora":
            typer.echo(
                "Error: --adapter-format only applies to --format lora",
                err = True,
            )
            raise typer.Exit(code = 2)
        if not _is_apple_silicon():
            typer.echo(
                "Error: --adapter-format is only available on Apple-silicon "
                "hosts; this platform exports the native PEFT format.",
                err = True,
            )
            raise typer.Exit(code = 2)

    # A full model is saved as loaded, so merged-4bit still needs the 4-bit load.
    if load_in_4bit is None and format == "merged-4bit":
        load_in_4bit = True

    with studio_backend_imports("unsloth export"):
        from studio.backend.core.export import ExportBackend

    backend = ExportBackend()

    typer.echo(f"Loading checkpoint: {checkpoint}")
    success, message = backend.load_checkpoint(
        checkpoint_path = str(checkpoint),
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        hf_token = hf_token,
    )
    if not success:
        typer.echo(f"Error: {message}", err = True)
        raise typer.Exit(code = 1)
    typer.echo(message)

    typer.echo(f"Exporting as {format}...")
    output_path: Optional[str] = None
    if format == "merged-16bit":
        success, message, output_path = backend.export_merged_model(
            save_directory = str(output_dir),
            format_type = "16-bit (FP16)",
            push_to_hub = push_to_hub,
            repo_id = repo_id,
            hf_token = hf_token,
            private = private,
        )
    elif format == "merged-4bit":
        success, message, output_path = backend.export_merged_model(
            save_directory = str(output_dir),
            format_type = "4-bit (FP4)",
            push_to_hub = push_to_hub,
            repo_id = repo_id,
            hf_token = hf_token,
            private = private,
        )
    elif format == "gguf":
        success, message, output_path = backend.export_gguf(
            save_directory = str(output_dir),
            quantization_method = quantization.upper(),
            push_to_hub = push_to_hub,
            repo_id = repo_id,
            hf_token = hf_token,
            private = private,
        )
    elif format == "lora":
        success, message, output_path = backend.export_lora_adapter(
            save_directory = str(output_dir),
            push_to_hub = push_to_hub,
            repo_id = repo_id,
            hf_token = hf_token,
            private = private,
            adapter_format = adapter_format,
        )

    if not success:
        typer.echo(f"Error: {message}", err = True)
        raise typer.Exit(code = 1)

    typer.echo(message)
    if output_path:
        typer.echo(f"Saved to: {output_path}")


MERGE_METHODS = ["linear", "svd", "cat", "ties", "dare_ties", "dare_linear", "magnitude_prune"]


def merge_adapters(
    base_model: str = typer.Argument(..., help = "Base model name or local path."),
    output_dir: Path = typer.Argument(..., help = "Directory to save the merged model."),
    adapters: List[str] = typer.Option(
        ..., "--adapter", "-a", help = "Path to a PEFT adapter directory. Repeat for each adapter."
    ),
    adapter_weights: Optional[List[float]] = typer.Option(
        None,
        "--weight",
        "-w",
        help = "Merge weight for each adapter (same order as --adapter). Defaults to equal.",
    ),
    method: str = typer.Option(
        "linear", "--method", "-m", help = f"Merge strategy: {', '.join(MERGE_METHODS)}"
    ),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize", help = "Normalize weights."),
    density: float = typer.Option(
        0.5, "--density", help = "Fraction of weight deltas kept (TIES/DARE/magnitude-prune)."
    ),
    target_rank: Optional[int] = typer.Option(
        None, "--target-rank", help = "Output adapter rank (svd only)."
    ),
    save_method: str = typer.Option(
        "merged_16bit",
        "--save-method",
        help = "Save format: merged_16bit, merged_4bit, lora.",
    ),
    max_seq_length: int = typer.Option(2048, "--max-seq-length"),
    load_in_4bit: bool = typer.Option(False, "--load-in-4bit/--no-load-in-4bit"),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar = "HF_TOKEN", help = "HuggingFace token."
    ),
):
    """Merge multiple LoRA adapters into a base model and save the result."""
    if method not in MERGE_METHODS:
        typer.echo(
            f"Error: Invalid method '{method}'. Choose from: {', '.join(MERGE_METHODS)}",
            err = True,
        )
        raise typer.Exit(code = 2)

    if adapter_weights and len(adapter_weights) != len(adapters):
        typer.echo(
            f"Error: Number of --weight values ({len(adapter_weights)}) must match "
            f"number of --adapter values ({len(adapters)}).",
            err = True,
        )
        raise typer.Exit(code = 2)

    import unsloth
    from unsloth import FastLanguageModel

    typer.echo(f"Loading base model: {base_model}")
    model, tokenizer = FastLanguageModel.merge_adapters(
        model_name = base_model,
        adapters = adapters,
        weights = adapter_weights,
        method = method,
        normalize_weights = normalize,
        density = density,
        target_rank = target_rank,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        token = hf_token,
    )

    typer.echo(f"Saving merged model to: {output_dir}")
    model.save_pretrained_merged(str(output_dir), tokenizer, save_method = save_method)
    typer.echo("Done!")
