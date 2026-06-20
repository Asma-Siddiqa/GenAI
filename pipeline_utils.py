"""Utility functions for the Speech-to-Reasoning pipeline."""

import torch
import numpy as np
import io
import os
import requests
from typing import Optional, Tuple

AUDIO_SAMPLE_RATE = 16000


def get_device() -> torch.device:
    """Detect available device (GPU preferred, CPU fallback)."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Available: {gpu_name}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return torch.device("cuda")
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        print("GPU Available: MPS (Apple Silicon)")
        return torch.device("mps")
    else:
        print("WARNING: No GPU detected - running on CPU (will be slow)")
        return torch.device("cpu")


def optimize_gpu_memory():
    """Clear GPU cache for efficient memory management."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def download_sample_audio(url: str, save_path: str = "sample_audio.wav") -> str:
    """Download a sample audio file from a URL."""
    if os.path.exists(save_path):
        print(f"Audio file already exists at {save_path}")
        return save_path

    print(f"Downloading sample audio from {url}...")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Saved to {save_path}")
    return save_path


def transcribe_with_whisper(
    whisper_pipeline,
    audio_input,
    temperature: float = 0.0,
    language: Optional[str] = None,
    task: str = "transcribe",
) -> Tuple[str, dict]:
    """Transcribe audio using Whisper pipeline with options."""
    generate_kwargs = {
        "task": task,
        "temperature": temperature,
    }
    if language:
        generate_kwargs["language"] = language

    result = whisper_pipeline(
        audio_input,
        generate_kwargs=generate_kwargs,
        return_timestamps=False,
    )

    text = result["text"].strip()
    return text, result


def build_reasoning_prompt(
    transcription: str,
    system_prompt: str = "You are a logical reasoning assistant. Think step by step and provide clear, accurate answers.",
    few_shot: bool = True,
) -> str:
    """Build a chat-formatted prompt for the reasoning model."""
    if few_shot:
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Analyze and answer the following based on the transcribed speech:\n\n"
                    f"Transcription: \"{transcription}\"\n\n"
                    f"Reason through this step by step, then provide the final answer."
                ),
            },
        ]
    else:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcription},
        ]


def run_reasoning(
    model,
    tokenizer,
    messages: list,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.95,
    top_k: int = 50,
    device: Optional[torch.device] = None,
) -> str:
    """Run the reasoning model on a chat-formatted prompt."""
    if device is None:
        device = get_device()
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = inputs.to(device) if hasattr(inputs, "to") else inputs
    if isinstance(inputs, torch.Tensor):
        generation_inputs = {"input_ids": inputs}
        prompt_length = inputs.shape[1]
    else:
        generation_inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        prompt_length = generation_inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **generation_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    return response.strip()
