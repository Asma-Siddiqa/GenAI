#!/usr/bin/env python3
"""Speech-to-Reasoning Pipeline — CLI entry point.

Transcribes audio with Whisper, then runs reasoning via a local LLM.
Supports GPU (CUDA) and CPU fallback.
"""

import argparse
import gc
import os
import sys
import torch
import warnings

warnings.filterwarnings("ignore")

from pipeline_utils import get_device, optimize_gpu_memory


def generate_sample_audio(text, filename="sample_query.wav", lang="en"):
    from gtts import gTTS
    if os.path.exists(filename):
        print(f"Using existing file: {filename}")
        return filename
    print(f"Generating audio from text: \"{text}\"")
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(filename)
    print(f"Saved to {filename}")
    return filename


def load_whisper(model_size="base", device=None):
    import whisper
    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)
    if device and device.type == "cuda":
        model = model.cuda()
        print("Whisper moved to GPU.")
    else:
        print("Whisper running on CPU.")
    return model


def transcribe_audio(model, audio_path, language="en"):
    import whisper
    result = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        fp16=torch.cuda.is_available(),
        temperature=0.0,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    text = result["text"].strip()
    duration = result["segments"][-1]["end"] if result["segments"] else 0
    print(f"Transcription complete ({duration:.1f}s audio -> {len(text.split())} words)")
    print(f"\n{'─' * 60}")
    print(f"TRANSCRIPTION:\n{text}")
    print(f"{'─' * 60}\n")
    return result


def load_reasoning_model(model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit", device=None):
    try:
        from unsloth import FastLanguageModel
        print(f"Loading quantized reasoning model: {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=torch.bfloat16 if (device and device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16 if device and device.type == "cuda" else torch.float32,
            load_in_4bit=True if (device and device.type == "cuda") else False,
            device_map="auto" if device else None,
        )
        if device and device.type == "cuda":
            FastLanguageModel.for_inference(model)
        else:
            model = model.to(device or torch.device("cpu"))
        return model, tokenizer
    except ImportError:
        print("unsloth not available, falling back to standard transformers loading...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto" if device and device.type == "cuda" else None,
        )
        if not device or device.type != "cuda":
            model = model.to("cpu")
        return model, tokenizer


REASONING_SYSTEM_PROMPT = (
    "You are a precise logical reasoning assistant. "
    "Think step by step, show your chain of thought, "
    "and provide a clear final answer."
)


def reason_on_transcription(model, tokenizer, transcription, device, max_new_tokens=512, temperature=0.1, top_p=0.95):
    messages = [
        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Transcribed speech: \"{transcription}\"\n\n"
                f"Analyze this query step by step, then provide the final answer."
            ),
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    print("Reasoning...")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    response = response.strip()

    print(f"\n{'═' * 60}")
    print("REASONING MODEL OUTPUT")
    print(f"{'═' * 60}")
    print(response)
    print(f"{'═' * 60}")

    return response


def speech_to_reasoning(audio_path, language="en", whisper_model_size="base",
                        reasoning_model_name=None, max_new_tokens=512,
                        temperature=0.1, verbose=True, device=None):
    if device is None:
        device = get_device()

    result = {"audio_path": audio_path}

    # Step 1: Load Whisper
    if verbose:
        print(f"\n{'=' * 60}")
        print("STEP 1: Loading Whisper ASR")
        print(f"{'=' * 60}")
    gc.collect()
    whisper_model = load_whisper(whisper_model_size, device if device.type == "cuda" else None)

    # Step 2: Transcribe
    if verbose:
        print(f"\n{'=' * 60}")
        print("STEP 2: Transcribing audio")
        print(f"{'=' * 60}")
    transcript_result = transcribe_audio(whisper_model, audio_path, language)
    transcription = transcript_result["text"].strip()
    result["transcription"] = transcription

    del whisper_model
    gc.collect()

    # Step 3: Reason
    if verbose:
        print(f"\n{'=' * 60}")
        print("STEP 3: Reasoning with LLM")
        print(f"{'=' * 60}")

    if reasoning_model_name:
        model, tokenizer = load_reasoning_model(reasoning_model_name, device)
        reasoning = reason_on_transcription(model, tokenizer, transcription, device,
                                            max_new_tokens, temperature)
        result["reasoning"] = reasoning
    else:
        print("No reasoning model specified. Skipping step 3.")
        result["reasoning"] = None

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Speech-to-Reasoning Pipeline: transcribe audio then reason with an LLM."
    )
    parser.add_argument("audio", nargs="?", help="Path to audio file (WAV/MP3/M4A)")
    parser.add_argument("--text", "-t", help="Generate audio from this text via TTS (for testing)")
    parser.add_argument("--whisper-model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--reasoning-model", help="HuggingFace model name (omit to skip reasoning step)")
    parser.add_argument("--language", default="en", help="Audio language code (default: en)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens for reasoning")
    parser.add_argument("--temperature", type=float, default=0.1, help="Reasoning temperature")
    parser.add_argument("--no-reasoning", action="store_true", help="Transcribe only, skip reasoning")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPU is available")

    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else get_device()
    print(f"Using device: {device}")

    # Resolve audio source
    audio_path = args.audio
    if args.text:
        audio_path = generate_sample_audio(args.text, "generated_query.wav", args.language)
    if not audio_path:
        parser.print_help()
        print("\nError: provide an audio file path or --text to generate one.")
        sys.exit(1)
    if not os.path.exists(audio_path):
        print(f"Error: audio file not found: {audio_path}")
        sys.exit(1)

    reasoning_model = None if args.no_reasoning else args.reasoning_model

    result = speech_to_reasoning(
        audio_path=audio_path,
        language=args.language,
        whisper_model_size=args.whisper_model,
        reasoning_model_name=reasoning_model,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        device=device,
    )

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"\nTranscription:\n  {result['transcription']}")
    if result.get("reasoning"):
        print(f"\nReasoning:\n{result['reasoning']}")
    print()


if __name__ == "__main__":
    main()
