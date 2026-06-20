# Speech-to-Reasoning Pipeline

This project transcribes speech audio with Whisper, then optionally sends the transcription to a Hugging Face reasoning model.

## Run in Google Colab

Use Colab with a GPU runtime so the project runs in the cloud instead of on your local machine.

1. Open a new notebook at <https://colab.research.google.com/>.
2. Select `Runtime` > `Change runtime type` > `T4 GPU`.
3. Clone and install:

```bash
!git clone https://github.com/Asma-Siddiqa/GenAI.git
%cd GenAI
!pip install -r requirements.txt
```

4. Run a quick transcription-only test:

```bash
!python run_pipeline.py --text "What is the capital of Pakistan?" --whisper-model tiny --no-reasoning
```

5. Run with a reasoning model:

```bash
!python run_pipeline.py --text "If I have 3 apples and buy 4 more, how many apples do I have?" \
  --whisper-model tiny \
  --reasoning-model Qwen/Qwen2.5-0.5B-Instruct
```

Larger Whisper or reasoning models may need more GPU memory.

## Run from an uploaded audio file

```bash
!python run_pipeline.py /content/audio.wav --whisper-model base --no-reasoning
```

For reasoning, add `--reasoning-model <huggingface-model-name>`.
