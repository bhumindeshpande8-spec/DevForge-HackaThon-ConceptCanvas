# DevForge-HackaThon-ConceptCanvas
# Manim Animation Generator API

A FastAPI backend that turns natural language prompts into rendered Manim MP4 animations using a local fine‑tuned SLM (GPT‑2 via HuggingFace) with Ollama as a backup code generator.

***

## Overview

This project exposes a single `/generate_scene` endpoint that:  
1) takes a text prompt,  
2) generates Manim `construct()` body code via a small language model,  
3) validates and optionally falls back to a safe template,  
4) calls the Manim CLI to render the scene, and  
5) serves the resulting MP4 from `/media`.

The repo also includes a tiny synthetic dataset generator and training script to fine‑tune GPT‑2 specifically for Manim body code.

***

## Tech Stack

| Layer        | Technology                                  | Purpose |
|-------------|----------------------------------------------|---------|
| API         | FastAPI, Pydantic                            | HTTP endpoints, request/response models  |
| ML Inference| HuggingFace Transformers (GPT‑2), PyTorch    | Local SLM for Manim code generation|
| Fallback ML | Ollama `qwen2.5-coder:1.5b` HTTP API         | Secondary code generator if local SLM fails |
| Rendering   | Manim Community CLI                          | Render Python scene to MP4  |
| Data/Training | `datasets`, `Trainer` (HF Transformers)    | Load JSONL, fine‑tune GPT‑2 on Manim dataset |
| Utilities   | Python stdlib (`json`, `subprocess`, `pathlib`, `uuid`, `random`, `re`, `shutil`, `os`, `time`) | Dataset creation, file I/O, process management |

***

## Features

- Text‑to‑animation pipeline with deterministic wrapping of LLM output into a Manim `Scene` subclass.  
- Local SLM (fine‑tuned GPT‑2) loaded once at startup, running on CPU or GPU depending on availability.
- Ollama fallback using `qwen2.5-coder:1.5b` over HTTP (`/api/generate`, non‑streaming).
- Strong output cleanup (removes markdown fences, comments) and structural validation of generated Manim code.  
- Special validator to reject invalid patterns like `Transform(Circle(...), ...)` where Mobjects are created inline.  
- Prompt‑based fallback code synthesizer for circles, squares, triangles, and text so a valid animation is always produced.  
- Manim render helper that:  
  - writes a temporary `generated_scene_<id>.py`,  
  - calls `manim render -ql` with a configured scene name,  
  - collects logs,  
  - moves the freshest MP4 into a flat `media/` directory for static serving.  
- Automatically stores failed scenes’ Python files in `failed_scenes/` for debugging.  

***

## Project Structure

Typical layout (filenames inferred from the code you shared):

- `app.py` – FastAPI app, model loading, Ollama client, validation, rendering.  
- `slm_dataset.json` – Synthetic instruction → Manim code pairs.  
- `slm_dataset_hf.jsonl` – Same dataset in JSONL format for HuggingFace.  
- `trained_slm/` – Saved GPT‑2 model and tokenizer after fine‑tuning.  
- `media/` – Output MP4 files served at `/media/...`.  
- `failed_scenes/` – Python files for scenes that failed to render.  

Dataset / training scripts (conceptually separate, though currently inline):

- **Dataset generator**
  - Builds random instructions with:
    - `type`: `circle | square | triangle | text`
    - `color`: Manim color constants
    - `action`: `create | move_{right,left} | rotate | scale_{up,down}`
    - `parameters`: distance / angle / scale / text content.
  - Uses `generate_manim_code(instr)` to create a valid Manim body string.
  - Saves `[{ "input": instr, "output": code }, ...]` to `slm_dataset.json`.

- **Dataset converter**
  - Reads `slm_dataset.json`.
  - Writes `slm_dataset_hf.jsonl` with:
    - `input_text` = JSON‑dumped instruction dict,
    - `target_text` = Manim code string.

- **Training script**
  - Loads `slm_dataset_hf.jsonl` using `datasets.load_dataset`.
  - Loads `gpt2` + tokenizer; uses `eos_token` as `pad_token` if needed.
  - Tokenizes inputs and labels to `max_length=128`.
  - Configures `TrainingArguments` with CPU‑friendly settings.
  - Trains with `Trainer` and saves to `./trained_slm`.

***

## How It Works

### 1. Model Loading

At startup, the app attempts to load the local SLM:

```python
LOCAL_SLM_TOKENIZER = AutoTokenizer.from_pretrained("./trained_slm")
LOCAL_SLM_MODEL = AutoModelForCausalLM.from_pretrained("./trained_slm")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOCAL_SLM_MODEL.to(DEVICE)
LOCAL_SLM_MODEL.eval()
```

If this fails (missing dependencies or directory), the app logs the error and continues in Ollama‑only mode.

### 2. Request Flow (`POST /generate_scene`)

**Request body:**

```json
{
  "prompt": "Explain Pythagoras with a right triangle and rotating labels"
}
```

**Handler logic (high level):**

1. **Ollama attempt**  
   - `call_ollama_for_body(prompt)` builds a strict system prompt (only body lines, no imports/classes).  
   - Sends a POST to `http://localhost:11434/api/generate` with model `qwen2.5-coder:1.5b`.
   - Extracts nested `response` / `content` fields and cleans markdown fences and headers.

2. **Validation**  
   - `looks_like_valid_manim_body(body)` checks:
     - presence of animation keywords like `self.play(`, `Create(`, etc.  
     - that shapes and animations co‑exist.  
     - rejects risky patterns like `Transform(Circle(...), ...)` where a Mobject is created inline.

3. **Local SLM fallback**  
   - If Ollama fails or produces invalid code and `LOCAL_SLM_AVAILABLE` is `True`:
     - `call_local_slm_for_body(prompt)` runs the trained GPT‑2 model with a Manim‑specific system prompt.  
     - The same validation is applied.

4. **Prompt‑based fallback**  
   - If neither model produces acceptable code:
     - `prompt_to_fallback_body(prompt)` synthesizes a safe default, e.g.:

       ```python
       c = Circle(color=BLUE)
       self.play(Create(c))
       self.wait(1)
       ```

     - Marks `fallback_reason = "model_failure"`.

5. **Rendering**  
   - Final code (from Ollama, local SLM, or fallback) is sent to `render_manim_to_video(body_code)`.

### 3. Rendering Pipeline

`render_manim_to_video(body_code)`:

1. Builds a complete script:

   ```python
   from manim import *

   class GeneratedScene(Scene):
       def construct(self):
           <indented body_code>
   ```

2. Saves to `generated_scene_<scene_id>.py`.  
3. Verifies Manim installation with `manim --version`.
4. Runs:

   ```bash
   manim render generated_scene_<id>.py GeneratedScene -ql \
       --media_dir <MEDIA_DIR> \
       --output_file scene_<id>.mp4
   ```

5. Captures stdout/stderr (truncated) as `manim_log`.  
6. On non‑zero exit:
   - Copies the script to `failed_scenes/failed_scene_<id>.py`.  
   - Returns `(None, manim_log)`.

7. On success:
   - Searches `MEDIA_DIR` recursively for `*GeneratedScene.mp4` or `scene_<id>.mp4`.  
   - Picks the most recent file, moves it to flat `MEDIA_DIR/scene_<id>.mp4`.  
   - Returns `("/media/scene_<id>.mp4", manim_log)`.

FastAPI exposes the folder:

```python
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
```

so the final file is reachable via HTTP.

***

## API Reference

### `POST /generate_scene`

**Request**

```json
{
  "prompt": "Show a blue circle that moves right and then scales up"
}
```

**Response (success)**

```json
{
  "prompt": "...",
  "manim_code": "c = Circle(color=BLUE)\nself.play(Create(c))\n...",
  "video_url": "/media/scene_ab12cd34.mp4",
  "error": null,
  "manim_log": "COMMAND: manim render ...\n--- STDOUT ---\n...\n--- STDERR ---\n..."
}
```

**Response (rendering error)**

```json
{
  "prompt": "...",
  "manim_code": " ... final code used ... ",
  "video_url": null,
  "error": "Rendering failed for the final code block. Fallback reason: model_failure",
  "manim_log": "COMMAND: manim render ...\n--- STDOUT ---\n...\n--- STDERR ---\n..."
}
```

***

## Setup & Running

### 1. Install Dependencies

```bash
pip install fastapi uvicorn pydantic requests manim
pip install transformers datasets torch
```

Install Manim Community as per the official guide and make sure:

```bash
manim --version
```

works in your shell.[5]

If you want Ollama fallback, also install Ollama and pull the coder model:

```bash
ollama pull qwen2.5-coder:1.5b
```

which exposes the model at `http://localhost:11434/api/generate`.[3]

### 2. Generate Dataset & Train SLM

```bash
python generate_dataset.py      # writes slm_dataset.json
python convert_dataset.py       # writes slm_dataset_hf.jsonl
python train_slm.py             # trains GPT-2, saves to ./trained_slm
```

Confirm `trained_slm/` contains a valid HuggingFace model and tokenizer (config, weights, vocab).

### 3. Run the API

```bash
uvicorn app:app --reload --port 8000
```

Open Swagger UI: http://localhost:8000/docs and test `POST /generate_scene`.

***

## Usage Example

1. Start FastAPI, verify `manim --version` works, and ensure either:
   - `trained_slm/` is present, or  
   - Ollama is running with `qwen2.5-coder:1.5b`.

2. In Swagger UI, send:

   ```json
   { "prompt": "Create a green square that rotates 90 degrees" }
   ```

3. In the JSON response:
   - Copy `video_url` and open `http://localhost:8000` + `video_url` in a browser.  
   - Inspect `manim_code` to see the generated Manim snippet.

Frontend example (plain JS):

```js
const res = await fetch('http://localhost:8000/generate_scene', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ prompt: 'Show a blue circle' })
});
const data = await res.json();
video.src = 'http://localhost:8000' + data.video_url;
codePre.textContent = data.manim_code;
```

***

## Error Handling & Debugging

- If **local SLM load fails**:  
  - The app logs a warning and continues using only Ollama + hardcoded fallback.

- If **both models fail validation**:  
  - `prompt_to_fallback_body` generates a simple shape animation so Manim still receives valid code.

- If **Manim is missing**:  
  - The error explicitly says `"Manim binary not found (FileNotFoundError)"`.

- If **rendering fails**:  
  - The failing scene is saved under `failed_scenes/failed_scene_<id>.py`.  
  - `manim_log` contains command and truncated stdout/stderr for inspection.

Checklist:

- `manim --version` works and matches the CLI options used.
- `curl http://localhost:11434/api/generate` returns a response if using Ollama.
- `trained_slm/` exists and was produced by the given training script.

***

## Possible Extensions

- Expand the synthetic dataset with more shapes, colors, and animation patterns.  
- Add AST‑based validation for generated code to catch more subtle errors.  
- Run Manim rendering in a worker queue (Celery/RQ) for non‑blocking behavior.
- Expose a second endpoint to return only `manim_code` without rendering for faster prototyping.


