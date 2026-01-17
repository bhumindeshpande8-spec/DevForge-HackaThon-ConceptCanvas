from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
import textwrap
import uuid
import subprocess
from pathlib import Path
from typing import Optional
import shutil
import time
import re
import json
import os 

# ---------------------------------------------
# ---------- Load Local HuggingFace SLM ---------- 
# ---------------------------------------------
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer 
    import torch
    
    print("Loading trained SLM model...") 
    LOCAL_SLM_TOKENIZER = AutoTokenizer.from_pretrained("./trained_slm") 
    LOCAL_SLM_MODEL = AutoModelForCausalLM.from_pretrained("./trained_slm") 
    LOCAL_SLM_MODEL.eval() 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOCAL_SLM_MODEL.to(DEVICE)
    print(f"Local SLM Loaded Successfully on {DEVICE}.")
    LOCAL_SLM_AVAILABLE = True
except ImportError:
    print("Warning: HuggingFace transformers not installed or local model failed to load. Only Ollama will be used.")
    LOCAL_SLM_AVAILABLE = False
except Exception as e:
    print(f"Error loading local SLM: {e}")
    LOCAL_SLM_AVAILABLE = False


# ---------- Config ----------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5-coder:1.5b"
MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)
FAILED_DIR = Path("failed_scenes") 
FAILED_DIR.mkdir(exist_ok=True)

SCENE_NAME = "GeneratedScene"
# Use '-ql' for low quality/fast rendering. If rendering fails, you might try removing it.
QUALITY_FLAG = "-ql" 

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

# ---------- Schemas ----------
class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    prompt: str
    manim_code: str
    video_url: Optional[str]
    error: Optional[str] = None
    manim_log: Optional[str] = None

# ---------- Helpers: Core Functions ----------

def indent_code(code: str) -> str:
    """
    FIXED: Correctly indents the Manim code block using 8 standard ASCII spaces.
    This resolves the IndentationError issue.
    """
    # NOTE: Using 4 spaces for better readability/consistency in Python standards, 
    # but the original code used 8 spaces, so sticking to 8 spaces (two tabs) here
    # to maintain the original intent/look for this specific function.
    return "\n".join("        " + line for line in code.split("\n"))

def _extract_text_from_ollama_json(data) -> str:
    """Robustly extracts text from nested Ollama-like JSON responses."""
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        parts = []
        for p in data:
            parts.append(_extract_text_from_ollama_json(p))
        return "\n".join(parts)
    if isinstance(data, dict):
        for k in ("response","output","outputs","result","text", "content"):
            if k in data:
                return _extract_text_from_ollama_json(data[k])
        parts = []
        for v in data.values():
            parts.append(_extract_text_from_ollama_json(v))
        return "\n".join(p for p in parts if p)
    return ""

def _clean_model_output(raw_text: str) -> str:
    """Applies common cleanup steps to raw LLM text output (Ollama or Local SLM)."""
    # Remove code fences/markdown
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        if lines and lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].startswith("```"): lines = lines[:-1]
        raw_text = "\n".join(lines)
    
    # Final cleanup: remove leading headers/comments and extraneous empty lines
    lines = raw_text.splitlines()
    cleaned = []
    started = False
    for line in lines:
        s = line.strip()
        if not started:
            if not s or s.startswith("#") or s.lower() in ("python", "```python"):
                continue
            started = True
        cleaned.append(line.rstrip())
    
    extracted = "\n".join(cleaned).strip()
    return extracted

def call_ollama_for_body(prompt: str) -> str:
    """Calls Ollama via API to generate Manim code."""
    system_instructions = textwrap.dedent("""
        You are a Manim animation code generator.
        TASK:
        - Output ONLY Python statements that go INSIDE: def construct(self):
        - Do NOT output imports, class definitions, comments, or explanations.
        - Do NOT output markdown or backticks.
        - Output only valid Manim statements; include at least one animation call (e.g., self.play(...)).
        - **IMPORTANT SYNTAX RULE:** Mobjects must be created and assigned to a variable (e.g., `c = Circle()`) before being used as the first argument in `self.play(Transform(c, ...))`. Do NOT create Mobjects directly inside self.play() unless using Create or DrawBorderThenFill.
        - Keep output to 3-15 lines.
    """).strip()


    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{system_instructions}\n\nUser request:\n{prompt}\n\nCode lines:",
        "stream": False,
        "options": {"num_predict": 1024},
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    raw_text = _extract_text_from_ollama_json(data)
    
    return _clean_model_output(raw_text)

def call_local_slm_for_body(prompt: str) -> str:
    """
    Uses the locally loaded HuggingFace model for inference. 
    Loads model once at startup and reuses it per request.
    """
    if not LOCAL_SLM_AVAILABLE:
        raise RuntimeError("Local SLM is not available.")

    # --- UPDATED SYSTEM INSTRUCTIONS FOR ROBUST MANIM CODE ---
    system_instructions = textwrap.dedent("""
        You are a Manim animation code generator. Output ONLY Python statements that go INSIDE: def construct(self):. Do NOT output imports, class definitions, comments, or explanations.
        - Output only valid Manim statements; always use the `Create` animation method instead of `ShowCreation`.
        - IMPORTANT SYNTAX RULE: Mobjects must be created and assigned to a variable (e.g., `c = Circle()`) before being used as the first argument in `self.play(Transform(c, ...))`.
    """).strip()
    # --------------------------------------------------------

    input_text = f"{system_instructions}\n\nUser request:\n{prompt}\n\nCode lines:"
    
    inputs = LOCAL_SLM_TOKENIZER(input_text, return_tensors="pt").to(DEVICE)
    
    outputs = LOCAL_SLM_MODEL.generate(
        **inputs, 
        max_new_tokens=300, 
        do_sample=True, 
        temperature=0.7, 
        eos_token_id=LOCAL_SLM_TOKENIZER.eos_token_id
    )
    
    raw_text = LOCAL_SLM_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    
    # Trim the input prompt and instructions from the raw output
    if raw_text.startswith(input_text):
        raw_text = raw_text[len(input_text):].strip()
    
    return _clean_model_output(raw_text)

# ---------- Validation helpers and Fallback-code synthesizer ----------
ANIMATION_KEYWORDS = ["self.play(","Create(","FadeIn(","FadeOut(","Write(","GrowFromCenter(","Rotate(","MoveAlongPath(","Transform(","Scale(","Shift(","Animate(","play(","Add()","DrawBorderThenFill("]
SHAPE_KEYWORDS = ["Circle(","Square(","Triangle(","Dot(", "Line(","Arc(", "Text(","MathTex(","VGroup("]

ANIMATION_KEYWORDS = ["self.play(","Create(","FadeIn(","FadeOut(","Write(","GrowFromCenter(","Rotate(","MoveAlongPath(","Transform(","Scale(","Shift(","Animate(","play(","Add()","DrawBorderThenFill("]
SHAPE_KEYWORDS = ["Circle(","Square(","Triangle(","Dot(", "Line(","Arc(", "Text(","MathTex(","VGroup("]

def looks_like_valid_manim_body(body: str) -> bool:
    if not body or not body.strip(): return False
    
    # ------------------ ADD THIS VALIDATION CHECK ------------------
    # Check for the common error: Transform(Mobject(...), ...)
    # This rejects the problematic pattern `Transform(Circle(...), ...)`
    if re.search(r"Transform\(\s*(" + "|".join([s[:-1] for s in SHAPE_KEYWORDS]) + r")\(", body, re.IGNORECASE):
        # Allow Transform only if the first argument is "self" or "self.mobject" 
        # (A more complex check than simple keyword exclusion might be needed, 
        # but this simple check catches most LLM errors.)
        if not re.search(r"Transform\(\s*self\.", body, re.IGNORECASE):
             print("Validation failed: Found Mobject creation inside Transform.")
             return False
    # ---------------------------------------------------------------
    
    if any(k in body for k in ANIMATION_KEYWORDS): return True
    if any(s in body for s in SHAPE_KEYWORDS) and "self.play" in body: return True
    return False

def prompt_to_fallback_body(prompt: str) -> str:
    p = prompt.lower()
    if re.search(r"\bcircle\b", p):
        color = "BLUE" if "blue" in p else ("GREEN" if "green" in p else "WHITE")
        return f"c = Circle(color={color})\nself.play(Create(c))\nself.wait(1)"
    if re.search(r"\bsquare\b", p):
        return "s = Square()\nself.play(GrowFromCenter(s))\nself.wait(1)"
    if re.search(r"\btext\b|\bhello\b|\bworld\b", p):
        content = "Hello World"
        return f"t = Text(\"{content}\", font_size=48)\nself.play(Write(t))\nself.wait(1)"
    h = abs(hash(prompt)) % 360
    shapes = ["Circle", "Square", "Triangle"]
    shape = shapes[h % len(shapes)] 
    colors = ["BLUE", "GREEN", "YELLOW", "RED", "ORANGE", "PURPLE"]
    color = colors[h % len(colors)]
    if shape == "Square": return f"s = Square(color={color})\nself.play(GrowFromCenter(s))\nself.wait(1)"
    elif shape == "Triangle": return f"t = Triangle(color={color})\nself.play(Create(t))\nself.wait(1)"
    else: return f"c = Circle(color={color})\nself.play(Create(c))\nself.wait(1)"

# ---------- Manim Renderer ----------
def render_manim_to_video(body_code: str) -> (Optional[str], Optional[str]):
    """Renders the given Manim code BODY, returns the video URL and the Manim log."""
    scene_id = uuid.uuid4().hex[:8]
    py_path = Path(f"generated_scene_{scene_id}.py")
    target_filename = f"scene_{scene_id}.mp4"
    manim_log = None
    full_manim_script = f"""
from manim import *

class {SCENE_NAME}(Scene):
    def construct(self):
{indent_code(body_code)}
""".lstrip()

    py_path.write_text(full_manim_script, encoding="utf-8")

    try:
        # --- FIX: Changed the Manim setup check to a reliable command ---
        # The original check 'manim render -h' failed with 'No such option: -h'.
        # Using 'manim --version' is a safer way to confirm the binary exists.
        subprocess.run(["manim", "--version"], capture_output=True, check=True, text=True)
        # ----------------------------------------------------------------
        
        # Command setup
        cmd_base = ["manim", "render", str(py_path.resolve()), SCENE_NAME]
        
        # Use -ql for quality and -o (output directory) and --output_file (final file name)
        cmd_options = [
            QUALITY_FLAG, 
            "--media_dir", str(MEDIA_DIR.resolve()), 
            "--output_file", target_filename,
        ]
        
        cmd = cmd_base + cmd_options
        print(f"Executing Manim command: {' '.join(cmd)}")
        
        # Run command from a neutral location (the current directory)
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(".").resolve()))
        
        # Capture full log (limiting size for FastAPI response)
        manim_log = f"COMMAND: {' '.join(cmd)}\n--- STDOUT ---\n{result.stdout[:2000]}\n--- STDERR ---\n{result.stderr[:2000]}"

        if result.returncode != 0:
            print(f"Manim failed (return code {result.returncode}). Check logs.")
            # Copy failed script for later inspection
            failed_file = FAILED_DIR / f"failed_scene_{scene_id}.py"
            shutil.copy(py_path, failed_file)
            return None, manim_log

        time.sleep(0.8) 
        
    except FileNotFoundError: return None, "Manim binary not found (FileNotFoundError). Ensure 'manim' is in your PATH."
    except subprocess.CalledProcessError as e: return None, f"Manim setup command failed: {e}. Log: {manim_log}"
    except Exception as e: return None, f"Unexpected Manim invocation error: {e}"
    finally:
        # Clean up the generated Python file
        if py_path.exists():
            try: py_path.unlink()
            except Exception: pass

    # ... (rest of the file location logic remains the same)
    # --- Robustly locate the video file ---
    candidates = list(MEDIA_DIR.rglob(f"*{SCENE_NAME}.mp4"))
    
    # Also check for the exact target filename, just in case the flags worked perfectly
    if not candidates:
        candidates = list(MEDIA_DIR.rglob(target_filename)) 

    if not candidates:
        print(f"No output video found after rendering. Searched for *{SCENE_NAME}.mp4 or {target_filename} in {MEDIA_DIR.resolve()}")
        return None, manim_log

    # Sort by modification time to get the newest file (most likely the one we just generated)
    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    out_path = candidates[0]
    
    # Rename/move to the flat media directory for serving
    target = MEDIA_DIR / target_filename
    
    try:
        if out_path != target: # Only move if the path is different (i.e., it's in a subdirectory)
            shutil.move(str(out_path), str(target))
    except Exception as e:
        return None, f"Failed moving final video from {out_path} to {target}: {e}"

    return f"/media/{target.name}", manim_log


# ------------------------------------
# ---------- Route ----------
# ------------------------------------
@app.post("/generate_scene", response_model=GenerateResponse)
def generate_scene(req: GenerateRequest):
    fallback_reason = None
    manim_log = ""
    used_body = ""

    # --- ATTEMPT 1: Ollama (Primary) ---
    try:
        print("Attempting to generate code via Ollama...")
        raw_body = call_ollama_for_body(req.prompt)
        if looks_like_valid_manim_body(raw_body):
            used_body = raw_body
            print("Ollama code accepted.")
        else:
            print("Ollama code invalid. Proceeding to next step.")
            raise ValueError("Ollama output invalid.")
    except Exception as e:
        manim_log += f"Ollama Call/Validation failed: {e}\n"
        print(f"Ollama failed. Trying local SLM (if available)...")

        # --- ATTEMPT 2: Local SLM (Fallback) ---
        if LOCAL_SLM_AVAILABLE and not used_body:
            try:
                raw_body_slm = call_local_slm_for_body(req.prompt)
                if looks_like_valid_manim_body(raw_body_slm):
                    used_body = raw_body_slm
                    manim_log += "Local SLM code accepted.\n"
                    print("Local SLM code accepted.")
                else:
                    manim_log += "Local SLM code invalid.\n"
                    print("Local SLM code invalid. Proceeding to generic fallback.")
            except Exception as e_slm:
                manim_log += f"Local SLM Call failed: {e_slm}\n"
                print(f"Local SLM failed: {e_slm}. Proceeding to generic fallback.")

    # --- ATTEMPT 3: Generic Code Synthesizer (Last Resort) ---
    if not used_body:
        print("Synthesizing prompt-based fallback code.")
        used_body = prompt_to_fallback_body(req.prompt)
        fallback_reason = "model_failure"
        manim_log += "Used generic prompt-based fallback code.\n"
        
    final_manim_code = used_body

    # --- RENDER ATTEMPT ---
    video_url, render_log = render_manim_to_video(used_body)
    manim_log += (render_log or "")

    # --- Final Response ---
    if video_url is None:
        error_msg = f"Rendering failed for the final code block. Fallback reason: {fallback_reason or 'Model Generated'}"
        return GenerateResponse(
            prompt=req.prompt,
            manim_code=final_manim_code,
            video_url=None,
            error=error_msg,
            manim_log=manim_log
        )

    # Success
    return GenerateResponse(
        prompt=req.prompt,
        manim_code=final_manim_code,
        video_url=video_url,
        error=None,
        manim_log=manim_log
    )