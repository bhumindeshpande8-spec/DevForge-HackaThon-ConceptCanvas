import json
from itertools import product
import random

# ---------- Config ----------
shapes = ["circle", "square", "triangle", "text"]
colors = ["BLUE", "RED", "GREEN", "YELLOW", "PURPLE", "ORANGE"]
actions = ["create", "move_right", "move_left", "rotate", "scale_up", "scale_down"]
distances = [1, 2, 3]
angles = [3.14/4, 3.14/2, 3.14]
scales = [1.2, 1.5, 2.0]

num_samples = 200  # total dataset size
dataset = []

# ---------- Helper functions ----------
def generate_manim_code(instr):
    t = instr["type"]
    action = instr["action"]
    color = instr.get("color", "WHITE").upper()
    params = instr.get("parameters", {})

    lines = []

    if t == "circle":
        lines.append(f'circle = Circle(color={color})')
        shape_name = "circle"
    elif t == "square":
        lines.append(f'square = Square(color={color})')
        shape_name = "square"
    elif t == "triangle":
        lines.append(f'triangle = Triangle(color={color})')
        shape_name = "triangle"
    elif t == "text":
        content = params.get("content", f"{color} Text")
        lines.append(f'text = Text("{content}", color={color})')
        shape_name = "text"
    else:
        lines.append(f'text = Text("{t}", color=BLUE)')
        shape_name = "text"

    if action == "create":
        lines.append(f"self.play(Create({shape_name}))")
    elif action == "move_right":
        dist = params.get("distance", 2)
        lines.append(f"self.play({shape_name}.animate.shift(RIGHT*{dist}))")
    elif action == "move_left":
        dist = params.get("distance", 2)
        lines.append(f"self.play({shape_name}.animate.shift(LEFT*{dist}))")
    elif action == "rotate":
        angle = params.get("angle", 3.14/2)
        lines.append(f"self.play({shape_name}.animate.rotate({angle}))")
    elif action == "scale_up":
        scale = params.get("scale", 1.5)
        lines.append(f"self.play({shape_name}.animate.scale({scale}))")
    elif action == "scale_down":
        scale = params.get("scale", 0.7)
        lines.append(f"self.play({shape_name}.animate.scale({scale}))")

    lines.append("self.wait(2)")
    return "\n".join(lines)

# ---------- Generate dataset ----------
for _ in range(num_samples):
    instr = {}
    instr["type"] = random.choice(shapes)
    instr["color"] = random.choice(colors)
    instr["action"] = random.choice(actions)
    instr["parameters"] = {}

    if instr["action"] in ["move_right", "move_left"]:
        instr["parameters"]["distance"] = random.choice(distances)
    elif instr["action"] == "rotate":
        instr["parameters"]["angle"] = random.choice(angles)
    elif instr["action"] in ["scale_up", "scale_down"]:
        instr["parameters"]["scale"] = random.choice(scales)
    elif instr["type"] == "text":
        instr["parameters"]["content"] = f"{instr['color']} {instr['type']}"

    code = generate_manim_code(instr)
    dataset.append({"input": instr, "output": code})

# ---------- Save dataset ----------
with open("slm_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Dataset generated with {len(dataset)} samples and saved to slm_dataset.json")
