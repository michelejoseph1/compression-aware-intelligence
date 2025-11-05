from cai.cfi import CFI

def generate(prompt: str) -> str:
    # Replace this with your model call
    return f"[stub model output] {prompt}"

cfi = CFI(generate=generate)
score, result = cfi.measure("Explain the difference between mass and weight.", k=8, grid_size=6)
print("CFI:", round(score, 3))
print("Components:", result.components)
if result.heatmap:
    result.heatmap.savefig("cfi_heatmap.png")
    print("Saved heatmap to cfi_heatmap.png")