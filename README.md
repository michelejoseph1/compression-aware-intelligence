<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">

</head>
<body>

<h1>compression aware intelligence (cai)</h1>

<p>
this repo contains a small implementation of the coherence field index (cfi). 
</p>

<p>
cfi measures how stable an llm internal representation is when you make small changes to the input prompt. if the model keeps the same internal reasoning path, cfi stays low. if the model switches between different hidden interpretations, cfi becomes higher. high cfi often correlates with hallucination or drift. low cfi indicates stable reasoning.
</p>

<p>
when two prompts with the same meaning produce different outputs, the model’s latent representations are in conflict and Compression-Aware Intelligence (CAI) measures that conflict as compression strain. this library does not correct responses. it only measures stability. it is intended to be used as one component inside a larger compression tension score or contradiction aware inference workflow.
</p>

<h2>installation</h2>

<pre>pip install -U scikit-learn matplotlib
pip install .</pre>

<h2>quick start</h2>

<pre>from cai.cfi import CFI

# simple generator placeholder. replace with your model call.
def generate(prompt: str) -&gt; str:
    return f"answer: {prompt}"

cfi = CFI(generate=generate)
score, res = cfi.measure("explain the difference between mass and weight.", k=8, grid_size=6)

print("cfi:", round(score, 3))
print("components:", res.components)

if res.heatmap:
    res.heatmap.savefig("cfi_heatmap.png")</pre>

<p>
after running, you receive:
</p>

<ul>
<li>cfi score in [0,1]</li>
<li>breakdown of residual curvature, spread, and entropy</li>
<li>a heatmap showing how stable the representation is across prompt variations</li>
</ul>

<h2>what cfi is capturing</h2>

<p>
when you slightly modify a prompt, a stable model retains the same internal narrative and the embeddings of outputs cluster tightly. this yields low curvature and a low cfi score.
</p>

<p>
if the model does not have a consistent way to represent or resolve its answer, slight prompt variations push it into different output modes. representation curvature increases. cfi score rises.
</p>

<pre>
low cfi  = the model "knows what it is talking about"
high cfi = the model is improvising under compression pressure
</pre>

<h2>using your own model</h2>

<pre>def generate(prompt):
    # example (openai style)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def embed(texts):
    # return numpy array of shape [n, d]
    # example: sentence transformer or openai embedding call
    ...
    
cfi = CFI(generate=generate, embed=embed)</pre>

<h2>how this relates to compression tension score (cts)</h2>

<p>
cfi measures representation stability alone. cts integrates cfi with grounding, contradiction rate, response entropy, and uncertainty. cfi is the internal field signal. cts decides release vs adjust vs refuse. compression loss → contradiction → compression strain → instability
</p>

<h2>directory</h2>

<pre>cai/
  cfi.py
  latent.py
  perturb.py
  visualize.py
examples/
  cfi_demo.py</pre>

<h2>license</h2>

<p>mit</p>

</body>
</html>
