# Qwen-Scope SAE Toxicity Circuits

This folder contains our small set of experiments using Qwen-Scope SAE features as interpretable binary inputs for toxicity classification.

The starting point is the Qwen-Scope technical report: [Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models](https://qianwen-res.oss-accelerate.aliyuncs.com/qwen-scope/Qwen_Scope.pdf). The [Qwen-Scope SAE model cards](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50) describe these as Sparse Autoencoders trained inside Qwen hidden layers to extract sparse, decoupled, interpretable features. The relevant Qwen citation is:

```bibtex
@misc{qwen_scope,
    title = {{Qwen-Scope}: Turning Sparse Features into Development Tools for Large Language Models},
    url = {https://qianwen-res.oss-accelerate.aliyuncs.com/qwen-scope/Qwen_Scope.pdf},
    author = {{Qwen Team}},
    month = {April},
    year = {2026}
}
```

## Setup

Script:

```text
qwen/qwen_scope_toxicity_classifier.py
```

Default model and SAE:

```text
model:   Qwen/Qwen3-1.7B-Base
SAE:     Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50
layer:   6
dataset: textdetox/multilingual_toxicity_dataset
lang:    en
```

The difflogic classifier depends on the local sibling checkout:

```text
difflogic/
```

The script imports it through `--difflogic-path`, defaulting to that local checkout. Because `difflogic` is its own git repository, the cleanest reproducible setup is to pin it as a dependency, for example as a git submodule:

```bash
git submodule add https://github.com/Felix-Petersen/difflogic difflogic
```

In this workspace we used the local CPU/Python implementation. That avoids needing the CUDA extension for these tiny Boolean circuits.

The script collects example-level SAE feature firing:

```text
h[i, f] = 1 if SAE feature f fires at any token in example i
```

The expensive firing matrices are cached under:

```text
qwen/out/firing_cache/
```

The cache key excludes `--top-k-features`, so top-2/top-4/top-8/top-16/top-32 sweeps reuse the same firing tensors when the dataset, layer, model, SAE, seed, max length, and epsilon match.

## Dataset Examples

A few raw English examples from `textdetox/multilingual_toxicity_dataset`:

| Label | Text |
| --- | --- |
| clean | The trans women reading this tweet right now is beautiful |
| clean | Hero Rohit Sharma love From Pakistan |
| toxic | Females are stupid dumb and ready to do anything for fucking clout |

The dataset contains social-media style text, so some toxic examples include offensive language.


## Paper-Style Baseline

Following the Qwen-Scope toxicity classifier idea, we first rank SAE features by:

```text
delta = P(feature fires | toxic) - P(feature fires | clean)
```

Then the paper-style classifier predicts toxic if **any** selected top-k feature fires:

```text
toxic = i1 OR i2 OR ... OR ik
```

This is simple and interpretable, but it becomes too permissive as `k` grows: recall goes up, false positives explode.

## OR Baseline Results

All results below are on the held-out eval split of 1000 examples, balanced 500 toxic / 500 clean.

| top-k | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.872 | 0.937 | 0.798 | 0.862 | 399 | 27 | 473 | 101 |
| 2 | 0.867 | 0.813 | 0.954 | 0.878 | 477 | 110 | 390 | 23 |
| 4 | 0.826 | 0.750 | 0.978 | 0.849 | 489 | 163 | 337 | 11 |
| 5 | 0.780 | 0.697 | 0.990 | 0.818 | 495 | 215 | 285 | 5 |
| 8 | 0.710 | 0.634 | 0.996 | 0.774 | 498 | 288 | 212 | 2 |
| 16 | 0.581 | 0.544 | 1.000 | 0.705 | 500 | 419 | 81 | 0 |
| 32 | 0.522 | 0.511 | 1.000 | 0.677 | 500 | 478 | 22 | 0 |

The pattern is clear: larger top-k OR rules catch nearly every toxic example, but they label many clean examples as toxic.

## Our Experiment: Logic Gates Over SAE Features

Instead of using a flat OR over top-k features, we train a tiny differentiable logic gate network using [`difflogic`](../difflogic/README.md).

The input is still the same interpretable binary SAE firing vector:

```text
i1, i2, ..., ik
```

But the classifier can learn connections such as:

```text
(i1 OR i6) AND (i2 OR i7)
```

This is useful because it can express feature interactions. For example, a feature may be noisy alone, but useful when combined with another feature.

## Gemma Scope Version

We also added a Gemma version of the same experiment:

```text
qwen/gemma_scope_toxicity_classifier.py
```

This keeps the same toxicity dataset, feature discovery rule, OR baseline, and difflogic classifier, but swaps Qwen-Scope for Google DeepMind's Gemma Scope SAEs:

```text
model:              google/gemma-2-2b
SAE release:        gemma-scope-2b-pt-res-canonical
SAE id:             layer_20/width_16k/canonical
Neuronpedia model:  gemma-2-2b
Neuronpedia source: 20-gemmascope-res-16k
```

Gemma Scope sources:

```text
Gemma Scope landing page:
https://huggingface.co/google/gemma-scope

Gemma 2 2B residual SAE repo:
https://huggingface.co/google/gemma-scope-2b-pt-res

Neuronpedia source for layer 20 residual 16k:
https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/3585
```

Gemma Scope loading uses SAELens, so install it in the environment before running:

```bash
pip install sae-lens
```

Run example:

```bash
qwen/.venv/bin/python qwen/gemma_scope_toxicity_classifier.py \
  --language en \
  --layer 20 \
  --top-k-features 8 \
  --batch-size 4 \
  --classifier difflogic \
  --logic-output binary \
  --logic-dims 8,4,2,1
```

The selected Gemma features are saved with Neuronpedia URLs and, when available, automatic explanations:

```text
features[*].neuronpedia.neuronpedia_url
features[*].neuronpedia.explanations
```

## Binary Toxic Circuit

The difflogic experiment uses a **single hard binary toxic circuit**:

```text
inputs -> LogicLayer -> ... -> LogicLayer(?, 1)
```

The final hard output means:

```text
1 = toxic
0 = clean / not toxic
```

Run example:

```bash
qwen/.venv/bin/python qwen/qwen_scope_toxicity_classifier.py \
  --language en \
  --layer 6 \
  --top-k-features 32 \
  --batch-size 16 \
  --classifier difflogic \
  --logic-output binary \
  --logic-dims 32,16,8,4,2,1
```

The learned circuit is saved in the output JSON at:

```text
logic_classifier.hard_circuit.binary_output.expression
```

## Binary Logic Results

| top-k | Architecture | Hard Toxic Circuit | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | `2,1` | `i1` | 0.872 | 0.937 | 0.798 | 0.862 | 399 | 27 | 473 | 101 |
| 4 | `4,2,1` | `(i2 AND i1)` | 0.882 | 0.978 | 0.782 | 0.869 | 391 | 9 | 491 | 109 |
| 8 | `8,4,2,1` | `(((i2 OR (i6 OR i7)) AND i1) OR ((i7 AND i2) OR i3))` | 0.907 | 0.930 | 0.880 | 0.904 | 440 | 33 | 467 | 60 |
| 16 | `16,8,4,2,1` | `(((i1 OR i7) OR i6) AND (i14 OR (i2 AND (i1 OR (i3 OR i5)))))` | 0.908 | 0.953 | 0.858 | 0.903 | 429 | 21 | 479 | 71 |
| 32 | `32,16,8,4,2,1` | `((i1 OR i6) AND (i2 OR i7))` | 0.897 | 0.950 | 0.838 | 0.891 | 419 | 22 | 478 | 81 |

For comparison, the top-32 OR baseline was:

```text
Accuracy  0.522
Precision 0.511
Recall    1.000
F1        0.677
FP        478
```

The learned binary hard circuit reduced false positives from `478` to `22`, while keeping useful recall. The best F1 among these binary hard circuits was top-8:

```text
F1        0.904
Accuracy  0.907
Precision 0.930
Recall    0.880
```

## Feature Meaning In The Circuits

The top-2 circuit:

```text
toxic = i1
```

uses:

```text
i1 = SAE feature 8674
```

The top-4 circuit:

```text
toxic = (i2 AND i1)
```

maps to:

```text
i1 = SAE feature 8674
i2 = SAE feature 7390
```

The top-32 circuit:

```text
toxic = ((i1 OR i6) AND (i2 OR i7))
```

uses:

```text
i1 = SAE feature 8674
i2 = SAE feature 7390
i6 = SAE feature 20110
i7 = SAE feature 8116
```

So even though we supplied 32 SAE features, the final hard circuit selected a compact interaction over 4 of them.

## Output Files

Important outputs:

```text
qwen/out/qwen_scope_layer6_en_top2_difflogic.json
qwen/out/qwen_scope_layer6_en_top4_difflogic.json
qwen/out/qwen_scope_layer6_en_top8_difflogic.json
qwen/out/qwen_scope_layer6_en_top16_difflogic.json
qwen/out/qwen_scope_layer6_en_top32_difflogic.json
```

Each current difflogic JSON contains:

```text
or_metrics
logic_classifier.eval_soft_metrics
logic_classifier.eval_hard_metrics
logic_classifier.hard_circuit
```

For visualization, prefer:

```text
logic_classifier.hard_circuit.layers
```

Each node has:

```json
{
  "id": "L0.g0",
  "op": "and",
  "inputs": ["i1", "i2"],
  "expression": "(i1 AND i2)"
}
```

For binary difflogic outputs, the final simplified toxic expression is in:

```text
logic_classifier.hard_circuit.binary_output.expression
```

## Yosys Export

To turn a learned binary toxic circuit JSON into Verilog, optimized Verilog, and a PNG graph via Yosys:

```bash
python qwen/yosys_logic_export.py qwen/out/qwen_scope_layer6_en_top32_difflogic.json
```

Default outputs:

```text
qwen/out/yosys/qwen_scope_layer6_en_top32_difflogic/toxic_circuit.v
qwen/out/yosys/qwen_scope_layer6_en_top32_difflogic/toxic_circuit_opt.v
qwen/out/yosys/qwen_scope_layer6_en_top32_difflogic/toxic_circuit_opt.png
```

If Yosys is not installed or not loaded on `PATH`, you can still check the Verilog generation:

```bash
python qwen/yosys_logic_export.py qwen/out/qwen_scope_layer6_en_top32_difflogic.json --skip-yosys
```

Example generated Verilog for the top-8 binary toxic circuit:

```verilog
// Original expression: (((i2 OR (i6 OR i7)) AND i1) OR ((i7 AND i2) OR i3))
module toxic_circuit(
    input wire i1, i2, i3, i4, i5, i6, i7, i8,
    output wire toxic
);
    assign toxic = (((i2 | (i6 | i7)) & i1) | ((i7 & i2) | i3));
endmodule
```
