# TODO Risk Analysis Report

**Generated**: 2025-11-15 10:09:30

## Executive Summary

- **Total Critical Path TODOs**: 33
  - Training path: 15
  - Conversion path: 18
- **"For Now" Implementations**: 20
- **High-Risk "For Now" in Critical Paths**: 8

### Risk Distribution

- **CRITICAL**: 3
- **HIGH**: 11
- **MEDIUM**: 19

## Critical Path TODO Inventory

### Training Path (15 TODOs)

#### `training/claim_extraction.py` (4 TODOs)

**Line 11** - Risk: **MEDIUM**

```
Claim extraction utilities for training dataset generation and loss computation. Provides simplified claim extraction for training purposes, focusing on: - Verifiable content detection - Atomic claim extraction - Claim extraction success rate measurement Reference: CLAIM_EXTRACTION_SKEPTICISM_GUARD_RAILS.md @author: @darianrosebrook
```

- Confidence: 0.85
- Context Score: -0.50

**Code Context:**

```python
       6: - Atomic claim extraction
       7: - Claim extraction success rate measurement
       8: 
       9: This is an intentionally simplified extractor optimized for training speed and efficiency.
      10: For production claim extraction with full verification capabilities, use ClaimifyPipeline
>>>   11: from arbiter.claims.pipeline.
      12: 
      13: Reference: CLAIM_EXTRACTION_SKEPTICISM_GUARD_RAILS.md
      14: @author: @darianrosebrook
      15: """
      16: 

```


**Line 20** - Risk: **MEDIUM**

```
Simplified claim representation for training.
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
      15: """
      16: 
      17: import re
      18: from typing import List, Dict, Any, Optional
      19: from dataclasses import dataclass
>>>   20: 
      21: 
      22: @dataclass
      23: class ExtractedClaim:
      24:     """Simplified claim representation for training."""
      25: 

```


**Line 36** - Risk: **MEDIUM**

```
Simplified claim extractor for training purposes. Detects verifiable claims using heuristics: - Factual indicators (dates, quantities, code references) - Structured content (code blocks, lists, JSON) - Atomic statements (single facts, not compound)
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
      31: 
      32: class SimpleClaimExtractor:
      33:     """
      34:     Simplified claim extractor for training purposes.
      35: 
>>>   36:     This is an intentionally simplified implementation optimized for training speed.
      37:     It uses heuristic-based detection rather than full NLP parsing, making it fast
      38:     but less comprehensive than ClaimifyPipeline.
      39: 
      40:     Detects verifiable claims using heuristics:
      41:     - Factual indicators (dates, quantities, code references)

```


**Line 148** - Risk: **MEDIUM**

```
Check if sentence has factual structure (simplified).
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
     143: 
     144:         # Check for factual statements (has subject + verb + object)
     145:         if self._has_factual_structure(sentence):
     146:             return True
     147: 
>>>  148:         return False
     149: 
     150:     def _has_unverifiable_content(self, sentence: str) -> bool:
     151:         """Check if sentence contains unverifiable/subjective content."""
     152:         for pattern in self.UNVERIFIABLE_PATTERNS:
     153:             if re.search(pattern, sentence, re.IGNORECASE):

```


#### `training/distill_kd.py` (1 TODOs)

**Line 1277** - Risk: **MEDIUM**

```
For now, check if it exists as a closure variable or use fallback
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
    1272:                             if tok_id is not None and tok_id < student_logits.size(-1):
    1273:                                 vocab_ids[marker] = tok_id
    1274:                         except Exception:
    1275:                             pass
    1276: 
>>> 1277:                 # Code-mode loss module should be initialized in main() and passed as parameter
    1278:                 # Fallback initialization only if not provided (for backward compatibility)
    1279:                 if code_mode_loss_module is None:
    1280:                     # Fallback: initialize on first call if not provided
    1281:                     # This maintains backward compatibility but is not recommended
    1282:                     code_mode_loss_module = CodeModePreferenceLoss(

```


#### `training/examples_priority3_integration.py` (5 TODOs)

**Line 183** - Risk: **HIGH**

```
Compute quality score for teacher output. This is a placeholder - in practice, you would use: - Human evaluation scores - Automated metrics (BLEU, ROUGE, etc.) - Model-based evaluation - Task-specific metrics Args: teacher_output: Teacher model generated text ground_truth: Optional ground truth text for comparison method: Scoring method ("heuristic", "bleu", "rouge", etc.) Returns: Quality score between 0.0 and 1.0
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
     178:     - Task-specific metrics (domain-dependent)
     179: 
     180:     Args:
     181:         teacher_output: Teacher model generated text
     182:         ground_truth: Optional ground truth text for comparison (required for BLEU)
>>>  183:         method: Scoring method ("heuristic", "bleu")
     184: 
     185:     Returns:
     186:         Quality score between 0.0 and 1.0
     187:     """
     188:     if method == "heuristic":

```


**Line 225** - Risk: **MEDIUM**

```
This is a simplified BLEU approximation without nltk
```

- Confidence: 0.94
- Context Score: -0.20

**Code Context:**

```python
     220:             # Try to use nltk if available
     221:             from nltk.translate.bleu_score import sentence_bleu
     222: 
     223:             reference = [ground_truth.split()]
     224:             candidate = teacher_output.split()
>>>  225:             bleu_score = sentence_bleu(reference, candidate)
     226:             return float(bleu_score)
     227:         except ImportError:
     228:             # Fallback: Simple n-gram overlap approximation
     229:             # This is a simplified BLEU approximation without nltk
     230:             reference_tokens = ground_truth.lower().split()

```


**Line 232** - Risk: **MEDIUM**

```
Unigram precision (simplified BLEU-1)
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
     227:         except ImportError:
     228:             # Fallback: Simple n-gram overlap approximation
     229:             # This is a simplified BLEU approximation without nltk
     230:             reference_tokens = ground_truth.lower().split()
     231:             candidate_tokens = teacher_output.lower().split()
>>>  232: 
     233:             if not reference_tokens or not candidate_tokens:
     234:                 return 0.0
     235: 
     236:             # Unigram precision (simplified BLEU-1)
     237:             reference_unigrams = set(reference_tokens)

```


**Line 239** - Risk: **MEDIUM**

```
Brevity penalty (simplified)
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
     234:                 return 0.0
     235: 
     236:             # Unigram precision (simplified BLEU-1)
     237:             reference_unigrams = set(reference_tokens)
     238:             candidate_unigrams = set(candidate_tokens)
>>>  239: 
     240:             matches = len(reference_unigrams & candidate_unigrams)
     241:             precision = matches / len(candidate_unigrams) if candidate_unigrams else 0.0
     242: 
     243:             # Brevity penalty (simplified)
     244:             brevity_penalty = (

```


**Line 244** - Risk: **MEDIUM**

```
Simplified BLEU score (unigram precision with brevity penalty)
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
     239: 
     240:             matches = len(reference_unigrams & candidate_unigrams)
     241:             precision = matches / len(candidate_unigrams) if candidate_unigrams else 0.0
     242: 
     243:             # Brevity penalty (simplified)
>>>  244:             brevity_penalty = (
     245:                 min(1.0, len(candidate_tokens) / len(reference_tokens)) if reference_tokens else 0.0
     246:             )
     247: 
     248:             # Simplified BLEU score (unigram precision with brevity penalty)
     249:             bleu_approx = precision * brevity_penalty

```


#### `training/losses.py` (1 TODOs)

**Line 892** - Risk: **HIGH**

```
Combine penalties (sum with equal weights for now)
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
     887:     # 2. Evaluate quality compliance (claim support, reasoning structure)
     888:     quality_penalty = _evaluate_quality_compliance(student_output, teacher_output, claim_extractor)
     889: 
     890:     # 3. Evaluate feature usage compliance (appropriate use of code-mode, latent reasoning)
     891:     feature_penalty = _evaluate_feature_usage_compliance(student_output)
>>>  892: 
     893:     # Combine penalties with configurable weights
     894:     # Quality is weighted higher as it directly affects output correctness
     895:     # Budget and feature usage are important but secondary concerns
     896:     total_loss = (
     897:         budget_weight * budget_penalty +

```


#### `training/quant_qat_int8.py` (2 TODOs)

**Line 279** - Risk: **HIGH**

```
For now, we'll use a simple approach: quantize the embedding weights
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
     274:     return QuantizedLinear(linear, weight_bits=weight_bits, act_bits=act_bits)
     275: 
     276: 
     277: def quantize_attention(
     278:     attn: MHA_GQA, weight_bits: int = 8, act_bits: int = 8, clamp_pre_softmax: bool = True
>>>  279: ) -> QuantizedAttention:
     280:     """Replace MHA_GQA with QuantizedAttention."""
     281:     return QuantizedAttention(
     282:         attn, weight_bits=weight_bits, act_bits=act_bits, clamp_pre_softmax=clamp_pre_softmax
     283:     )
     284: 

```


**Line 284** - Risk: **MEDIUM**

```
For now, we'll leave it as-is and document the trade-off
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
     279: ) -> QuantizedAttention:
     280:     """Replace MHA_GQA with QuantizedAttention."""
     281:     return QuantizedAttention(
     282:         attn, weight_bits=weight_bits, act_bits=act_bits, clamp_pre_softmax=clamp_pre_softmax
     283:     )
>>>  284: 
     285: 
     286: def quantize_swiglu(swiglu: SwiGLU, weight_bits: int = 8, act_bits: int = 8) -> nn.Module:
     287:     """Replace SwiGLU layers with quantized versions."""
     288:     # Create new SwiGLU with quantized linear layers
     289:     quantized = SwiGLU(swiglu.w1.in_features, swiglu.w1.out_features)

```


#### `training/run_toy_distill.py` (2 TODOs)

**Line 178** - Risk: **MEDIUM**

```
For now, assume 8-ball datasets may have them
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
     173:         print(f"[run_toy_distill] ERROR: Failed to load tokenizer: {e}")
     174:         sys.exit(1)
     175: 
     176:     # Create dataset
     177:     try:
>>>  178:         # Detect if dataset has teacher_logits by checking first sample
     179:         # This is more robust than assuming based on dataset type
     180:         teacher_logits_available = args.eight_ball
     181:         
     182:         # Try to peek at first sample to detect teacher_logits availability
     183:         try:

```


**Line 313** - Risk: **HIGH**

```
For now, weight all positions equally but this could be improved
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
     308:                 position_weights = torch.ones_like(labels, dtype=torch.float32, device=device)
     309:                 answer_position_weight = 3.0  # 3x higher weight for answer positions
     310:                 base_weight = 1.0
     311: 
     312:                 # For each sample in batch, identify mystical answer positions and weight them
>>>  313:                 for batch_idx_in_batch in range(labels.shape[0]):
     314:                     sample_data = batch["raw_data"][batch_idx_in_batch]
     315: 
     316:                     # Get mystical answer and find its position in the sequence
     317:                     mystical_answer = sample_data["metadata"]["mystical_answer"]
     318:                     full_text = sample_data["prompt"] + " " + sample_data["teacher_text"]

```



### Conversion Path (18 TODOs)

#### `conversion/convert_coreml.py` (7 TODOs)

**Line 11** - Risk: **HIGH**

```
Convert ONNX → CoreML (mlprogram). Uses public MIL converter API. Usage: python -m conversion.convert_coreml \ --backend onnx \ --in onnx/toy.sanitized.onnx \ --out coreml/artifacts/toy/model.mlpackage \ --target macOS13 \ --allow-placeholder   # optional; otherwise fails loud
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
       6:     --backend onnx \
       7:     --in onnx/toy.sanitized.onnx \
       8:     --out coreml/artifacts/toy/model.mlpackage \
       9:     --target macOS13 \
      10:     --allow-placeholder   # optional; otherwise fails loud
>>>   11: """
      12: 
      13: import warnings
      14: import argparse
      15: import os
      16: import sys

```


**Line 80** - Risk: **HIGH**

```
Convert PyTorch model (TorchScript or ExportedProgram) to CoreML. Args: pytorch_model: TorchScript module or torch.export.ExportedProgram output_path: Output path for .mlpackage compute_units: "all", "cpuandgpu", or "cpuonly" target: Deployment target (e.g., "macOS13") allow_placeholder: If True, create placeholder on failure instead of raising Returns: Path to converted model or None if placeholder created Note: torch and coremltools are imported inside the function to avoid import errors when 
```

- Confidence: 1.00
- Context Score: 0.30

**Code Context:**

```python
      75:         torch and coremltools are imported inside the function to avoid
      76:         import errors when these dependencies are not available. This allows
      77:         the module to be imported even if CoreML conversion dependencies
      78:         are missing, which is useful for environments that only need PyTorch
      79:         export functionality.
>>>   80:     """
      81:     # Import here to avoid import errors when coremltools is not available
      82:     # This allows the module to be imported even without CoreML dependencies
      83:     import torch
      84:     import coremltools as ct
      85: 

```


**Line 419** - Risk: **HIGH**

```
Convert ONNX model to CoreML using public MIL converter API. Args: onnx_path: Path to ONNX model file output_path: Output path for .mlpackage compute_units: "all", "cpuandgpu", or "cpuonly" target: Deployment target (e.g., "macOS13") allow_placeholder: If True, create placeholder on failure instead of raising Returns: Path to converted model or None if placeholder created
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
     414:         target: Deployment target (e.g., "macOS13")
     415:         allow_placeholder: If True, create placeholder on failure instead of raising
     416: 
     417:     Returns:
     418:         Path to converted model or None if placeholder created
>>>  419:     """
     420:     import coremltools as ct
     421:     import onnx
     422: 
     423:     cu_map = {
     424:         "all": ct.ComputeUnit.ALL,

```


**Line 473** - Risk: **CRITICAL**

```
ONNX is not a supported production path - always create placeholder
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
     468:             if output.name.startswith("var_") or output.name == "" or not output.name:
     469:                 output.name = "logits"
     470:                 print(f"[convert_coreml] Renamed output '{old_name}' to 'logits'")
     471: 
     472:     except Exception:
>>>  473:         # ONNX→CoreML conversion is not supported by CoreMLTools 9.0
     474:         # This is a documented limitation, not a bug
     475:         # Production path: Use PyTorch→CoreML conversion instead
     476:         if allow_placeholder:
     477:             print("[convert_coreml] WARN: ONNX conversion not supported by CoreMLTools")
     478:             print("[convert_coreml] Creating placeholder (SKIP parity)")

```


**Line 505** - Risk: **MEDIUM**

```
Create a placeholder .mlpackage for smoke tests.
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
     500:     output_path_obj.parent.mkdir(parents=True, exist_ok=True)
     501:     mlmodel.save(str(output_path_obj))
     502:     print(f"[convert_coreml] Saved → {output_path}")
     503:     return output_path
     504: 
>>>  505: 
     506: def create_placeholder(output_path: str, onnx_path: str, error_msg: str):
     507:     """Create a placeholder .mlpackage for smoke tests."""
     508:     out = Path(output_path)
     509:     out.mkdir(parents=True, exist_ok=True)
     510: 

```


**Line 509** - Risk: **MEDIUM**

```
Create .placeholder marker
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
     504: 
     505: 
     506: def create_placeholder(output_path: str, onnx_path: str, error_msg: str):
     507:     """Create a placeholder .mlpackage for smoke tests."""
     508:     out = Path(output_path)
>>>  509:     out.mkdir(parents=True, exist_ok=True)
     510: 
     511:     # Create .placeholder marker
     512:     placeholder_marker = out.parent / ".placeholder"
     513:     placeholder_marker.write_text("")
     514: 

```


**Line 543** - Risk: **MEDIUM**

```
Smoke test (creates placeholder on failure):
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
     538:         description="Convert ONNX model to CoreML mlprogram",
     539:         formatter_class=argparse.RawDescriptionHelpFormatter,
     540:         epilog="""
     541: Examples:
     542:   # Production conversion (fails loud if conversion unavailable):
>>>  543:   python -m conversion.convert_coreml --backend onnx --in model.onnx --out model.mlpackage
     544: 
     545:   # Smoke test (creates placeholder on failure):
     546:   python -m conversion.convert_coreml --backend onnx --in model.onnx --out model.mlpackage --allow-placeholder
     547:         """,
     548:     )

```


#### `conversion/export_pytorch.py` (1 TODOs)

**Line 108** - Risk: **MEDIUM**

```
For now, keep returning tensor directly - CoreML will name it
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
     103:         # Note: If wrapper returns dict, TorchScript will preserve structure
     104:         traced = torch.jit.trace(wrapper, example_input)
     105:         traced.eval()
     106: 
     107:         # If wrapper returns dict, we need to handle it differently
>>>  108:         # For now, keep returning tensor directly - CoreML will name it
     109:         # The output name stabilization happens in convert_coreml.py
     110: 
     111:     output_path.parent.mkdir(parents=True, exist_ok=True)
     112:     traced.save(str(output_path))
     113:     print(f"[export_pytorch] Saved prefill model: {output_path}")

```


#### `conversion/judge_export_coreml.py` (1 TODOs)

**Line 40** - Risk: **CRITICAL**

```
Convert Judge ONNX model to CoreML with INT8 quantization. Note: CoreMLTools does not natively support ONNX→CoreML conversion. For production, convert ONNX→PyTorch first, then use PyTorch→CoreML. INT8 quantization should be applied at the ONNX level before conversion, or use PyTorch quantization APIs. Args: onnx_path: Path to Judge ONNX model output_path: Output path for CoreML model compute_units: Compute units ("all", "cpuandgpu", "cpuonly") target: Deployment target (e.g., "macOS13", "macOS14
```

- Confidence: 1.00
- Context Score: 0.30

**Code Context:**

```python
      35:     
      36:     INT8 quantization should be applied at the ONNX level before conversion,
      37:     or use PyTorch quantization APIs.
      38: 
      39:     Args:
>>>   40:         onnx_path: Path to Judge ONNX model
      41:         output_path: Output path for CoreML model
      42:         compute_units: Compute units ("all", "cpuandgpu", "cpuonly")
      43:         target: Deployment target (e.g., "macOS13", "macOS14")
      44:         allow_placeholder: If True, create placeholder on failure instead of raising
      45:     """

```


#### `conversion/make_toy_block.py` (1 TODOs)

**Line 45** - Risk: **MEDIUM**

```
Simplified attention for parity testing (no GQA, no RoPE).
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
      40:         b = self.w3(x)
      41:         return self.w2(F.silu(a) * b)
      42: 
      43: 
      44: class MultiHeadAttention(nn.Module):
>>>   45:     """Simplified attention for parity testing.
      46:     
      47:     This is intentionally simplified for parity testing purposes:
      48:     - No GQA (Grouped Query Attention) - uses standard multi-head attention
      49:     - No RoPE (Rotary Position Embeddings) - uses standard positional encoding
      50:     

```


#### `coreml/ane_checks.py` (1 TODOs)

**Line 82** - Risk: **HIGH**

```
Check for placeholder marker
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
      77:     ap.add_argument(
      78:         "--min-ane-pct", type=float, default=0.80, help="Minimum ANE percentage (default: 0.80)"
      79:     )
      80:     args = ap.parse_args()
      81: 
>>>   82:     # Check for placeholder marker
      83:     mlpackage_path = Path(args.mlpackage)
      84:     is_placeholder = (mlpackage_path.parent / ".placeholder").exists() or (
      85:         mlpackage_path / ".placeholder"
      86:     ).exists()
      87: 

```


#### `coreml/probes/compare_probes.py` (2 TODOs)

**Line 36** - Risk: **HIGH**

```
Run CoreML model inference. Assumes placeholder check already done in main().
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
      31:     )[0]
      32:     return out
      33: 
      34: 
      35: def run_coreml(mlpackage_path, input_ids):
>>>   36:     """Run CoreML model inference.
      37:     
      38:     Note: Placeholder check should be performed in main() before calling this function.
      39:     This function assumes a valid CoreML model (not a placeholder).
      40:     """
      41:     import coremltools as ct

```


**Line 60** - Risk: **HIGH**

```
Check for placeholder marker
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
      55:     ap.add_argument("--onnx", required=False)
      56:     ap.add_argument("--ml", required=False)
      57:     ap.add_argument("--pt", required=False, help="PyTorch probe npz file")
      58:     ap.add_argument("--seq", type=int, default=128)
      59:     ap.add_argument("--dmodel", type=int, default=64)
>>>   60:     ap.add_argument("--rel_err_tol", type=float, default=0.02)
      61:     ap.add_argument("--mse_tol", type=float, default=1e-3)
      62:     args = ap.parse_args()
      63: 
      64:     # Check for placeholder marker
      65:     from pathlib import Path

```


#### `coreml/runtime/ane_monitor.py` (3 TODOs)

**Line 127** - Risk: **MEDIUM**

```
For now, we'll use a simplified approach:
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
     122:         # Note: Full Instruments integration would require:
     123:         # 1. Creating an Instruments trace template
     124:         # 2. Running inference under Instruments profiling
     125:         # 3. Parsing the trace file for ANE/GPU/CPU time
     126: 
>>>  127:         # For now, we'll use a simplified approach:
     128:         # Run inference and estimate based on timing patterns
     129:         # In production, you'd use Instruments.app GUI or command-line tools
     130: 
     131:         # NOTE: This is an intentional fallback implementation, not a placeholder.
     132:         # Full Instruments.app integration would require macOS-specific tooling.

```


**Line 131** - Risk: **HIGH**

```
NOTE: This is an intentional fallback implementation, not a placeholder.
```

- Confidence: 0.94
- Context Score: -0.20

**Code Context:**

```python
     126: 
     127:         # For now, we'll use a simplified approach:
     128:         # Run inference and estimate based on timing patterns
     129:         # In production, you'd use Instruments.app GUI or command-line tools
     130: 
>>>  131:         # NOTE: This is an intentional fallback implementation, not a placeholder.
     132:         # Full Instruments.app integration would require macOS-specific tooling.
     133:         print("[ane_monitor] WARN: Full Instruments integration not yet implemented")
     134:         print("[ane_monitor] Falling back to wall-clock sampling")
     135:         return self._measure_wall_clock(inference_fn, num_samples)
     136: 

```


**Line 189** - Risk: **CRITICAL**

```
This is a simplified heuristic - production would use more sophisticated analysis
```

- Confidence: 0.94
- Context Score: -0.20

**Code Context:**

```python
     184:         variance = np.var(timings_array)
     185:         # Coefficient of variation
     186:         cv = variance / (median_time**2) if median_time > 0 else 0
     187: 
     188:         # Estimate compute unit distribution based on timing characteristics
>>>  189:         # This is a heuristic fallback when Instruments.app is not available
     190:         # For production monitoring, use Instruments.app for accurate ANE/GPU/CPU breakdown
     191:         # This heuristic works reasonably well for development and smoke testing
     192:         if median_time < 50 and cv < 0.1:
     193:             # Fast and consistent → mostly ANE
     194:             ane_pct = 0.85

```


#### `coreml/runtime/constrained_decode.py` (1 TODOs)

**Line 150** - Risk: **MEDIUM**

```
This FSM is deliberately simplified: we enforce key/value separators, brackets balance,
```

- Confidence: 0.94
- Context Score: -0.20

**Code Context:**

```python
     145:             return set()  # nothing more
     146: 
     147:         # If we're not inside a string, we only allow structural chars, digits signs, t/f/n, quote
     148:         # If we are inside a string, anything except unescaped quotes; we allow \ escapes.
     149: 
>>>  150:         # This FSM is intentionally simplified: we enforce key/value separators, brackets balance,
     151:         # and string delimiters. This design is sufficient for practical tool-JSON generation
     152:         # and provides good performance with acceptable correctness guarantees.
     153: 
     154:         st.buffer.rstrip()
     155:         # Are we currently inside an open string?

```


#### `coreml/runtime/speculative_decode.py` (1 TODOs)

**Line 410** - Risk: **MEDIUM**

```
Simplified threshold-based acceptance (fallback)
```

- Confidence: 1.00
- Context Score: 0.00

**Code Context:**

```python
     405:             acceptance_prob = min(1.0, worker_prob / drafter_prob)
     406: 
     407:             # Sample random uniform [0, 1] and compare
     408:             return acceptance_prob >= random.uniform(0.0, 1.0)
     409:         else:
>>>  410:             # Threshold-based acceptance fallback (used when use_standard_acceptance=False)
     411:             # This is a simpler acceptance criterion that accepts tokens with probability >= 0.1
     412:             # Standard acceptance criterion is preferred for better distribution matching
     413:             worker_probs = np.exp(worker_logits - np.max(worker_logits))
     414:             worker_probs = worker_probs / np.sum(worker_probs)
     415:             worker_prob = worker_probs[draft_token]

```



## "For Now" Implementation Risk Analysis

### Summary

- **Total "For Now" Instances**: 20
- **In Critical Paths**: 8
  - Training: 6
  - Conversion: 2
  - Other: 12

### Distribution by Context

- **other**: 14
- **data**: 3
- **weights**: 2
- **quantization**: 1

### Top Files with "For Now" Implementations

- `evaluation/perf_mem_eval.py`: 3
- `training/quant_qat_int8.py`: 2
- `training/run_toy_distill.py`: 2
- `scripts/assess_readiness.py`: 2
- `training/losses.py`: 1
- `training/distill_kd.py`: 1
- `runtime/api_contract.py`: 1
- `coreml/runtime/ane_monitor.py`: 1
- `scripts/verify_contextual_set.py`: 1
- `scripts/inference_production.py`: 1

### High-Risk "For Now" Instances in Critical Paths

**`training/quant_qat_int8.py`** (Line 279)

```
For now, we'll use a simple approach: quantize the embedding weights
```

- Confidence: 1.00
- Path Type: training

**`training/quant_qat_int8.py`** (Line 284)

```
For now, we'll leave it as-is and document the trade-off
```

- Confidence: 1.00
- Path Type: training

**`training/run_toy_distill.py`** (Line 178)

```
For now, assume 8-ball datasets may have them
```

- Confidence: 1.00
- Path Type: training

**`training/run_toy_distill.py`** (Line 313)

```
For now, weight all positions equally but this could be improved
```

- Confidence: 1.00
- Path Type: training

**`training/losses.py`** (Line 892)

```
Combine penalties (sum with equal weights for now)
```

- Confidence: 1.00
- Path Type: training

**`training/distill_kd.py`** (Line 1277)

```
For now, check if it exists as a closure variable or use fallback
```

- Confidence: 1.00
- Path Type: training

**`coreml/runtime/ane_monitor.py`** (Line 127)

```
For now, we'll use a simplified approach:
```

- Confidence: 1.00
- Path Type: conversion

**`conversion/export_pytorch.py`** (Line 108)

```
For now, keep returning tensor directly - CoreML will name it
```

- Confidence: 1.00
- Path Type: conversion

## Risk Heat Map by Module

| Module | CRITICAL | HIGH | MEDIUM | LOW | Total |
|--------|----------|------|--------|-----|-------|
| `ane_checks.py` | 0 | 1 | 0 | 0 | 1 |
| `ane_monitor.py` | 1 | 1 | 1 | 0 | 3 |
| `claim_extraction.py` | 0 | 0 | 4 | 0 | 4 |
| `compare_probes.py` | 0 | 2 | 0 | 0 | 2 |
| `constrained_decode.py` | 0 | 0 | 1 | 0 | 1 |
| `convert_coreml.py` | 1 | 3 | 3 | 0 | 7 |
| `distill_kd.py` | 0 | 0 | 1 | 0 | 1 |
| `examples_priority3_integration.py` | 0 | 1 | 4 | 0 | 5 |
| `export_pytorch.py` | 0 | 0 | 1 | 0 | 1 |
| `judge_export_coreml.py` | 1 | 0 | 0 | 0 | 1 |
| `losses.py` | 0 | 1 | 0 | 0 | 1 |
| `make_toy_block.py` | 0 | 0 | 1 | 0 | 1 |
| `quant_qat_int8.py` | 0 | 1 | 1 | 0 | 2 |
| `run_toy_distill.py` | 0 | 1 | 1 | 0 | 2 |
| `speculative_decode.py` | 0 | 0 | 1 | 0 | 1 |

## Production Readiness Impact Assessment

**BLOCKER**: 3 CRITICAL risk TODOs must be resolved before production.

**WARNING**: 11 HIGH risk TODOs should be addressed before production.

### Impact by Category

- **Model Correctness**: 9 high-risk TODOs
- **Conversion Success**: 5 high-risk TODOs

