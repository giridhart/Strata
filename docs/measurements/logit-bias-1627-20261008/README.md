# Chat logit-bias validation — issue #1627

The API previously accepted `logit_bias` without applying it. This change carries
the field into native target sampling and rejects unsupported configurations.
No cache correction, private profile, model weights, or unrelated integration
work is included in this branch.

## Live result

**52 HTTP checks passed: 26 on CUDA and 26 on HIP**, each across an unchanged
main binary, the candidate without MTP, and the candidate with MTP. All requests used a loopback HTTP server and
recorded emitted token IDs at the native-engine boundary.

The primary usage example is now English: **Hello → Hi**, with an explanation of
why token 9419 was selected and full before/after requests in
[LOGIT_BIAS.md](../../LOGIT_BIAS.md#example-hello-becomes-hi).
Four additional CUDA requests passed: no bias gives `Hello`, object and pair-list
bans of 9419 give `Hi`, then omitting the field restores `Hello`.
These are [separate receipts](english/rows.json) from the 52-check CUDA/HIP matrix,
with a [completion receipt](english/complete.json) and [engine log](english/engine.txt).

The original issue reproduction is preserved below. For the prompt
"Write the Chinese word for hello. Only the word.":

| Request | Output | First token | Result |
|---|---|---:|---|
| No bias | 你好 | 109266 | Baseline |
| `{"109266": -100}` | 您好 | 109430 | Banned token absent |
| `[[109266, false]]` | 您好 | 109430 | Same result as object |
| Next request omits bias | 你好 | 109266 | Previous bias cleared |

The same checks passed on both GPUs, with and without MTP. A positive bias selected the boosted token;
streaming and sampled requests excluded the banned token. A 103,215-entry ban
list produced no excluded IDs. This large list is a transport/masking fixture,
not a language-filter quality evaluation.

Out-of-vocabulary IDs and out-of-range values returned HTTP 400 even when the
request asked for streaming. Empty biases restored default behavior. Two
matched no-bias prompts (greedy and seeded sampling) produced identical token
sequences with the pristine baseline and candidate binaries. This is a focused
default-path regression check, not proof for every prompt or sampling mode.

## Builds and tests

| Check | CUDA / llm-60 | HIP / llm-79 |
|---|---|---|
| Engine build | Pass | Pass |
| Native bias parser, including 103,215 bans | Pass | Pass |
| Python bias/server tests | 276 passed | 276 passed |
| GPU sampler parity, default/one-block/old paths | 3 passed | 3 passed |
| Real-model HTTP requests | 26 checks passed | 26 checks passed |

GPU parity covers positive and negative biases, greedy and sampled choices,
penalties, all target rows, clearing the device bias, unmodified raw logits,
and a 248,320-token vocabulary with only one permitted token.

Base: upstream main `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`.
CUDA hardware: RTX PRO 6000 Blackwell 96 GB, CUDA 13.2. Model: ISTA IQ3_XXS,
INT8 KV, 32,768-token allocated context. Main and candidate used the same model
and settings. The MTP arm used `--spec 4 --mtp`; the non-MTP arms used `--spec 2`.
Requests were deliberately short (2–8 output tokens) to test semantics; do not
use their timings as throughput benchmarks. MTP acceptance counters in the raw
rows confirm that the draft path was active.

HIP was compiled for gfx1100 and tested on llm-79, Radeon RX 7900 XTX 24 GB,
with ROCm and mmap-backed CPU expert offload. Each backend was compared against
its own pristine-main build; this is not a cross-hardware speed comparison. SYCL has its own unchanged engine/sampler sources
and does not advertise support; the Python capability gate rejects its biased
requests. Continuous batching is explicitly unsupported. Multi-GPU, startup
bias files, and experimental coupled/probabilistic MTP were not validated.

## Reproduce and inspect

- [Usage and request formats](../../LOGIT_BIAS.md).
- `python -m unittest serve.test_logit_bias serve.test_server`
- Build with `STRATA_BUILD_TESTS=ON`, then run `logit_bias_test` and the three
  `sampler_parity` CTest cases on a supported CUDA GPU.
- [Generic HTTP probe](../../../tools/logit_bias_http_probe.py) accepts paths to
  the source tree, model configuration, baseline binary, candidate binary, MTP
  runtime, and output directory. See its `--help`.
- [All HTTP responses, token IDs and counters](live/rows.json),
  [completion receipt](live/complete.json).
- [CUDA build/hash](cuda/build-state.json), [parser](cuda/parser.txt),
  [Python tests](cuda/python.txt), [sampler parity](cuda/sampler.txt).
- [HIP build/hash](hip/build-state.json), [parser](hip/parser.txt),
  [Python tests](hip/python.txt), [GPU sampler parity](hip/sampler.txt).
- HIP [HTTP responses and token IDs](hip-live/rows.json),
  [completion receipt](hip-live/complete.json), [job receipt](hip/live-state.json),
  engine logs: [baseline](hip-live/baseline.txt), [candidate](hip-live/off.txt),
  [MTP candidate](hip-live/mtp.txt).
- Engine logs: [baseline](live/baseline.txt), [candidate](live/off.txt),
  [candidate with MTP](live/mtp.txt).

The public probe replaces host-specific paths from the CUDA harness with
command-line arguments. CUDA receipts come from the original harness; the HIP
campaign executed this public version.
