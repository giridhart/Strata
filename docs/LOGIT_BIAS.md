# Per-request token biases

`POST /v1/chat/completions` accepts `logit_bias` with a supporting CUDA or HIP
engine, without continuous batching. The same bias applies to every generated
token in the request, including tokens checked together during speculative
decoding. It does not force single-token verification windows.

Supply an OpenAI-style object mapping vocabulary token IDs to numeric biases:

```json
{
  "model": "your-model",
  "messages": [{"role": "user", "content": "Write the Chinese word for hello. Only the word."}],
  "max_tokens": 128,
  "temperature": 0,
  "logit_bias": {"109266": -100}
}
```

The equivalent llama.cpp-style pair list is `"logit_bias": [[109266, false]]`.
Numeric pair values are also supported, for example `[[42, 2.5]]`.
Token IDs belong to the loaded model's tokenizer; the example ID is not portable
between arbitrary models.

Biases must be finite numbers in `[-100, 100]`. Negative values discourage a
token, positive values encourage it, and **-100 excludes it** (`false` in a
pair list has the same meaning). The bias is added before repetition/frequency/
presence penalties, temperature, and candidate filtering. Raw model logits
are not overwritten. Banning a token does not ban a word or language: other
token sequences may produce the same text.

Omitting the field, `null`, `{}`, or `[]` clears the previous request's bias.
Biases are request sampling state; they are not stored in a conversation cache.
They do not change the model weights or prompt KV. A fixed dense vocabulary
vector is uploaded when the bias changes and reused within the request.

Malformed entries, duplicate IDs in pair lists, IDs outside the vocabulary,
out-of-range biases, or banning the entire vocabulary return HTTP 400. Validation
on the chat route occurs before SSE streaming headers are sent. Nonempty biases
also return HTTP 400 with older engines, the current SYCL engine, or continuous
batching (`--batch`), rather than being silently ignored. The engine advertises
`logit_bias=1` in its `INFO` response when this implementation is present.

MTP target verification applies the bias; the draft model is not biased, so a
restrictive list may reduce draft acceptance. Ordinary MTP was tested. The
experimental coupled/rejection-sampling combinations and multi-GPU execution
were not validated in this campaign. There is no startup bias-file option in
this change: send the field with each request.

See the [validation report](measurements/logit-bias-1627-20261008/README.md)
for exact scope, examples, and raw receipts.
