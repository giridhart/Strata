"""serve/runtime_info.py - Runtime options, defaults and environment snapshot for GET /v1/runtime.

Exposes the exact server CLI options, engine CLI options (explicit and effective with code defaults
and INFO overrides), and environment variables (explicit, config, wrapper defaults, and code defaults)
in effect for the running server so benchmark tools (`tools/strata_bench.py`) can record and compare
runs even when code defaults change between commits or options are passed via environment variables.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

SECRET_KEY = re.compile(r"(?i)key|token|secret|password|auth")
SAFE_SECRET_LIKE_ENV = frozenset({
    "NEOReadDebugKeys",
    "STRATA_COALESCE_TOKENS",
    "STRATA_PLE_KEY",
})
ENV_PREFIXES = (
    "STRATA_",
    "SYCL_",
    "ONEAPI_",
    "UR_",
    "IGC_",
    "ZES_",
    "ZE_",
    "LEVEL_ZERO_",
    "CM_",
    "NEO_",
    "INTEL_",
    "CUDA_",
    "HIP_",
    "HSA_",
    "ROCR_",
    "GPU_",
    "OMP_",
    "MKL_",
    "GOMP_",
    "KMP_",
)
EXTRA_ENV_NAMES = frozenset({
    "NEOReadDebugKeys",
    "OverrideDefaultFP64Settings",
    "CreateMultipleSubDevices",
})

# Default values for server CLI options in serve/server.py
SERVER_CLI_DEFAULTS: dict[str, object] = {
    "--engine": "mock",
    "--config": None,
    "--host": "127.0.0.1",
    "--port": 8095,
    "--ready-file": None,
    "--shutdown-timeout": None,
    "--parallel": None,
    "--batch-groups": None,
    "--max-queue": None,
    "--queue-timeout": None,
    "--batch-telemetry": None,
    "--offline": False,
    "--gpu": None,
    "--tokenizer": "pack/full/tokenizer",
    "--open": False,
    "--fit-max-tokens": True,
    "--no-api-key": False,
    "--mcp-config": None,
    "--codemode": False,
    "--codemode-lang": None,
    "--lazy": False,
    "--api-monitor": False,
    "--idle-unload": None,
    "--min-free-vram-mib": None,
    "--before-load": None,
    "--slot-save-path": None,
    "--replicas": None,
    "--split-serve": None,
}

SERVER_BOOL_FLAGS = frozenset({
    "--script-cycle",
    "--batch-telemetry",
    "--offline",
    "--open",
    "--fit-max-tokens",
    "--no-api-key",
    "--codemode",
    "--lazy",
    "--api-monitor",
})

SERVER_NEGATED_BOOL_FLAGS: dict[str, str] = {
    "--no-batch-telemetry": "--batch-telemetry",
    "--no-fit-max-tokens": "--fit-max-tokens",
}

ENGINE_FLAG_ALIASES: dict[str, str] = {
    "--slots": "--batch",
    "--parallel": "--batch",
}

# Boolean flags in the C++ engine CLI (take no value unless followed by 0/1 for --adapt-async / --pipeline-windows)
ENGINE_BOOL_FLAGS = frozenset({
    "--serve",
    "--vision",
    "--stream-experts",
    "--mmap-experts",
    "--resident-experts",
    "--resident-cpu-experts",
    "--no-prefill-borrow",
    "--prefill-head-alias",
    "--vram-elastic",
    "--kv-grow",
    "--trim-stage-weights",
    "--split-skip-if-fits",
    "--batch-mtp",
    "--prompt-cache-tail",
    "--session-save-reclaim",
    "--spec-split",
    "--coupled-draft",
    "--no-coupled-draft",
    "--mtp-hnorm-stream",
    "--expert-cache-per-layer",
    "--expert-cache-cpu-order",
    "--remote-expert-opt",
    "--no-ple",
    "--no-ple-prefetch",
    "--ple-sync-submit",
    "--no-capture",
    "--no-pool",
    "--no-host-worker",
    "--no-hit-poke",
    "--no-token-graph",
    "--no-fused-gr",
    "--no-fast-attn",
    "--no-publish-kernel",
    "--no-fused-gdn",
    "--no-fast-select",
    "--no-split-rows",
    "--stream-token",
    "--check-logits",
    "--gr-fp32-activations",
    "--gr-native-mmvf",
    "--native-bf16",
    "--native-bf16-extra",
    "--native-ple-key",
    "--native-moe-combine",
    "--native-gdn",
    "--native-flash-attn-short",
    "--native-qsa-indexer",
    "--native-qsa",
    "--native-rope",
    "--native-ple-postops",
    "--native-router",
    "--cpu-oracle-q8-0",
    "--sync-every-layer",
    "--stage-timing",
    "--graph-only",
    "--gpu-only-full",
    "--gpu-stages",
    "--stats",
    "--shared-late",
    "--keep-canonical",
    "--stop-eos",
})

# Compiled defaults for engine CLI options in generate.cpp
ENGINE_CLI_DEFAULTS: dict[str, object] = {
    "--pack": "pack/full",
    "--native": None,
    "--max-context": 4096,
    "--max-new": 16,
    "--kv": "fp16",
    "--kv-resident": 0,
    "--kv-grow": False,
    "--ple-io": "direct",
    "--ple-row-cache": 1048576,
    "--ple-inflight": 256,
    "--ple-sync-submit": False,
    "--no-ple": False,
    "--no-ple-prefetch": False,
    "--expert-cache": 0,
    "--expert-cache-per-layer": False,
    "--expert-cache-device1": 0,
    "--expert-cache-device2": 0,
    "--expert-cache-device3": 0,
    "--expert-cache-remote-placement": "stripe",
    "--expert-profile": None,
    "--expert-profile-save": None,
    "--expert-profile-save-min": 10.0,
    "--peer-device": -1,
    "--peer-reserve-mib": 600,
    "--peer-slots": 0,
    "--peer-adapt-swaps": -1,
    "--peer-prefill-rows": -1,
    "--vram-reserve-mib": 700,
    "--vram-reserve-later-mib": -1,
    "--vram-elastic": False,
    "--vram-segment-mib": 512,
    "--prefill": 0,
    "--no-prefill-borrow": False,
    "--prefill-head-alias": False,
    "--short-read": 64,
    "--spec": 0,
    "--mtp": None,
    "--mtp-window": 32768,
    "--mtp-max-t": 0,
    "--mtp-q4": None,
    "--mtp-hnorm-stream": False,
    "--suffix-draft": 3,
    "--lookup-chain": 0,
    "--lookup-chain-min": 3,
    "--spec-min-p": 0.0,
    "--spec-split": False,
    "--pcie-frac": -1.0,
    "--pcie-mode": "auto",
    "--adapt-every": 4,
    "--adapt-swaps": 96,
    "--adapt-decay": 0.7,
    "--adapt-async": 0,
    "--layer-split": None,
    "--split-device": None,
    "--split-skip-if-fits": False,
    "--pipeline-windows": -1,
    "--trim-stage-weights": False,
    "--batch": 0,
    "--batch-groups": 0,
    "--batch-mtp": False,
    "--prompt-cache": 6,
    "--prompt-cache-every": 16384,
    "--prompt-cache-root": 2048,
    "--prompt-cache-tail": False,
    "--turn-token": 248045,
    "--tail-role-token": -1,
    "--conversation-cache-mib": 0,
    "--conversation-cache-slots": 4,
    "--conversation-cache-min-free-mib": 2560,
    "--conversation-cache-min-tokens": 256,
    "--session-min-free-mib": 4096,
    "--session-save-reclaim": False,
    "--pool-workers": 0,
    "--pool-affinity": "all",
    "--host-core": None,
    "--pool-cores": None,
    "--no-host-worker": False,
    "--mmap-experts": False,
    "--stream-experts": False,
    "--resident-experts": False,
    "--resident-cpu-experts": False,
    "--resident-budget-gib": 0,
    "--shared-expert-arena": None,
    "--vision": False,
    "--no-hit-poke": False,
    "--no-token-graph": False,
    "--no-fused-gr": False,
    "--no-fast-attn": False,
    "--no-publish-kernel": False,
    "--no-fused-gdn": False,
    "--no-fast-select": False,
    "--no-split-rows": False,
    "--no-capture": False,
    "--no-pool": False,
}

# Defaults set by `--native` in generate.cpp before subsequent flags override them
NATIVE_PRESET_DEFAULTS: dict[str, object] = {
    "--kv": "int8",
    "--expert-cache-per-layer": True,
    "--gr-fp32-activations": True,
    "--gr-native-mmvf": True,
    "--native-bf16": True,
    "--native-bf16-extra": True,
    "--native-ple-key": True,
    "--native-ple-postops": True,
    "--native-moe-combine": True,
    "--native-gdn": True,
    "--native-router": True,
    "--native-qsa": True,
    "--native-qsa-indexer": True,
    "--native-rope": True,
}

# Code defaults for known STRATA_* and SYCL/Xe2 environment variables.
# Recording these in every benchmark run ensures that if a future commit changes a default
# in C++ or Python, `strata_bench.py --compare` surfaces the exact difference.
COMMON_ENV_DEFAULTS: dict[str, str] = {
    # Server runtime & scheduling
    "STRATA_WATCHDOG_S": "60",
    "STRATA_STREAM_WRITE_TIMEOUT_S": "45",
    "STRATA_START_RETRIES": "3",
    "STRATA_HTTP_BACKLOG": "128",
    "STRATA_MAX_BODY_MIB": "256",
    "STRATA_UNDECLARED_TOOL_CALLS": "reject",
    "STRATA_COALESCE_PREFILL": "0",
    "STRATA_COALESCE_TOKENS": "1024",
    "STRATA_COALESCE_WAIT_MS": "4",
    "STRATA_ADMIT_SHORT_FIRST": "0",
    "STRATA_BATCH_TELEMETRY": "0",
    # Engine general / adaptive / prefill / speculation
    "STRATA_PREFILL_AUTO_MAX": "8192",
    "STRATA_ADAPT_LAG": "0",
    "STRATA_ADAPT_OVERLAP": "1",
    "STRATA_ADAPT_NOWAIT": "1",
    "STRATA_ADAPT_EVICT_SYNC": "0",
    "STRATA_REFILL_BLOCKING": "0",
    "STRATA_SPEC_COUPLED": "1",
    "STRATA_SPEC_COST": "0",
    "STRATA_PIPELINE_AUTO": "1",
    "STRATA_SPLIT_OWN": "1",
    "STRATA_SPLIT_OWN_BUFFERS": "1",
    "STRATA_STAGE_TRIM": "0",
    "STRATA_BATCH_MTP": "0",
    "STRATA_KV_GROW": "0",
}

SYCL_ENV_DEFAULTS: dict[str, str] = {
    # SYCL server / watchdog
    "STRATA_WATCHDOG_S": "180",
    # SYCL command graph & doorbell verification
    "STRATA_VERIFY_GRAPH": "1",
    "STRATA_VERIFY_GRAPH_CHUNK": "0",
    "STRATA_VERIFY_DEVICE_PLAN": "0",
    "STRATA_VERIFY_NO_HOST": "0",
    "STRATA_VERIFY_RESIDENT_GRAPH": "0",
    "STRATA_VERIFY_LEAN": "1",
    "STRATA_VERIFY_STEPPED": "0",
    "STRATA_HOST_UNCACHED": "0",
    "STRATA_SPIN_MAX": "1000000",
    "STRATA_RING_TIMEOUT_S": "30",
    # SYCL / Xe2 MMVQ & MoE kernel knobs
    "STRATA_MMVQ_DECODE_256GRF": "1",
    "STRATA_MMVQ_LARGE_GRF": "1",
    "STRATA_MMVQ_SG": "32",
    "STRATA_B6_MMVQ_ROWS": "0",
    "STRATA_MMVF_ROWS": "0",
    "STRATA_MMVQ_A2": "1",
    "STRATA_MMVQ_WIDE": "1",
    "STRATA_MMVQ_KREG": "1",
    "STRATA_KREG2": "1",
    "STRATA_XE2_SPLIT_K": "1",
    "STRATA_XE2_SLM_PAD": "1",
    "STRATA_XE2_DUAL_COLUMN": "1",
    "STRATA_XE2_CACHE_HINT": "1",
    "STRATA_XE2_XMX_MOE": "1",
    "STRATA_XE2_IQ": "1",
    "STRATA_XE2_KQ": "1",
    "STRATA_IQ_DECODE_FAST": "1",
    "STRATA_IQ_GRID": "1",
    "STRATA_IQ_DT": "1",
    "STRATA_IQ3S_GU_ROW_REUSE": "1",
    # SYCL / Xe2 Attention & QSA knobs
    "STRATA_PROMPT_ATTN_ESIMD": "1",
    "STRATA_ESIMD_LV": "2",
    "STRATA_ATTN_MERGE_ACTIVE": "0",
    "STRATA_PA3_PF": "1",
    "STRATA_SCORES_MULTI": "1",
    "STRATA_TOPK_MULTI": "1",
    "STRATA_ATTN_LANECELL": "1",
    "STRATA_F16X_ANY": "1",
    "STRATA_F16X_SMALL": "1",
    "STRATA_I8_WG": "256",
    "STRATA_I8_GRF": "1",
    # SYCL / Xe2 Prefill & expert streaming knobs
    "STRATA_PREFILL_XMX_GEMM": "1",
    "STRATA_PF_PREDICT": "0",
    "STRATA_PF_PREDICT_MAX_T": "256",
    "STRATA_PREFILL_PREFETCH": "1",
    "STRATA_PF_EARLY_COPY": "1",
    "STRATA_PF_SHARED_FIRST": "1",
    "STRATA_STAGER_THREADS": "4",
    "STRATA_GROUP_COPY": "1",
    "STRATA_DEC_BATCH": "1",
    "STRATA_COPY_ENGINE": "1",
    "STRATA_CHUNKED_FILL": "1",
    "STRATA_ARENA_ALIAS_CHECK": "0",
}


def base_if_path(v):
    """Shorten filesystem paths to file names (or build-<dir>/<bin> for engine binaries) so personal paths never leak."""
    if not isinstance(v, str):
        return v
    if ("/" in v or "\\" in v) or re.search(r"\.(gguf|bin|json|pack|txt|jinja|so|dll|exe|sh|py)$", v, re.I):
        parts = [p for p in re.split(r"[\\/]", v.rstrip("\\/")) if p]
        if len(parts) >= 2 and (parts[-2].startswith("build-") or parts[-2] in ("serve", "sycl")):
            if len(parts) >= 3 and parts[-3] == "sycl" and parts[-2] == "serve":
                return "/".join(parts[-3:])
            return "/".join(parts[-2:])
        return parts[-1] if parts else v
    return v


def is_secret_env_name(name: str) -> bool:
    if name in SAFE_SECRET_LIKE_ENV:
        return False
    return bool(SECRET_KEY.search(name))


def is_relevant_env_name(name: str) -> bool:
    if is_secret_env_name(name):
        return False
    return name.startswith(ENV_PREFIXES) or name in EXTRA_ENV_NAMES


def _coerce_scalar(val: str):
    if not isinstance(val, str):
        return val
    s = val.strip()
    if re.fullmatch(r"-?\d+", s):
        try:
            return int(s)
        except ValueError:
            return val
    if re.fullmatch(r"-?\d+\.\d+", s):
        try:
            return float(s)
        except ValueError:
            return val
    return base_if_path(val)


def config_view(cfg: dict | None) -> dict:
    """Sanitize a strata-*.json config dict: drop secrets and shorten paths to file names."""
    if not isinstance(cfg, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        if SECRET_KEY.search(str(k)):
            continue
        if k == "args" and isinstance(v, list):
            out[k] = [base_if_path(x) for x in v]
        elif k == "env" and isinstance(v, dict):
            out[k] = {
                str(ek): base_if_path(ev)
                for ek, ev in v.items()
                if not is_secret_env_name(str(ek))
            }
        elif isinstance(v, str):
            out[k] = base_if_path(v)
        elif isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, list):
            out[k] = [base_if_path(x) for x in v]
        elif isinstance(v, dict):
            out[k] = {
                str(dk): base_if_path(dv)
                for dk, dv in v.items()
                if not SECRET_KEY.search(str(dk))
            }
    return out


def parse_cli_tokens(
    tokens: list[str],
    bool_flags: set[str] | None = None,
    negated_bool_flags: dict[str, str] | None = None,
    aliases: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Parse CLI argv tokens into (explicit_options_dict, sanitized_tokens)."""
    bool_flags = bool_flags or set()
    negated_bool_flags = negated_bool_flags or {}
    aliases = aliases or {}
    explicit: dict[str, Any] = {}
    sanitized: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = str(tokens[i])
        if tok in ("--api-key",):
            # Never record the raw API key value
            sanitized.append(tok)
            sanitized.append("<redacted>")
            explicit[tok] = "<redacted>"
            if i + 1 < n and not str(tokens[i + 1]).startswith("--"):
                i += 2
            else:
                i += 1
            continue
        if tok.startswith("--api-key="):
            sanitized.append("--api-key=<redacted>")
            explicit["--api-key"] = "<redacted>"
            i += 1
            continue
        if tok.startswith("-") and "=" in tok and tok.startswith("--"):
            flag, raw_val = tok.split("=", 1)
            canon = aliases.get(flag, flag)
            val = _coerce_scalar(raw_val)
            sanitized.append(f"{flag}={base_if_path(raw_val)}")
            explicit[canon] = val
            i += 1
            continue
        canon = aliases.get(tok, tok)
        if canon in negated_bool_flags:
            sanitized.append(tok)
            explicit[negated_bool_flags[canon]] = False
            ExplicitKey = canon
            explicit[ExplicitKey] = True
            i += 1
            continue
        if canon in bool_flags:
            sanitized.append(tok)
            explicit[canon] = True
            i += 1
            continue
        if tok.startswith("-"):
            sanitized.append(tok)
            if i + 1 < n:
                nxt = str(tokens[i + 1])
                # Negative numbers like -1 are values, not flags
                is_flag = nxt.startswith("-") and not re.fullmatch(r"-\d+(\.\d+)?", nxt)
                if not is_flag:
                    sanitized.append(str(base_if_path(nxt)))
                    val = _coerce_scalar(nxt)
                    if canon in explicit:
                        prev = explicit[canon]
                        if isinstance(prev, list):
                            prev.append(val)
                        else:
                            explicit[canon] = [prev, val]
                    else:
                        explicit[canon] = val
                    i += 2
                    continue
            explicit[canon] = True
            i += 1
            continue
        sanitized.append(str(base_if_path(tok)))
        i += 1
    return explicit, sanitized


def parse_wrapper_script(exe_path: str | None) -> dict[str, Any]:
    """Inspect a wrapper script such as `sycl/serve/strata-sycl.sh` to extract its default env vars and target binary."""
    result: dict[str, Any] = {"wrapper_defaults": {}, "default_sycl_bin": None}
    if not exe_path:
        return result
    p = Path(exe_path)
    if not p.is_file() or p.suffix not in (".sh", ""):
        return result
    try:
        head = p.read_bytes()[:4096]
    except OSError:
        return result
    if b"\x00" in head:
        # ELF/Mach-O/PE binary, not a shell wrapper
        return result
    try:
        text = head.decode("utf-8", errors="replace")
    except Exception:
        return result

    # 1. Match associative-array defaults like: [STRATA_VERIFY_DEVICE_PLAN]=1
    for m in re.finditer(r"\[([A-Za-z_][A-Za-z0-9_]*)\]=([^\s\)]+)", text):
        var_name, default_val = m.group(1), m.group(2).strip("\"'")
        if not is_secret_env_name(var_name):
            result["wrapper_defaults"][var_name] = str(base_if_path(default_val))

    # 2. Match ${VAR:-default} expansions anywhere in the wrapper script
    for m in re.finditer(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}$]*)\}", text):
        var_name, default_val = m.group(1), m.group(2).strip("\"'")
        if not default_val:
            continue
        if var_name == "STRATA_SYCL_BIN":
            result["default_sycl_bin"] = str(base_if_path(default_val))
        elif not is_secret_env_name(var_name):
            result["wrapper_defaults"][var_name] = str(base_if_path(default_val))

    # 3. Match patterns like: export VAR="${VAR:-val}" or VAR="${VAR:-val}"
    for m in re.finditer(
        r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=\"\$\{\1:-([^}]*)\}\"",
        text,
        re.MULTILINE,
    ):
        var_name, default_val = m.group(1), m.group(2)
        if var_name == "STRATA_SYCL_BIN":
            result["default_sycl_bin"] = str(base_if_path(default_val))
        elif not is_secret_env_name(var_name):
            result["wrapper_defaults"][var_name] = str(base_if_path(default_val))
    return result


def parse_server_cli(
    launch_args: list[str] | None = None,
    parsed_args: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
    svc: Any = None,
    script: str | None = None,
) -> dict[str, Any]:
    """Build structured metadata for the Python server's CLI options, defaults, and config."""
    raw_tokens = [str(x) for x in (launch_args or [])]
    explicit, sanitized_argv = parse_cli_tokens(
        raw_tokens,
        bool_flags=SERVER_BOOL_FLAGS,
        negated_bool_flags=SERVER_NEGATED_BOOL_FLAGS,
        aliases={"--codemode-language": "--codemode-lang"},
    )
    defaults = dict(SERVER_CLI_DEFAULTS)
    options = dict(defaults)

    # Merge values from argparse namespace if available
    if isinstance(parsed_args, dict):
        arg_to_flag = {
            "engine": "--engine",
            "config": "--config",
            "host": "--host",
            "port": "--port",
            "ready_file": "--ready-file",
            "shutdown_timeout": "--shutdown-timeout",
            "parallel": "--parallel",
            "batch_groups": "--batch-groups",
            "max_queue": "--max-queue",
            "queue_timeout": "--queue-timeout",
            "batch_telemetry": "--batch-telemetry",
            "offline": "--offline",
            "gpu": "--gpu",
            "tokenizer": "--tokenizer",
            "open": "--open",
            "fit_max_tokens": "--fit-max-tokens",
            "no_api_key": "--no-api-key",
            "mcp_config": "--mcp-config",
            "codemode": "--codemode",
            "codemode_lang": "--codemode-lang",
            "lazy": "--lazy",
            "api_monitor": "--api-monitor",
            "idle_unload": "--idle-unload",
            "min_free_vram_mib": "--min-free-vram-mib",
            "before_load": "--before-load",
            "slot_save_path": "--slot-save-path",
            "replicas": "--replicas",
            "split_serve": "--split-serve",
        }
        for attr, flag in arg_to_flag.items():
            if attr in parsed_args:
                val = parsed_args[attr]
                options[flag] = base_if_path(val) if isinstance(val, str) else val

    # Overlay explicit CLI options
    for k, v in explicit.items():
        options[k] = v

    # Overlay live Service fields when available
    if svc is not None:
        if getattr(svc, "max_queue", None) is not None:
            options["--max-queue"] = svc.max_queue
        if getattr(svc, "queue_timeout_s", None) is not None:
            options["--queue-timeout"] = svc.queue_timeout_s
        if getattr(svc, "fit_max_tokens", None) is not None:
            options["--fit-max-tokens"] = bool(svc.fit_max_tokens)
        if getattr(svc, "offline", None) is not None:
            options["--offline"] = bool(svc.offline)

    cfg_path = None
    if isinstance(parsed_args, dict) and parsed_args.get("config"):
        cfg_path = base_if_path(str(parsed_args["config"]))
    elif "--config" in explicit and isinstance(explicit["--config"], str):
        cfg_path = base_if_path(explicit["--config"])

    return {
        "script": base_if_path(script or "serve/server.py"),
        "argv": sanitized_argv,
        "explicit_options": explicit,
        "defaults": defaults,
        "options": options,
        "config_path": cfg_path,
        "config": config_view(cfg),
    }


def parse_engine_cli(
    exe: str | None,
    args: list[str] | None,
    info: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build structured metadata for the C++ engine's CLI options, defaults, and reported INFO."""
    raw_args = [str(x) for x in (args or [])]
    explicit, sanitized_argv = parse_cli_tokens(
        raw_args,
        bool_flags=ENGINE_BOOL_FLAGS,
        aliases=ENGINE_FLAG_ALIASES,
    )
    defaults = dict(ENGINE_CLI_DEFAULTS)
    options = dict(defaults)

    # Apply `--native` preset if `--native` was passed before individual overrides
    if explicit.get("--native"):
        options.update(NATIVE_PRESET_DEFAULTS)
    for k, v in explicit.items():
        options[k] = v

    # Overlay resolved runtime parameters reported by the C++ engine's INFO line
    clean_info: dict[str, Any] = {}
    if isinstance(info, dict):
        for k, v in info.items():
            clean_info[str(k)] = base_if_path(v) if isinstance(v, str) else v
        info_to_opt = {
            "context": "--max-context",
            "kv": "--kv",
            "expert_slots": "--expert-cache",
            "spec": "--spec",
            "batch_slots": "--batch",
            "batch_groups": "--batch-groups",
            "prefill_cap": "--prefill",
            "vram_reserve_mib": "--vram-reserve-mib",
            "adapt_every": "--adapt-every",
            "adapt_swaps": "--adapt-swaps",
            "ple_io": "--ple-io",
            "pipeline_windows": "--pipeline-windows",
            "short_read": "--short-read",
            "prompt_cache": "--prompt-cache",
            "prompt_cache_every": "--prompt-cache-every",
            "adapt_async": "--adapt-async",
        }
        for ik, flag in info_to_opt.items():
            if ik in clean_info:
                options[flag] = clean_info[ik]
        info_bool_to_opt = {
            "no_prefill_borrow": "--no-prefill-borrow",
            "mmap_experts": "--mmap-experts",
            "stream_experts": "--stream-experts",
            "vram_elastic": "--vram-elastic",
            "kv_grow": "--kv-grow",
        }
        for ik, flag in info_bool_to_opt.items():
            if ik in clean_info:
                options[flag] = bool(clean_info[ik])

    wrapper = parse_wrapper_script(exe)
    sycl_bin = None
    if env and env.get("STRATA_SYCL_BIN"):
        sycl_bin = base_if_path(env["STRATA_SYCL_BIN"])
    elif wrapper.get("default_sycl_bin"):
        sycl_bin = wrapper["default_sycl_bin"]

    return {
        "exe": base_if_path(exe) if exe else None,
        "sycl_bin": sycl_bin,
        "argv": sanitized_argv,
        "explicit_options": explicit,
        "defaults": defaults,
        "options": options,
        "info": clean_info,
    }


def collect_env(
    engine_env: dict[str, str] | None = None,
    cfg: dict[str, Any] | None = None,
    exe: str | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Collect explicit, config, wrapper, and code-default environment variables."""
    raw_env = dict(engine_env if engine_env is not None else os.environ)
    cfg_env_raw = (cfg or {}).get("env") if isinstance(cfg, dict) else None
    cfg_env: dict[str, str] = {}
    if isinstance(cfg_env_raw, dict):
        for k, v in cfg_env_raw.items():
            ks = str(k)
            if not is_secret_env_name(ks):
                cfg_env[ks] = str(base_if_path(str(v)))

    wrapper = parse_wrapper_script(exe)
    wrapper_defaults: dict[str, str] = dict(wrapper.get("wrapper_defaults") or {})

    # Determine whether SYCL defaults apply
    is_sycl = (
        (backend and str(backend).lower() == "sycl")
        or (isinstance(cfg, dict) and str(cfg.get("backend", "")).lower() == "sycl")
        or (exe and "sycl" in str(exe).lower())
        or bool(wrapper_defaults)
    )

    defaults: dict[str, str] = dict(COMMON_ENV_DEFAULTS)
    if is_sycl:
        defaults.update(SYCL_ENV_DEFAULTS)

    explicit: dict[str, str] = {}
    for k, v in sorted(raw_env.items()):
        if is_relevant_env_name(k):
            explicit[k] = str(base_if_path(str(v)))

    effective: dict[str, str] = {}
    sources: dict[str, str] = {}

    for k, v in sorted(defaults.items()):
        effective[k] = v
        sources[k] = "default"

    for k, v in sorted(wrapper_defaults.items()):
        effective[k] = v
        sources[k] = "wrapper_default"

    for k, v in sorted(cfg_env.items()):
        effective[k] = v
        sources[k] = "config"

    for k, v in sorted(explicit.items()):
        effective[k] = v
        sources[k] = "config" if (k in cfg_env and cfg_env[k] == v) else "env"

    return {
        "explicit": explicit,
        "config_env": cfg_env,
        "wrapper_defaults": wrapper_defaults,
        "defaults": defaults,
        "effective": effective,
        "sources": sources,
    }


def snapshot(svc: Any) -> dict[str, Any]:
    """Assemble the complete `/v1/runtime` payload from a running Service instance."""
    eng = getattr(svc, "engine", None)
    if eng is None:
        eng = getattr(svc, "eng", None)
    raw_cfg = getattr(svc, "runtime_cfg", None)
    cfg = raw_cfg if isinstance(raw_cfg, dict) else None
    spawn = getattr(eng, "spawn", None)
    exe = None
    eng_args: list[str] = []
    spawn_env = None
    if isinstance(spawn, (tuple, list)) and len(spawn) >= 2:
        exe = spawn[0] if isinstance(spawn[0], str) else None
        eng_args = list(spawn[1] or []) if isinstance(spawn[1], (list, tuple)) else []
        if len(spawn) >= 5 and isinstance(spawn[4], dict):
            spawn_env = spawn[4]
    if exe is None:
        eng_exe = getattr(eng, "exe", None)
        exe = eng_exe if isinstance(eng_exe, str) else (cfg.get("exe") if isinstance(cfg, dict) else None)
    if not eng_args:
        raw_eng_args = getattr(eng, "args", None)
        if not isinstance(raw_eng_args, (list, tuple)):
            raw_eng_args = cfg.get("args") if isinstance(cfg, dict) else None
        eng_args = list(raw_eng_args or [])
    raw_info = getattr(eng, "info", None)
    eng_info = raw_info if isinstance(raw_info, dict) else {}
    svc_env = getattr(svc, "engine_env", None)
    eng_env_attr = getattr(eng, "env", None)
    eng_env = (
        svc_env if isinstance(svc_env, dict)
        else (spawn_env or (eng_env_attr if isinstance(eng_env_attr, dict) else None))
    )
    raw_argv = getattr(svc, "launch_argv", None)
    launch_argv = list(raw_argv) if isinstance(raw_argv, (list, tuple)) else None
    raw_parsed = getattr(svc, "parsed_args", None)
    parsed_args = raw_parsed if isinstance(raw_parsed, dict) else None
    raw_script = getattr(svc, "launch_script", None)
    launch_script = raw_script if isinstance(raw_script, str) else None
    backend = eng_info.get("backend") if isinstance(eng_info, dict) else None

    server_data = parse_server_cli(
        launch_args=launch_argv,
        parsed_args=parsed_args,
        cfg=cfg,
        svc=svc,
        script=launch_script,
    )
    engine_data = parse_engine_cli(
        exe=exe,
        args=eng_args,
        info=eng_info,
        env=eng_env,
    )
    env_data = collect_env(
        engine_env=eng_env,
        cfg=cfg,
        exe=exe,
        backend=backend,
    )
    return {
        "server": server_data,
        "engine": engine_data,
        "env": env_data,
    }

