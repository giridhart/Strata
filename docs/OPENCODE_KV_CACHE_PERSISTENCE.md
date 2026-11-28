# Persisting Strata KV Cache Across Restarts with OpenCode

## Why OpenCode Re-Prefills After a Strata Restart

- **OpenCode is stateless:** OpenCode sends the full `messages[]` array on every request (standard OpenAI/Anthropic chat-completions behavior).
- **While Strata stays running:** Strata matches the shared token prefix against its live KV cache (and parked host-RAM conversations) and skips re-prefilling matching tokens.
- **When Strata restarts:** Volatile VRAM and host-RAM process memory are freed on exit, so the next turn from OpenCode triggers a full cold prefill unless the KV state was saved outside the process and restored.

---

## Built-In Strata Support (Verified on `b580-wip`)

### 1. Saving & Restoring KV Snapshots Across Restarts (`--slot-save-path`)

Strata already implements session-file persistence in `serve/server.py` and the SYCL engine (`sycl/src/program/generate.cpp`: `SAVE` / `RESTORE`, `conversation_file.cpp`).

Start Strata with `--slot-save-path DIR` (or `"slot_save_path"` in config):

- **RAM-speed persistence across `strata` restarts (huge-RAM hosts):**
  ```bash
  --slot-save-path /dev/shm/strata-sessions
  ```
  Files live in Linux `tmpfs` (RAM) outside the `strata` process, so they survive server restarts at memory bandwidth and are only cleared on a full machine reboot.
- **Disk persistence across full machine reboots:**
  ```bash
  --slot-save-path /path/to/nvme/strata-sessions
  ```

Save before stopping Strata and restore after Strata starts back up:

```bash
# Save slot 0 KV state before stopping Strata
curl -X POST "http://127.0.0.1:8080/slots/0?action=save" \
  -H "Content-Type: application/json" \
  -d '{"filename": "opencode.bin"}'

# Restore slot 0 KV state after Strata restarts
curl -X POST "http://127.0.0.1:8080/slots/0?action=restore" \
  -H "Content-Type: application/json" \
  -d '{"filename": "opencode.bin"}'
```

### 2. Multi-Session Host-RAM Cache (While Strata Stays Running)

When switching between multiple OpenCode sessions or subagents without restarting Strata, allocate host RAM for parked conversations:

```bash
--conversation-cache-mib 32768 --conversation-cache-slots 8
```

### 3. How Snapshots Map Back to OpenCode Sessions

No changes to OpenCode are required:

1. After `action=restore`, Strata holds the restored KV pages and token IDs.
2. When OpenCode sends its next request (with the full conversation history), Strata tokenizes the prompt and matches the **longest common token prefix** against the restored state, prefilling only the new tail.
3. Each snapshot header validates the model files, engine version, KV quantization (`--kv`), `--max-context`, and resolved RoPE settings before restoring.

### 4. Current Limitations & Caveats

- **Manual trigger:** Strata does not auto-save on `SIGTERM`/`SIGINT` or auto-restore on startup.
- **Single slot:** Works on slot 0 only (returns `501` with parallel requests or `--batch`).
- **Unsupported modes:** Not supported with `--layer-split`, `--peer-device`, `--batch`, or `--prompt-cache 0`.
- **Size & cleanup:** Expect ~`1.2 GB` per 63K-token conversation. Strata never auto-deletes old snapshot files and enforces `--session-min-free-mib` (default `4096`; `0` disables the check). Use `--session-save-reclaim` if RAM gets tight during a save.
- **Prefix stability:** Any change at the start of the prompt (such as a dynamic timestamp or reordered tool definitions) invalidates the prefix match.

---

## Automating Save/Restore with a Bash Wrapper for OpenCode

Add this `oc` function to `~/.bashrc` (or `~/.zshrc`) on the client machine. It automatically restores the project's KV snapshot when launching `opencode` and saves it via a `trap` when `opencode` exits (including `Ctrl+C`):

```bash
oc() {
  local STRATA_URL="${STRATA_URL:-http://<strata-host>:8080}"
  # One snapshot per project directory (plain filename required by Strata)
  local SNAP="opencode_$(basename "$PWD" | tr -cd 'A-Za-z0-9_-').bin"

  # Restore previous KV snapshot if present (silent on first-run 404)
  curl -sf -X POST "$STRATA_URL/slots/0?action=restore" \
    -H "Content-Type: application/json" \
    -d "{\"filename\": \"$SNAP\"}" >/dev/null 2>&1 \
    && echo "[strata] restored $SNAP"

  # Save KV snapshot automatically on interrupt/exit
  trap 'curl -sf -X POST "'"$STRATA_URL"'/slots/0?action=save" \
    -H "Content-Type: application/json" \
    -d "{\"filename\": \"'"$SNAP"'\"}" >/dev/null 2>&1 \
    && echo "[strata] saved '"$SNAP"'"' EXIT INT TERM

  command opencode "$@"
  local rc=$?
  trap - EXIT INT TERM

  # Explicit save on normal exit
  curl -sf -X POST "$STRATA_URL/slots/0?action=save" \
    -H "Content-Type: application/json" \
    -d "{\"filename\": \"$SNAP\"}" >/dev/null 2>&1 \
    && echo "[strata] saved $SNAP"
  return $rc
}
```

### Usage Tips

- Set `export STRATA_URL="http://<strata-host>:8080"` in your shell profile.
- If you restart Strata *while* OpenCode is still open, exit `oc` first (triggering the save) or include the `action=save` curl in your Strata stop script.
- Run `/compact` in OpenCode before planned model/config changes that invalidate cached KV snapshots.
