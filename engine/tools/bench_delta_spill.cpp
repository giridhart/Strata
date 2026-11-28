#include "strata/core/conversation_file.hpp"
#include "strata/core/conversation_kv_store.hpp"
#include "strata/core/conversation_spill.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace strata::core;
using Clock = std::chrono::steady_clock;

namespace {

SavedConversation make_mock_conversation(size_t num_layers, size_t k_bytes_per_layer, size_t v_bytes_per_layer, size_t tokens) {
    SavedConversation sc;
    sc.geometry = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    sc.live.ids.resize(tokens);
    for (size_t i = 0; i < tokens; ++i) sc.live.ids[i] = (int32_t) (i + 1);
    sc.live.gdn.resize(64 * 1024, 0x5a); // 64 KB GDN state
    sc.kv.resize(num_layers);

    for (size_t j = 0; j < num_layers; ++j) {
        sc.kv[j].format = 0;
        sc.kv[j].cells = (int64_t) tokens;
        sc.kv[j].heads = 8;
        sc.kv[j].head_dim = 128;
        sc.kv[j].page_size = 64;
        sc.kv[j].pooled_rows = 0;
        sc.kv[j].idx_dim = 0;

        sc.kv[j].k.resize_for_overwrite(k_bytes_per_layer);
        sc.kv[j].k.visit(0, k_bytes_per_layer, [](uint8_t* p, size_t n, size_t at) {
            std::memset(p, (uint8_t) (at & 0xff), n);
            return true;
        });

        sc.kv[j].v.resize_for_overwrite(v_bytes_per_layer);
        sc.kv[j].v.visit(0, v_bytes_per_layer, [](uint8_t* p, size_t n, size_t at) {
            std::memset(p, (uint8_t) ((at + 1) & 0xff), n);
            return true;
        });
    }
    return sc;
}

std::vector<SessionKvSource> make_sources_from_kv(const std::vector<ConversationKv>& kv) {
    std::vector<SessionKvSource> sources(kv.size());
    for (size_t j = 0; j < kv.size(); ++j) {
        sources[j].format = kv[j].format;
        sources[j].cells = kv[j].cells;
        sources[j].heads = kv[j].heads;
        sources[j].head_dim = kv[j].head_dim;
        sources[j].page_size = kv[j].page_size;
        sources[j].pooled_rows = kv[j].pooled_rows;
        sources[j].idx_dim = kv[j].idx_dim;
        sources[j].sizes = {kv[j].k.size(), kv[j].v.size(), 0, 0, 0};

        const auto* k_buf = &kv[j].k;
        const auto* v_buf = &kv[j].v;
        sources[j].read = [k_buf, v_buf](size_t part, size_t offset, void* dst, size_t n) {
            if (part == 0) {
                return k_buf->read(dst, offset, n);
            } else if (part == 1) {
                return v_buf->read(dst, offset, n);
            }
            return false;
        };
        sources[j].error = [] { return "read error"; };
    }
    return sources;
}

} // namespace

int main(int argc, char** argv) {
    const size_t num_layers = 32;
    const size_t base_part_mb = (argc > 1) ? std::stoul(argv[1]) : 8; // 8 MB per part -> 16 MB per layer -> 512 MB total
    const size_t delta_kb = (argc > 2) ? std::stoul(argv[2]) : 512;   // 512 KB per part -> 1 MB per layer -> 32 MB delta

    const size_t base_k = base_part_mb * 1024 * 1024;
    const size_t base_v = base_part_mb * 1024 * 1024;
    const size_t delta_bytes = delta_kb * 1024;

    const size_t updated_k = base_k + delta_bytes;
    const size_t updated_v = base_v + delta_bytes;

    const size_t total_base_kv = num_layers * (base_k + base_v);
    const size_t total_updated_kv = num_layers * (updated_k + updated_v);
    const size_t total_delta_bytes = num_layers * (delta_bytes * 2);

    std::printf("=================================================================================\n");
    std::printf(" Strata Delta SSD Conversation Spill Benchmark\n");
    std::printf(" Configuration: %zu layers, Initial KV: %.2f MB (32 x %zu MB), Delta: %.2f MB\n",
                num_layers, (double) total_base_kv / (1024 * 1024), base_part_mb * 2,
                (double) total_delta_bytes / (1024 * 1024));
    std::printf("=================================================================================\n\n");

    const fs::path temp_dir = fs::temp_directory_path() / ("strata_spill_bench_" + std::to_string(::getpid()));
    fs::create_directories(temp_dir);

    struct Cleanup {
        fs::path p;
        ~Cleanup() { std::error_code ec; fs::remove_all(p, ec); }
    } cleanup{temp_dir};

    const SessionFileIdentity id{0x12345678, 0x87654321};

    // Prepare initial and updated mock conversations
    std::printf("[Setup] Allocating initial and updated conversation states in memory...\n");
    SavedConversation base_conv = make_mock_conversation(num_layers, base_k, base_v, 8192);
    SavedConversation updated_conv = make_mock_conversation(num_layers, updated_k, updated_v, 8704);

    // Setup KV Store Layout
    KvStoreLayout layout;
    layout.capacity.resize(num_layers);
    for (size_t j = 0; j < num_layers; ++j) {
        layout.capacity[j] = {32ull * 1024 * 1024, 32ull * 1024 * 1024, 65536, 65536, 1024 * 1024}; // 32MB max per part
    }

    std::string err;

    // 1. Benchmark Drive Speed Probe
    std::printf("\n--- 1. SSD Drive Speed Probe ---\n");
    SessionDriveSpeed drive_speed;
    auto t0 = Clock::now();
    bool probe_ok = session_drive_probe(temp_dir.string(), 64 * 1024 * 1024, 2.0, drive_speed, err);
    auto t1 = Clock::now();
    if (probe_ok) {
        std::printf("Drive Probe: Write: %.1f MB/s | Read: %.1f MB/s | Direct: %s | Elapsed: %.2f ms\n",
                    drive_speed.write_mbps, drive_speed.read_mbps, drive_speed.direct ? "YES (O_DIRECT)" : "NO (buffered)",
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
    } else {
        std::printf("Drive Probe skipped or error: %s\n", err.c_str());
    }

    // 2. Whole-File Initial Park vs KV Store Initial Park
    std::printf("\n--- 2. Initial Conversation Spill (Turn 1: %.2f MB) ---\n", (double) total_base_kv / (1024 * 1024));
    const std::string whole_file_path = (temp_dir / "whole_session.sess").string();
    const std::string kv_store_path = (temp_dir / "delta_store.kv").string();

    // 2a. Whole-file write initial
    t0 = Clock::now();
    SessionWriteOptions write_opt;
    write_opt.durable = false; // direct disk write
    size_t wf_init_bytes = 0;
    bool wf_ok = session_file_write(whole_file_path, base_conv, id, wf_init_bytes, err, write_opt);
    t1 = Clock::now();
    double wf_init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    uint64_t wf_init_size = fs::file_size(whole_file_path);
    std::printf("Whole-file Park (Initial): %.2f ms | File Size: %.2f MB | Write Throughput: %.1f MB/s\n",
                wf_init_ms, (double) wf_init_size / (1024 * 1024),
                ((double) wf_init_size / 1e6) / (wf_init_ms / 1000.0));

    // 2b. KV store write initial (create sparse file)
    std::vector<SessionKvSource> base_sources = make_sources_from_kv(base_conv.kv);
    std::vector<std::array<uint64_t, 5>> zero_from(num_layers, {0, 0, 0, 0, 0});
    uint64_t kv_written = 0;

    t0 = Clock::now();
    bool kv_init_ok = kv_store_write(kv_store_path, id, layout, base_sources, zero_from, 8192, true, kv_written, err);
    t1 = Clock::now();
    double kv_init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("KV Store Park (Initial):   %.2f ms | Bytes Written: %.2f MB | Write Throughput: %.1f MB/s\n",
                kv_init_ms, (double) kv_written / (1024 * 1024),
                ((double) kv_written / 1e6) / (kv_init_ms / 1000.0));

    // 3. Delta Spill on Subsequent Turn (Turn 2: Conversation grew by delta_bytes)
    std::printf("\n--- 3. Subsequent Spill on Turn Growth (Delta: +%.2f MB across %zu layers) ---\n",
                (double) total_delta_bytes / (1024 * 1024), num_layers);

    // 3a. Whole-file re-park (must re-write entire updated conversation from scratch)
    t0 = Clock::now();
    size_t wf_re_bytes = 0;
    bool wf_re_ok = session_file_write(whole_file_path, updated_conv, id, wf_re_bytes, err, write_opt);
    t1 = Clock::now();
    double wf_re_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    uint64_t wf_re_size = fs::file_size(whole_file_path);
    std::printf("Whole-File Re-Park:        Time: %7.2f ms | Written: %6.2f MB | Rate: %6.1f MB/s\n",
                wf_re_ms, (double) wf_re_size / (1024 * 1024),
                ((double) wf_re_size / 1e6) / (wf_re_ms / 1000.0));

    // 3b. Delta KV Store re-park (updates ONLY the changed bytes since base_conv in place)
    std::vector<SessionKvSource> updated_sources = make_sources_from_kv(updated_conv.kv);
    std::vector<std::array<uint64_t, 5>> from_sizes(num_layers);
    for (size_t j = 0; j < num_layers; ++j) {
        from_sizes[j] = {base_k, base_v, 0, 0, 0};
    }

    uint64_t delta_written = 0;
    t0 = Clock::now();
    bool kv_delta_ok = kv_store_write(kv_store_path, id, layout, updated_sources, from_sizes, 8704, false, delta_written, err);
    t1 = Clock::now();
    double kv_delta_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Delta KV Store Park:       Time: %7.2f ms | Written: %6.2f MB | Rate: %6.1f MB/s\n",
                kv_delta_ms, (double) delta_written / (1024 * 1024),
                ((double) delta_written / 1e6) / (kv_delta_ms / 1000.0));

    double bytes_saved_mb = ((double) wf_re_size - (double) delta_written) / (1024 * 1024);
    double pct_saved = (1.0 - (double) delta_written / (double) wf_re_size) * 100.0;
    double speedup = wf_re_ms / std::max(0.001, kv_delta_ms);

    std::printf("\n>>> DELTA SPILL SAVINGS: <<<\n");
    std::printf("    Bytes Written:  %.2f MB (Delta) vs %.2f MB (Whole)\n",
                (double) delta_written / (1024 * 1024), (double) wf_re_size / (1024 * 1024));
    std::printf("    I/O Saved:      %.2f MB reduction (%.1f%% fewer bytes written to SSD)\n",
                bytes_saved_mb, pct_saved);
    std::printf("    Write Latency:  %.2f ms (Delta) vs %.2f ms (Whole)\n",
                kv_delta_ms, wf_re_ms);
    std::printf("    Latency Speedup: %.2fx faster spill\n", speedup);

    // 4. Restore Benchmark: Streaming Restore vs Whole-File Restore
    std::printf("\n--- 4. Restore Comparison ---\n");
    // 4a. Whole file read back
    SavedConversation restored_conv;
    size_t wf_read_bytes = 0;
    SessionReadLimits limits;
    limits.skip_payload_hash = true; // as used in SSD tier

    t0 = Clock::now();
    bool wf_read_ok = session_file_read(whole_file_path, id, restored_conv, wf_read_bytes, err, limits);
    t1 = Clock::now();
    double wf_read_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Whole-File Read Back:      Time: %7.2f ms | Read: %6.2f MB | Rate: %6.1f MB/s\n",
                wf_read_ms, (double) wf_read_bytes / (1024 * 1024),
                ((double) wf_read_bytes / 1e6) / (wf_read_ms / 1000.0));

    // 4b. Direct Streaming restore (kv_store_stream)
    KvStoreInfo stream_info;
    size_t streamed_bytes = 0;
    t0 = Clock::now();
    bool stream_ok = kv_store_stream(
        kv_store_path, id, layout,
        [](const KvStoreInfo&, std::string&) { return true; },
        [&streamed_bytes](size_t, size_t, uint64_t, const uint8_t*, size_t n, std::string&) {
            streamed_bytes += n;
            return true;
        },
        stream_info, err);
    t1 = Clock::now();
    double stream_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Direct Streaming Restore:  Time: %7.2f ms | Streamed: %6.2f MB | Rate: %6.1f MB/s\n",
                stream_ms, (double) streamed_bytes / (1024 * 1024),
                ((double) streamed_bytes / 1e6) / (stream_ms / 1000.0));

    std::printf("\n=================================================================================\n");
    std::printf(" Benchmark finished successfully.\n");
    std::printf("=================================================================================\n");

    return 0;
}
