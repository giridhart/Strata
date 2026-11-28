#pragma once

#include <sycl/sycl.hpp>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::sycl_tuning {

enum class CachePolicy : uint8_t {
    Default = 0,
    StreamingL3 = 1,
    UncachedL1L3 = 2,
    CachedL1L3 = 3
};

inline const char* to_string(CachePolicy p) {
    switch (p) {
        case CachePolicy::StreamingL3:  return "streaming";
        case CachePolicy::UncachedL1L3: return "uncached";
        case CachePolicy::CachedL1L3:   return "cached";
        case CachePolicy::Default:
        default:                        return "default";
    }
}

inline CachePolicy parse_cache_policy(std::string_view s) {
    if (s == "streaming") return CachePolicy::StreamingL3;
    if (s == "uncached")  return CachePolicy::UncachedL1L3;
    if (s == "cached")    return CachePolicy::CachedL1L3;
    return CachePolicy::Default;
}

enum class KernelType : uint8_t {
    UNKNOWN = 0,
    BF16_GEMV = 1,
    BF16_GEMV_PAIR = 2,
    S_GEMV = 3,
    S_GEMV_Q8K = 4,
    S_GEMV_Q8_0 = 5,
    S2_GEMV = 6,
    S2_GEMV_FAST = 7,
    S2_EXPERT = 8,
    GR_NORM = 9,
    FLASH_ATTN = 10,
    ROUTER = 11,
    GDN = 12,
    MOE_SCATTER_ADD = 13,
    PLE_DEQUANT = 14,
    QSA_SELECT_LONG = 15
};

inline const char* to_string(KernelType k) {
    switch (k) {
        case KernelType::BF16_GEMV:      return "bf16_gemv";
        case KernelType::BF16_GEMV_PAIR: return "bf16_gemv_pair";
        case KernelType::S_GEMV:         return "s_gemv";
        case KernelType::S_GEMV_Q8K:     return "s_gemv_q8k";
        case KernelType::S_GEMV_Q8_0:    return "s_gemv_q8_0";
        case KernelType::S2_GEMV:        return "s2_gemv";
        case KernelType::S2_GEMV_FAST:   return "s2_gemv_fast";
        case KernelType::S2_EXPERT:      return "s2_expert";
        case KernelType::GR_NORM:        return "gr_norm";
        case KernelType::FLASH_ATTN:     return "flash_attn";
        case KernelType::ROUTER:         return "native_router";
        case KernelType::GDN:            return "native_gdn";
        case KernelType::MOE_SCATTER_ADD: return "moe_scatter_add";
        case KernelType::PLE_DEQUANT:     return "ple_dequant";
        case KernelType::QSA_SELECT_LONG: return "qsa_select_long";
        case KernelType::UNKNOWN:
        default:                         return "unknown";
    }
}

inline KernelType parse_kernel_type(std::string_view s) {
    if (s == "bf16_gemv")      return KernelType::BF16_GEMV;
    if (s == "bf16_gemv_pair") return KernelType::BF16_GEMV_PAIR;
    if (s == "s_gemv")         return KernelType::S_GEMV;
    if (s == "s_gemv_q8k")     return KernelType::S_GEMV_Q8K;
    if (s == "s_gemv_q8_0")    return KernelType::S_GEMV_Q8_0;
    if (s == "s2_gemv")        return KernelType::S2_GEMV;
    if (s == "s2_gemv_fast")   return KernelType::S2_GEMV_FAST;
    if (s == "s2_expert")      return KernelType::S2_EXPERT;
    if (s == "gr_norm")        return KernelType::GR_NORM;
    if (s == "flash_attn")     return KernelType::FLASH_ATTN;
    if (s == "native_router" || s == "router") return KernelType::ROUTER;
    if (s == "native_gdn" || s == "gdn") return KernelType::GDN;
    if (s == "moe_scatter_add") return KernelType::MOE_SCATTER_ADD;
    if (s == "ple_dequant")     return KernelType::PLE_DEQUANT;
    if (s == "qsa_select_long") return KernelType::QSA_SELECT_LONG;
    return KernelType::UNKNOWN;
}

struct MemHints {
    CachePolicy weight = CachePolicy::StreamingL3;     // Weights: large, single-use per token -> streaming on L3
    CachePolicy activation = CachePolicy::CachedL1L3; // Activations (x): reused across output rows -> cached in L1/L3
    CachePolicy output = CachePolicy::Default;         // Output: written once -> default write-back
};

struct LaunchParams {
    int lws = 256;
    int tpr = 0;          // 0 = single thread/subgroup per row; >0 = split reduction
    int slm_pad = 0;      // Shared local memory padding elements for bank conflict avoidance
    CachePolicy cache = CachePolicy::StreamingL3;      // Primary weight cache policy
    MemHints hints = {};

    LaunchParams() = default;
    LaunchParams(int l, int t, int p, CachePolicy c = CachePolicy::StreamingL3)
        : lws(l), tpr(t), slm_pad(p), cache(c) {
        hints.weight = c;
    }
    LaunchParams(int l, int t, int p, CachePolicy c, MemHints h)
        : lws(l), tpr(t), slm_pad(p), cache(c), hints(h) {}

    void set_cache_policies(CachePolicy w, CachePolicy a = CachePolicy::CachedL1L3, CachePolicy o = CachePolicy::Default) {
        cache = w;
        hints.weight = w;
        hints.activation = a;
        hints.output = o;
    }
};

/// 64-bit collision-free composite key for O(1) kernel dispatch lookups
inline uint64_t make_kernel_key(KernelType k, int64_t n_in, int64_t n_out,
                                int code_bits = 0, int group_elems = 0, int act_kind = 0) {
    const uint64_t k_id  = static_cast<uint64_t>(k) & 0xFFULL;
    const uint64_t in_id = static_cast<uint64_t>(n_in) & 0xFFFFFULL;
    const uint64_t out_id= static_cast<uint64_t>(n_out) & 0xFFFFFULL;
    const uint64_t cb    = static_cast<uint64_t>(code_bits) & 0xFULL;
    const uint64_t ge    = static_cast<uint64_t>(group_elems) & 0xFFULL;
    const uint64_t ak    = static_cast<uint64_t>(act_kind) & 0xFULL;

    return (k_id << 56) | (in_id << 36) | (out_id << 16) | (cb << 12) | (ge << 4) | ak;
}

namespace detail {

inline std::string_view trim_leading(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r')) {
        s.remove_prefix(1);
    }
    return s;
}

inline bool extract_token(std::string_view& s, std::string& token) {
    s = trim_leading(s);
    if (s.empty()) return false;
    if (s.front() == '"') {
        s.remove_prefix(1);
        const auto end_q = s.find('"');
        if (end_q == std::string_view::npos) return false;
        token = std::string(s.substr(0, end_q));
        s.remove_prefix(end_q + 1);
        return true;
    }
    const auto end_ws = s.find_first_of(" \t\r\n");
    if (end_ws == std::string_view::npos) {
        token = std::string(s);
        s = {};
    } else {
        token = std::string(s.substr(0, end_ws));
        s.remove_prefix(end_ws);
    }
    return true;
}

}  // namespace detail

struct CapturedShape {
    KernelType type = KernelType::UNKNOWN;
    int64_t n_in = 0;
    int64_t n_out = 0;
    int code_bits = 0;
    int group_elems = 0;
    int act_kind = 0;
};

class ShapeCollector {
public:
    static ShapeCollector& instance() {
        static ShapeCollector sc;
        return sc;
    }

    void record(KernelType k, int64_t n_in, int64_t n_out, int code_bits = 0, int group_elems = 0, int act_kind = 0) {
        if (!active_) return;
        const uint64_t key = make_kernel_key(k, n_in, n_out, code_bits, group_elems, act_kind);
        bool is_new = false;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto [it, inserted] = shapes_.emplace(key, CapturedShape{k, n_in, n_out, code_bits, group_elems, act_kind});
            is_new = inserted;
        }
        if (is_new) {
            // Immediate atomic persistence so Ctrl+C or kill never loses captured shapes
            std::string err;
            dump_workload(get_default_out_path(), err);
        }
    }

    void enable(bool e = true) { active_ = e; }
    bool is_active() const { return active_; }
    size_t count() const { std::lock_guard<std::mutex> lock(mu_); return shapes_.size(); }

    static std::string get_default_out_path() {
        if (const char* out_env = std::getenv("STRATA_SYCL_WORKLOAD_OUT")) {
            if (std::strlen(out_env) > 0) return out_env;
        }
        return "data/tuning/sycl/workload_trace.txt";
    }

    bool dump_workload(const std::string& path, std::string& err) const {
        std::unordered_map<uint64_t, CapturedShape> merged;
        {
            std::lock_guard<std::mutex> lock(mu_);
            merged = shapes_;
        }

        // If file already exists, merge previously captured shapes so successive runs accumulate
        if (std::filesystem::exists(path)) {
            std::vector<CapturedShape> existing;
            std::string load_err;
            if (load_workload(path, existing, load_err)) {
                for (const auto& s : existing) {
                    const uint64_t k = make_kernel_key(s.type, s.n_in, s.n_out, s.code_bits, s.group_elems, s.act_kind);
                    merged.emplace(k, s);
                }
            }
        }

        std::string parent_dir = "data/tuning/sycl";
        if (const auto slash = path.rfind('/'); slash != std::string::npos) {
            parent_dir = path.substr(0, slash);
        }
        std::error_code ec;
        std::filesystem::create_directories(parent_dir, ec);

        std::ofstream out(path);
        if (!out.is_open()) {
            err = "cannot open workload file for writing: " + path;
            return false;
        }
        out << "# Captured kernel workload for Strata SYCL Autotuning\n";
        out << "STRATA_SYCL_WORKLOAD_V1\n";
        out << "# kernel_name n_in n_out code_bits group_elems act_kind\n";
        for (const auto& [key, s] : merged) {
            (void)key;
            out << to_string(s.type) << " " << s.n_in << " " << s.n_out << " "
                << s.code_bits << " " << s.group_elems << " " << s.act_kind << "\n";
        }
        return true;
    }

    static bool load_workload(const std::string& path, std::vector<CapturedShape>& out_shapes, std::string& err) {
        std::ifstream in(path);
        if (!in.is_open()) {
            err = "cannot open workload file for reading: " + path;
            return false;
        }
        out_shapes.clear();
        std::string line;
        size_t line_no = 0;
        bool saw_header = false;
        while (std::getline(in, line)) {
            ++line_no;
            std::string_view sv(line);
            if (const auto hash = sv.find('#'); hash != std::string_view::npos) {
                sv = sv.substr(0, hash);
            }
            sv = detail::trim_leading(sv);
            if (sv.empty()) continue;

            if (!saw_header) {
                std::string magic;
                if (!detail::extract_token(sv, magic) || magic != "STRATA_SYCL_WORKLOAD_V1") {
                    err = "invalid workload header at line " + std::to_string(line_no);
                    return false;
                }
                saw_header = true;
                continue;
            }

            std::string k_str, in_str, out_str, cb_str, ge_str, ak_str;
            if (!detail::extract_token(sv, k_str) ||
                !detail::extract_token(sv, in_str) ||
                !detail::extract_token(sv, out_str) ||
                !detail::extract_token(sv, cb_str) ||
                !detail::extract_token(sv, ge_str) ||
                !detail::extract_token(sv, ak_str)) {
                err = "invalid workload row format at line " + std::to_string(line_no);
                return false;
            }
            CapturedShape s;
            s.type = parse_kernel_type(k_str);
            s.n_in = std::strtoll(in_str.c_str(), nullptr, 10);
            s.n_out = std::strtoll(out_str.c_str(), nullptr, 10);
            s.code_bits = std::atoi(cb_str.c_str());
            s.group_elems = std::atoi(ge_str.c_str());
            s.act_kind = std::atoi(ak_str.c_str());
            out_shapes.push_back(s);
        }
        return true;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        shapes_.clear();
    }

private:
    ShapeCollector() {
        if (const char* env = std::getenv("STRATA_SYCL_RECORD_SHAPES")) {
            active_ = (std::atoi(env) != 0);
        }
        if (const char* env2 = std::getenv("STRATA_SYCL_TRACE")) {
            if (std::atoi(env2) != 0) active_ = true;
        }
        if (active_) {
            const std::string out_path = get_default_out_path();
            if (std::filesystem::exists(out_path)) {
                std::vector<CapturedShape> existing;
                std::string err;
                if (load_workload(out_path, existing, err)) {
                    std::lock_guard<std::mutex> lock(mu_);
                    for (const auto& s : existing) {
                        const uint64_t k = make_kernel_key(s.type, s.n_in, s.n_out, s.code_bits, s.group_elems, s.act_kind);
                        shapes_[k] = s;
                    }
                }
            }
            std::signal(SIGINT, handle_signal);
            std::signal(SIGTERM, handle_signal);
            std::atexit(handle_exit);
        }
    }

    ~ShapeCollector() {
        if (active_ && count() > 0) {
            std::string out_path = get_default_out_path();
            std::string err;
            if (dump_workload(out_path, err)) {
                std::fprintf(stderr, "\n[STRATA_SYCL_TRACE] Captured %zu kernel shapes flushed to: %s\n",
                             count(), out_path.c_str());
            }
        }
    }

    static void handle_signal(int sig) {
        if (ShapeCollector::instance().is_active() && ShapeCollector::instance().count() > 0) {
            std::string err;
            ShapeCollector::instance().dump_workload(get_default_out_path(), err);
            std::fprintf(stderr, "\n[STRATA_SYCL_TRACE] Signal %d (Ctrl+C/term) intercepted: %zu shapes safely saved to %s\n",
                         sig, ShapeCollector::instance().count(), get_default_out_path().c_str());
        }
        std::signal(sig, SIG_DFL);
        std::raise(sig);
    }

    static void handle_exit() {
        if (ShapeCollector::instance().is_active() && ShapeCollector::instance().count() > 0) {
            std::string err;
            ShapeCollector::instance().dump_workload(get_default_out_path(), err);
        }
    }

    std::atomic<bool> active_{false};
    mutable std::mutex mu_;
    std::unordered_map<uint64_t, CapturedShape> shapes_;
};

#define STRATA_SYCL_RECORD_SHAPE(k, n_in, n_out, code_bits, group_elems, act_kind) \
    do { \
        if (strata::sycl_tuning::ShapeCollector::instance().is_active()) { \
            strata::sycl_tuning::ShapeCollector::instance().record( \
                k, n_in, n_out, code_bits, group_elems, act_kind); \
        } \
    } while (0)

struct DeviceFingerprint {
    std::string device_name;
    std::string driver_version;
    uint32_t compute_units = 0;
    int native_subgroup_size = 16;

    static DeviceFingerprint from_device(const sycl::device& dev) {
        DeviceFingerprint fp;
        fp.device_name = dev.get_info<sycl::info::device::name>();
        fp.driver_version = dev.get_info<sycl::info::device::driver_version>();
        fp.compute_units = dev.get_info<sycl::info::device::max_compute_units>();

        const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
        if (!sg_sizes.empty()) {
            // Pick native Xe2 subgroup size (16) if supported, else first supported
            fp.native_subgroup_size = (std::find(sg_sizes.begin(), sg_sizes.end(), 16) != sg_sizes.end()) ? 16 : (int)sg_sizes[0];
        } else {
            fp.native_subgroup_size = 16;
        }
        return fp;
    }

    bool matches(const sycl::device& dev) const {
        const auto cur = from_device(dev);
        if (device_name != cur.device_name) return false;
        if (compute_units != cur.compute_units) return false;
        // Subgroup size must be supported by device
        const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
        if (std::find(sg_sizes.begin(), sg_sizes.end(), (size_t)native_subgroup_size) == sg_sizes.end()) {
            return false;
        }
        return true;
    }
};

class SyclTuningRegistry {
public:
    static SyclTuningRegistry& instance() {
        static SyclTuningRegistry reg;
        return reg;
    }

    bool load_from_string(const std::string& content, const sycl::device& dev, std::string& err) {
        clear();
        std::istringstream stream(content);
        std::string line;
        size_t line_no = 0;
        bool saw_header = false;

        while (std::getline(stream, line)) {
            ++line_no;
            std::string_view sv(line);
            // Strip comments starting with '#'
            if (const auto hash = sv.find('#'); hash != std::string_view::npos) {
                sv = sv.substr(0, hash);
            }
            sv = detail::trim_leading(sv);
            if (sv.empty()) continue;

            if (!saw_header) {
                std::string magic, dev_name, driver, cu_str, sg_str;
                if (!detail::extract_token(sv, magic) || magic != "STRATA_SYCL_TUNING_V1" ||
                    !detail::extract_token(sv, dev_name) ||
                    !detail::extract_token(sv, driver) ||
                    !detail::extract_token(sv, cu_str) ||
                    !detail::extract_token(sv, sg_str)) {
                    err = "invalid tuning header at line " + std::to_string(line_no);
                    return false;
                }
                fingerprint_.device_name = dev_name;
                fingerprint_.driver_version = driver;
                fingerprint_.compute_units = static_cast<uint32_t>(std::strtoul(cu_str.c_str(), nullptr, 10));
                fingerprint_.native_subgroup_size = std::atoi(sg_str.c_str());

                if (!fingerprint_.matches(dev)) {
                    const auto cur = DeviceFingerprint::from_device(dev);
                    err = "device mismatch: tuning file is for [" + dev_name + " (CU: " + cu_str +
                          ")], but current device is [" + cur.device_name + " (CU: " +
                          std::to_string(cur.compute_units) + ")]";
                    return false;
                }
                saw_header = true;
                continue;
            }

            // Parse entry:
            // <kernel_name> <n_in> <n_out> <code_bits> <group_elems> <act_kind> <lws> <tpr> <slm_pad> <cache_hint>
            std::string k_str, in_str, out_str, cb_str, ge_str, ak_str, lws_str, tpr_str, pad_str, cache_str;
            if (!detail::extract_token(sv, k_str) ||
                !detail::extract_token(sv, in_str) ||
                !detail::extract_token(sv, out_str) ||
                !detail::extract_token(sv, cb_str) ||
                !detail::extract_token(sv, ge_str) ||
                !detail::extract_token(sv, ak_str) ||
                !detail::extract_token(sv, lws_str) ||
                !detail::extract_token(sv, tpr_str) ||
                !detail::extract_token(sv, pad_str) ||
                !detail::extract_token(sv, cache_str)) {
                err = "invalid row format at line " + std::to_string(line_no);
                return false;
            }

            const auto k_type = parse_kernel_type(k_str);
            if (k_type == KernelType::UNKNOWN) {
                err = "unknown kernel type '" + k_str + "' at line " + std::to_string(line_no);
                return false;
            }

            const int64_t n_in = std::strtoll(in_str.c_str(), nullptr, 10);
            const int64_t n_out = std::strtoll(out_str.c_str(), nullptr, 10);
            const int code_bits = std::atoi(cb_str.c_str());
            const int group_elems = std::atoi(ge_str.c_str());
            const int act_kind = std::atoi(ak_str.c_str());

            LaunchParams p;
            p.lws = std::atoi(lws_str.c_str());
            p.tpr = std::atoi(tpr_str.c_str());
            p.slm_pad = std::atoi(pad_str.c_str());

            const CachePolicy w_cache = parse_cache_policy(cache_str);
            CachePolicy a_cache = CachePolicy::CachedL1L3;
            CachePolicy o_cache = CachePolicy::Default;

            std::string act_cache_str, out_cache_str;
            if (detail::extract_token(sv, act_cache_str)) {
                a_cache = parse_cache_policy(act_cache_str);
            }
            if (detail::extract_token(sv, out_cache_str)) {
                o_cache = parse_cache_policy(out_cache_str);
            }
            p.set_cache_policies(w_cache, a_cache, o_cache);

            if (p.lws <= 0 || (p.lws % fingerprint_.native_subgroup_size != 0)) {
                err = "invalid lws " + lws_str + " (must be multiple of " +
                      std::to_string(fingerprint_.native_subgroup_size) + ") at line " + std::to_string(line_no);
                return false;
            }

            const uint64_t key = make_kernel_key(k_type, n_in, n_out, code_bits, group_elems, act_kind);
            table_[key] = p;
        }

        if (!saw_header) {
            err = "tuning file has no header";
            return false;
        }

        loaded_ = true;
        return true;
    }

    bool load_from_file(const std::string& path, const sycl::device& dev, std::string& err) {
        std::ifstream file(path);
        if (!file.is_open()) {
            err = "could not open tuning file: " + path;
            return false;
        }
        std::stringstream ss;
        ss << file.rdbuf();
        return load_from_string(ss.str(), dev, err);
    }

    void try_auto_load(const sycl::device& dev) {
        if (loaded_) return;
        if (const char* disable = std::getenv("STRATA_SYCL_NO_TUNING")) {
            if (std::atoi(disable) != 0) return;
        }
        const auto fp = DeviceFingerprint::from_device(dev);
        if (const char* env_path = std::getenv("STRATA_SYCL_TUNING_FILE")) {
            std::string err;
            if (load_from_file(env_path, dev, err)) {
                std::fprintf(stderr, "strata: SYCL tuning table loaded for [%s] (%zu entries from %s)\n",
                             fp.device_name.c_str(), entry_count(), env_path);
                return;
            } else {
                std::fprintf(stderr, "strata: warning: STRATA_SYCL_TUNING_FILE=%s failed to load: %s\n",
                             env_path, err.c_str());
            }
        }
        std::string safe_dev = fp.device_name;
        for (char& c : safe_dev) {
            if (!std::isalnum(c)) c = '_';
        }
        const std::string search_paths[] = {
            "data/tuning/sycl/tuning_" + safe_dev + ".txt",
            "tuning/sycl/tuning_" + safe_dev + ".txt",
            "tuning_" + safe_dev + ".txt"
        };
        for (const auto& p : search_paths) {
            std::string err;
            if (load_from_file(p, dev, err)) {
                std::fprintf(stderr, "strata: SYCL tuning table loaded for [%s] (%zu entries from %s)\n",
                             fp.device_name.c_str(), entry_count(), p.c_str());
                return;
            }
        }
    }

    LaunchParams get_or_fallback(uint64_t key, LaunchParams fallback) const {
        if (!loaded_) return fallback;
        const auto it = table_.find(key);
        if (it != table_.end()) {
            return it->second;
        }
        return fallback;
    }

    void clear() {
        table_.clear();
        loaded_ = false;
        fingerprint_ = {};
    }

    bool is_loaded() const { return loaded_; }
    SyclTuningRegistry() = default;
    size_t entry_count() const { return table_.size(); }
    const DeviceFingerprint& fingerprint() const { return fingerprint_; }

private:
    std::unordered_map<uint64_t, LaunchParams> table_;
    DeviceFingerprint fingerprint_;
    bool loaded_ = false;
};

}  // namespace strata::sycl_tuning
