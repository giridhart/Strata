#include "strata/core/mock_cuda_graph.hpp"
#include <sycl/sycl.hpp>
#include "strata/core/remote_experts.hpp"
#include "strata/core/device.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cmath>

namespace strata::core {
namespace {
constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t FF = strata::kernels::cpu::FF;
constexpr int64_t CAP = strata::kernels::cpu::MAXT * 10;

// Small enough to copy as one pinned buffer per layer. The kernels read the
// individual arrays through pointers into the same device allocation.
struct RemoteMeta {
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t count;
};

inline sycl::queue make_device_queue(int ordinal) {
    auto devs = sycl::device::get_devices(sycl::info::device_type::all);
    if (ordinal >= 0 && ordinal < static_cast<int>(devs.size())) {
        return sycl::queue(devs[ordinal], sycl::property::queue::in_order());
    }
    return sycl::queue(sycl::default_selector_v, sycl::property::queue::in_order());
}

}  // namespace

RemoteExperts::~RemoteExperts() { close(); }

bool RemoteExperts::preflight(int device, double& free_gib, std::string& err) try {
    int count = device_count();
    if (device < 1 || device >= count) {
        err = "CUDA" + std::to_string(device) + " experts: device is not visible (have " + std::to_string(count) + ")";
        return false;
    }
    DeviceInfo d = device_info(device);
    free_gib = (double)d.free_bytes / 1073741824.0;
    return true;
} catch (const std::exception& exc) {
    err = std::string("preflight failed: ") + exc.what();
    return false;
}

void RemoteExperts::close() {
    if (device_ < 0) return;
    if (stream_) {
        try {
            stream_->wait();
        } catch (...) {
        }
        cache_.close();
        if (d_x_) sycl::free(d_x_, *stream_);
        if (d_out_) sycl::free(d_out_, *stream_);
        if (d_q8_) sycl::free(d_q8_, *stream_);
        if (d_scales_) sycl::free(d_scales_, *stream_);
        if (d_scratch_) sycl::free(d_scratch_, *stream_);
        if (d_meta_) sycl::free(d_meta_, *stream_);
        if (h_x_) sycl::free(h_x_, *stream_);
        if (h_out_) sycl::free(h_out_, *stream_);
        if (h_meta_) sycl::free(h_meta_, *stream_);
        delete stream_;
        stream_ = nullptr;
    }
    device_ = -1;
    h_x_ = h_out_ = d_x_ = d_out_ = nullptr;
    h_meta_ = d_meta_ = nullptr;
    d_q8_ = nullptr;
    d_scales_ = nullptr;
    d_scratch_ = nullptr;
    d_start_ = d_dst_ = d_tok_ = d_count_ = nullptr;
    d_ptr_ = nullptr;
    original_row_.clear();
    layers_present_.clear();
}

bool RemoteExperts::open(int device, int slots, int64_t layers, int64_t experts,
                         const std::vector<std::pair<int32_t, int32_t>>& ranked, const ExpertCache& primary,
                         ExpertSource& source, std::vector<uint8_t>& claimed, std::string& err) try {
    close();
    int count = device_count();
    if (device < 1 || device >= count || slots <= 0 || ranked.empty() || layers <= 0 || experts <= 0 ||
        claimed.size() != (size_t)layers * (size_t)experts) {
        err = "CUDA" + std::to_string(device) + " experts: need the device, ranked experts and positive slot count";
        return false;
    }
    device_ = device;
    n_expert_ = experts;
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<std::pair<int32_t, int32_t>> selected;
    selected.reserve((size_t)slots);
    std::vector<uint8_t> picked(claimed.size(), 0);
    for (const auto& pair : ranked) {
        if (pair.first < 0 || pair.first >= layers || pair.second < 0 || pair.second >= experts) continue;
        const size_t index = (size_t)pair.first * (size_t)experts + (size_t)pair.second;
        if (primary.slot_of(pair.first, pair.second) < 0 && !claimed[index] && !picked[index]) {
            selected.push_back(pair);
            picked[index] = 1;
            if ((int)selected.size() >= slots) break;
        }
    }
    if (selected.empty()) {
        err = "CUDA" + std::to_string(device) + " experts: no unclaimed experts remain";
        close();
        return false;
    }
    std::vector<int64_t> sizes;
    if (lay.native) {
        sizes.reserve(selected.size());
        for (const auto& pair : selected)
            sizes.push_back((int64_t)lay.blob_bytes(pair.first));
    }
    DeviceInfo d = device_info(device);
    size_t free_bytes = d.free_bytes;
    uint64_t needed = 0;
    for (const auto& pair : selected)
        needed += lay.native ? (lay.blob_bytes(pair.first) + 255) / 256 * 256 : lay.max_blob;
    // Leave room for staging and later driver allocations
    if (needed + (512ull << 20) > free_bytes) {
        err = "CUDA" + std::to_string(device) +
              " experts: slots leave less than 512 MiB free; reduce --expert-cache-device" + std::to_string(device);
        close();
        return false;
    }
    const bool cache_ok = lay.native
                              ? cache_.open_sized(sizes, layers, experts, err)
                              : cache_.open((int64_t)selected.size(), layers, experts, (int64_t)lay.max_blob, err);
    if (!cache_ok) {
        err = "CUDA" + std::to_string(device) + " experts: " + err;
        close();
        return false;
    }
    for (const auto& pair : selected) {
        const int32_t slot = cache_.admit(pair.first, pair.second);
        const uint8_t* blob = source.blob(pair.first, pair.second);
        if (slot < 0 || !blob || !cache_.fill_slot_blocking(slot, blob, err, (int64_t)lay.blob_bytes(pair.first))) {
            err = "CUDA" + std::to_string(device) + " experts: " + (err.empty() ? "cache fill failed" : err);
            close();
            return false;
        }
    }
    const auto& first = selected.front();
    if (!cache_.verify_slot(cache_.slot_of(first.first, first.second), source.blob(first.first, first.second), err,
                            (int64_t)lay.blob_bytes(first.first))) {
        err = "CUDA" + std::to_string(device) + " experts: " + err;
        close();
        return false;
    }

    const size_t scratch = std::max<size_t>((size_t)strata::kernels::moe_hit_grouped_scratch_bytes(CAP, H, FF),
                                            strata::kernels::native_expert_scratch_bytes(CAP, FF));
    stream_ = new sycl::queue(make_device_queue(device));
    auto& q = *stream_;
    h_x_ = sycl::malloc_host<float>((size_t)CAP * H, q);
    h_out_ = sycl::malloc_host<float>((size_t)CAP * H, q);
    h_meta_ = sycl::malloc_host(sizeof(RemoteMeta), q);
    d_x_ = sycl::malloc_device<float>((size_t)CAP * H, q);
    d_out_ = sycl::malloc_device<float>((size_t)CAP * H, q);
    d_q8_ = (uint8_t*)sycl::malloc_device((size_t)CAP * (H / 32) * 36, q);
    d_scales_ = sycl::malloc_device<float>((size_t)CAP * (H / 32), q);
    d_scratch_ = sycl::malloc_device(scratch, q);
    d_meta_ = sycl::malloc_device(sizeof(RemoteMeta), q);
    if (!h_x_ || !h_out_ || !h_meta_ || !d_x_ || !d_out_ || !d_q8_ || !d_scales_ || !d_scratch_ || !d_meta_) {
        err = "CUDA" + std::to_string(device) + " experts: allocation failed";
        close();
        return false;
    }
    const char* zc = std::getenv("STRATA_REMOTE_ZEROCOPY");
    zero_copy_ = !(zc && zc[0] == '0');
    z_x_ = (float*)h_x_;
    z_out_ = (float*)h_out_;

    auto* meta = (RemoteMeta*)d_meta_;
    d_ptr_ = meta->ptr;
    d_start_ = meta->start;
    d_dst_ = meta->dst;
    d_tok_ = meta->tok;
    d_count_ = &meta->count;
    owned_.resize(CAP);
    layers_present_.assign((size_t)layers, 0);
    group_of_.resize(CAP);
    group_id_.reserve(CAP);
    ptr_.reserve(CAP);
    start_.reserve(CAP + 1);
    dst_.reserve(CAP);
    tok_.reserve(CAP);
    original_row_.reserve(CAP);
    computed_ = 0;
    launched_layers_ = 0;
    returned_bytes_ = full_row_bytes_ = 0;
    for (const auto& pair : selected) {
        layers_present_[(size_t)pair.first] = 1;
        claimed[(size_t)pair.first * (size_t)experts + (size_t)pair.second] = 1;
    }
    return true;
} catch (const std::exception& exc) {
    err = std::string("open failed: ") + exc.what();
    return false;
}

bool RemoteExperts::begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                          const int32_t* kind, const int32_t* primary_res, std::string& err) try {
    // cumulative host time in here (staging and launches), reported per request by the driver
    struct Timer {
        double& acc;
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timer() { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
    } timer{ms_begin_};
    const int64_t n = n_tok * k;
    if (n <= 0 || n > CAP || n_tok > strata::kernels::cpu::MAXT || k != 10 || layer < 0 ||
        (size_t)layer >= layers_present_.size() || device_ < 0) {
        err = "CUDA" + std::to_string(device_) + " experts: invalid layer, routing width or window size";
        return false;
    }
    std::fill(owned_.begin(), owned_.begin() + n, 0);
    group_id_.clear();
    ptr_.clear();
    start_.clear();
    dst_.clear();
    tok_.clear();
    original_row_.clear();
    if (!layers_present_[(size_t)layer]) return true;
    std::fill(group_of_.begin(), group_of_.begin() + n, -1);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if ((kind && kind[i] != -1) || e < 0 || e >= n_expert_ ||
            (primary_res && primary_res[(size_t)layer * (size_t)n_expert_ + (size_t)e] >= 0))
            continue;
        const int32_t slot = cache_.slot_of(layer, e);
        if (slot < 0) continue;
        owned_[(size_t)i] = 1;
        int32_t group = -1;
        for (size_t g = 0; g < group_id_.size(); ++g)
            if (group_id_[g] == e) {
                group = (int32_t)g;
                break;
            }
        if (group < 0) {
            group = (int32_t)group_id_.size();
            group_id_.push_back(e);
            ptr_.push_back((unsigned long long)cache_.device_slot(slot));
        }
        group_of_[(size_t)i] = group;
        ++computed_;
    }
    if (group_id_.empty()) return true;
    for (size_t g = 0; g < group_id_.size(); ++g) {
        start_.push_back((int32_t)dst_.size());
        for (int64_t i = 0; i < n; ++i)
            if (group_of_[(size_t)i] == (int32_t)g) {
                // The grouped kernels read the activation from `tok`, so their
                // output row can instead be packed densely for the USB4 return.
                original_row_.push_back((int32_t)i);
                dst_.push_back((int32_t)dst_.size());
                tok_.push_back((int32_t)(i / k));
            }
    }
    start_.push_back((int32_t)dst_.size());
    // Private pinned buffers survive until this GPU has consumed them. The CPU
    // pool can write other output rows without a cross-device race.
    std::memcpy(h_x_, x, (size_t)n_tok * H * sizeof(float));
    auto* meta = (RemoteMeta*)h_meta_;
    std::memcpy(meta->ptr, ptr_.data(), ptr_.size() * sizeof(ptr_[0]));
    std::memcpy(meta->start, start_.data(), start_.size() * sizeof(start_[0]));
    std::memcpy(meta->dst, dst_.data(), dst_.size() * sizeof(dst_[0]));
    std::memcpy(meta->tok, tok_.data(), tok_.size() * sizeof(tok_[0]));
    meta->count = (int32_t)group_id_.size();
    groups_ = (int32_t)group_id_.size();
    const auto s = stream_;
    if (!zero_copy_) {
        s->memcpy(d_x_, h_x_, (size_t)n_tok * H * sizeof(float));
    }
    s->memcpy(d_meta_, h_meta_, sizeof(RemoteMeta));
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (lay.native) {
        strata::kernels::quantize_q8_1_rows(zero_copy_ ? z_x_ : d_x_, n_tok, H, d_q8_, s);
        const auto& fmt = lay.fmt[(size_t)layer];
        auto L = strata::kernels::native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff);
        strata::kernels::native_expert_grouped(L, d_ptr_, d_start_, d_count_, d_dst_, d_tok_, groups_,
                                               (int64_t)dst_.size(), d_q8_, d_scratch_, zero_copy_ ? z_out_ : d_out_,
                                               s);
    } else {
        strata::kernels::quantize_q8_0_scaled(zero_copy_ ? z_x_ : d_x_, d_q8_, d_scales_, n_tok * H, s);
        strata::kernels::moe_grouped_s2(d_ptr_, d_start_, d_count_, d_dst_, d_tok_, groups_, (int64_t)dst_.size(),
                                        d_q8_, d_scales_, d_scratch_, zero_copy_ ? z_out_ : d_out_, s);
    }
    const uint64_t compact_bytes = (uint64_t)dst_.size() * H * sizeof(float);
    if (!zero_copy_) {
        s->memcpy(h_out_, d_out_, (size_t)compact_bytes);
    }
    ++launched_layers_;
    returned_bytes_ += compact_bytes;
    full_row_bytes_ += (uint64_t)n * H * sizeof(float);
    return true;
} catch (const std::exception& exc) {
    err = std::string("begin failed: ") + exc.what();
    return false;
}

bool RemoteExperts::finish(float* out, std::string& err) try {
    if (group_id_.empty()) return true;
    const auto w0 = std::chrono::steady_clock::now();
    stream_->wait_and_throw();
    ms_wait_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    for (size_t i = 0; i < original_row_.size(); ++i)
        std::memcpy(out + (size_t)original_row_[i] * H, h_out_ + i * H, (size_t)H * sizeof(float));
    return true;
} catch (const std::exception& exc) {
    err = std::string("finish failed: ") + exc.what();
    return false;
}

}  // namespace strata::core
