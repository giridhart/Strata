#include "strata/core/mock_cuda_graph.hpp"
#include <sycl/sycl.hpp>
#include "strata/core/native_head.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <climits>
#include <cstdio>
#include <cstring>
#include <exception>

namespace strata::core {

namespace {
inline sycl::queue& default_queue() {
    static sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    return q;
}
}  // namespace

NativeHead::~NativeHead() {
    if (scratch_) {
        try {
            sycl::free((void*)scratch_, default_queue());
        } catch (...) {}
    }
    if (weights_) {
        try {
            sycl::free((void*)weights_, default_queue());
        } catch (...) {}
    }
}

bool NativeHead::load(const std::vector<std::string> &shards, int64_t n_in,
                      int64_t n_out, std::string &err) {
    if (loaded()) { err = "native head is already loaded"; return false; }
    if (n_in <= 0 || n_out <= 0 || n_in > INT_MAX || n_out > INT_MAX || n_in % 256) {
        err = "native head requires positive int32 dimensions and whole 256-value rows";
        return false;
    }
    try {
        const strata::GgufModel model(shards);
        err = strata::check_architecture(model.meta());
        if (!err.empty()) return false;
        size_t at = 0;
        const strata::TensorInfo* tensor = model.find("output.weight", &at);
        const strata::GgufFile& gguf = model.shard(at);
        if (!tensor || !strata::kernels::native_mmvq_supported((int) tensor->type) || tensor->shape.size() != 2 ||
            tensor->shape[0] != (uint64_t) n_in || tensor->shape[1] != (uint64_t) n_out) {
            err = "native head: expected a natively supported output.weight with the canonical head dimensions";
            return false;
        }
        const uint64_t bytes = strata::kernels::native_mmvq_weight_bytes((int) tensor->type, (int) n_in, (int) n_out);
        const uint64_t payload = gguf.file_size() - gguf.data_start();
        if (tensor->offset > payload || bytes > payload - tensor->offset) {
            err = "native head: truncated output.weight payload";
            return false;
        }
        void* weights = nullptr;
        void* scratch = nullptr;
        try {
            weights = sycl::malloc_device(bytes, default_queue());
            scratch = sycl::malloc_device(strata::kernels::native_q8_1_bytes((int)n_in, 1), default_queue());
        } catch (...) {
            weights = nullptr;
            scratch = nullptr;
        }
        if (!weights || !scratch) {
            if (scratch) sycl::free(scratch, default_queue());
            if (weights) sycl::free(weights, default_queue());
            err = "native head: device memory allocation failed";
            return false;
        }
        try {
            default_queue().memcpy(weights, gguf.tensor_data(*tensor), bytes).wait();
        } catch (const sycl::exception& e) {
            if (scratch) sycl::free(scratch, default_queue());
            if (weights) sycl::free(weights, default_queue());
            err = std::string("native head upload: ") + e.what();
            return false;
        }
        weights_ = weights;
        scratch_ = scratch;
        bytes_ = bytes;
        n_in_ = (int) n_in;
        n_out_ = (int) n_out;
        type_ = (int) tensor->type;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native head: ") + error.what();
        return false;
    }
}

bool NativeHead::run(const float *mixed, float *logits, void *stream,
                     std::string &err) const {
    if (!loaded() || !mixed || !logits || !stream) {
        err = "native head requires loaded weights, device buffers and an explicit stream";
        return false;
    }
    try {
        if (type_ == 13) {
            strata::kernels::native_q5_k_f32(weights_, mixed, scratch_, logits, n_in_, n_out_, 1, stream);
        } else {
            strata::kernels::native_quantize_q8_1(mixed, scratch_, n_in_, 1, stream);
            strata::kernels::native_mmvq(type_, weights_, scratch_, logits, n_in_, n_out_, 1, stream);
        }
    } catch (const std::exception& error) {
        err = std::string("native head launch: ") + error.what();
        return false;
    }
    return true;
}

// ================================ plan v0.3 P6: THE NATIVE EMBEDDING ================================

namespace {
const NativeEmbed* g_embed = nullptr;
}
void set_native_embed(const NativeEmbed* e) { g_embed = e; }
const NativeEmbed* native_embed() { return g_embed; }

NativeEmbed::~NativeEmbed() {
    if (host_) {
        try {
            sycl::free(host_, default_queue());
        } catch (...) {}
    } else if (dev_) {
        try {
            sycl::free((void*)dev_, default_queue());
        } catch (...) {}
    }
}

bool NativeEmbed::load(const std::vector<std::string> &shards, int64_t n_embd,
                       int64_t n_vocab, std::string &err) {
    try {
        const strata::GgufModel model(shards);
        const strata::MetaValue* arch = model.meta().get("general.architecture");
        err = arch != nullptr && arch->s == "strata-embd" ? std::string() : strata::check_architecture(model.meta());
        if (!err.empty()) { err = "native embedding: " + err; return false; }
        size_t at = 0;
        const strata::TensorInfo* t = model.find("token_embd.weight", &at);
        const strata::GgufFile& gguf = model.shard(at);
        if (!t || t->shape.size() != 2 || t->shape[0] != (uint64_t) n_embd || t->shape[1] != (uint64_t) n_vocab ||
            !strata::kernels::embed_type_supported((int) t->type) || n_embd % 256) {
            err = "native embedding: token_embd.weight is absent, of another shape, or of a type without a GPU "
                  "dequantizer";
            return false;
        }
        row_ = strata::kernels::iq_row_bytes((int) t->type, n_embd);
        bytes_ = (uint64_t) row_ * (uint64_t) n_vocab;
        if (!model.in_bounds(*t, at) || strata::tensor_payload_bytes(*t) != bytes_) {
            err = "native embedding: token_embd.weight's payload is truncated or not " + std::to_string(bytes_) +
                  " B (" + gguf.path() + ")";
            bytes_ = 0;
            return false;
        }
        try {
            host_ = sycl::malloc_host(bytes_, default_queue());
        } catch (...) {
            host_ = nullptr;
        }
        if (host_ == nullptr) {
            void* d = nullptr;
            try {
                d = sycl::malloc_device(bytes_, default_queue());
                default_queue().memcpy(d, gguf.tensor_data(*t), bytes_).wait();
                dev_ = d;
            } catch (const sycl::exception& e) {
                if (d) sycl::free(d, default_queue());
                err = "native embedding: cannot allocate " + std::to_string(bytes_ >> 20) + " MiB in VRAM: " + e.what();
                return false;
            }
            std::fprintf(stderr, "strata: native embedding: kept in VRAM (%llu MiB)\n",
                         (unsigned long long) (bytes_ >> 20));
        } else {
            std::memcpy(host_, gguf.tensor_data(*t), bytes_);
            dev_ = host_;
        }
        type_ = (int) t->type;
        n_embd_ = n_embd;
        n_vocab_ = n_vocab;
        return true;
    } catch (const std::exception& e) {
        err = std::string("native embedding: ") + e.what();
        return false;
    }
}

void NativeEmbed::gather_dev(const int32_t* tokens, int64_t n_tok, float* out, void* stream) const {
    strata::kernels::iq_embed_rows(type_, dev_, row_, tokens, n_tok, n_embd_, out, stream);
}

void NativeEmbed::gather_one(int64_t token, float* out, void* stream) const {
    strata::kernels::iq_dequant_f32(type_, (const uint8_t*) dev_ + (size_t) token * row_, n_embd_, out, stream);
}

}  // namespace strata::core
