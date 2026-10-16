"""tools/test_calibrate_from_report.py - Unit tests for calibrate_from_report.py

Tests parsing of benchmark reports for Intel Arc B580 desktop and Lunar Lake iGPU.
Runs offline without a GPU:
    python3 -m unittest tools.test_calibrate_from_report
"""
from __future__ import annotations

import unittest
from pathlib import Path

from tools.calibrate_from_report import ReportCalibrator, format_report_summary

MOCK_B580_REPORT = """# Intel Xe2 / Arc B580 Performance Benchmark Tuning Report
> **Execution Timestamp**: 2026-10-15_18-00-00  
> **Target Hardware**: `Intel(R) Arc(TM) B580 Graphics [0x56a0]`  
> **Microarchitecture**: Intel Xe2-HPG (Battlemage Discrete GPU)  
> **Compute Units**: 160 multiprocessors  
> **VRAM Pool**: 11.85 GiB / 12.00 GiB  

## 1. Executive Performance Summary
| Benchmark Suite | Key Evaluated Stage | Primary Measured Metric | Hardware Status |
|---|---|---|---|
| **Quantized KV Cache Decode Attention** | Context sweep | Decode Attention Bandwidth: **385.2 GB/s** | ✅ PASS |
| **Fused GDN Linear Attention Recurrence** | Recurrence | Fused: **12.4 µs** vs Unfused: **54.2 µs** (**4.37x faster**) | ✅ PASS |

## 2. Detailed Performance Tables by Kernel
### Quantized KV Cache Decode Attention (`perf_kv_decode`)
| Benchmark Name | Hardware Latency | CPU Latency | Bandwidth / Throughput | Key Counters |
|---|---|---|---|---|
| `BM_Xe2_KV_Decode_INT8_SplitK/16384/4/manual_time` | 820.0 µs | 900.0 µs | 185.2 GB/s | `Context_T`=16384, `Splits`=4 |
| `BM_Xe2_KV_Decode_INT8_SplitK/16384/8/manual_time` | 550.0 µs | 620.0 µs | 260.4 GB/s | `Context_T`=16384, `Splits`=8 |
| `BM_Xe2_KV_Decode_INT8_SplitK/16384/16/manual_time` | 410.0 µs | 480.0 µs | 335.8 GB/s | `Context_T`=16384, `Splits`=16 |
| `BM_Xe2_KV_Decode_INT8_SplitK/16384/32/manual_time` | 340.0 µs | 400.0 µs | 385.2 GB/s | `Context_T`=16384, `Splits`=32 |
| `BM_Xe2_KV_Decode_INT8_SplitK/16384/64/manual_time` | 380.0 µs | 440.0 µs | 345.1 GB/s | `Context_T`=16384, `Splits`=64 |

### Fused GDN Linear Attention Recurrence (`perf_fused_gdn`)
| Benchmark Name | Hardware Latency | CPU Latency | Bandwidth / Throughput | Key Counters |
|---|---|---|---|---|
| `BM_Xe2_GDN_Fused_SLM/16/16/manual_time` | 18.2 µs | 120.0 µs | 35.2 GB/s | `Pad_Elements`=0 |
| `BM_Xe2_GDN_Fused_SLM_Padded<4>/16/16/manual_time` | 14.5 µs | 115.0 µs | 48.0 GB/s | `Pad_Elements`=4 |
| `BM_Xe2_GDN_Fused_SLM_Padded<8>/16/16/manual_time` | 12.4 µs | 110.0 µs | 58.1 GB/s | `Pad_Elements`=8 |
| `BM_Xe2_GDN_Fused_SLM_Padded<16>/16/16/manual_time` | 13.8 µs | 112.0 µs | 51.5 GB/s | `Pad_Elements`=16 |

### MoE Expert Weighted Accumulation & Grouped GEMV (`perf_moe_accumulate`)
| Benchmark Name | Hardware Latency | CPU Latency | Bandwidth / Throughput | Key Counters |
|---|---|---|---|---|
| `BM_Xe2_MoE_Separate_Gate_Up_SwiGLU/1/manual_time` | 42.1 µs | 150.0 µs | - | - |
| `BM_Xe2_MoE_DualColumn_SwiGLU/1/manual_time` | 24.3 µs | 135.0 µs | - | - |

### Native Prefill Tiled GEMM Scaling (`perf_prefill_gemm`)
| Benchmark Name | Hardware Latency | CPU Latency | Bandwidth / Throughput | Key Counters |
|---|---|---|---|---|
| `BM_Xe2_Prefill_GEMM/512/2560/2560/manual_time` | 1.12 ms | 1200.0 µs | 28.5 TFLOPS | `Chunk_M`=512 |
| `BM_Xe2_Prefill_GEMM/1024/2560/2560/manual_time` | 1.95 ms | 2100.0 µs | 34.2 TFLOPS | `Chunk_M`=1024 |
| `BM_Xe2_Prefill_GEMM/2048/2560/2560/manual_time` | 3.50 ms | 3700.0 µs | 38.6 TFLOPS | `Chunk_M`=2048 |
"""


class CalibrateFromReportTest(unittest.TestCase):
    def test_b580_report_calibration(self):
        calibrator = ReportCalibrator(MOCK_B580_REPORT)
        cal = calibrator.calibrate()

        self.assertTrue(cal["device"]["is_bmg"])
        self.assertEqual(cal["device"]["compute_units"], 160)

        # Assert correct optimal parameters extracted
        params = cal["parameters"]
        self.assertEqual(params["STRATA_XE2_SPLIT_K"], 32, "B580 should pick 32 splits for peak bandwidth")
        self.assertEqual(params["STRATA_XE2_SLM_PAD"], 8, "B580 should pick 8-element padding for lowest latency")
        self.assertEqual(params["STRATA_XE2_DUAL_COLUMN"], 1, "B580 should enable dual-column when faster")
        self.assertEqual(params["STRATA_XE2_CACHE_HINT"], 1, "B580 should enable cache hint")
        self.assertEqual(params["PREFILL_CHUNK"], 2048, "B580 should pick 2048 for peak TFLOPS")

        # Format output
        summary = format_report_summary(cal)
        self.assertIn("export STRATA_XE2_SPLIT_K=32", summary)
        self.assertIn("export STRATA_XE2_SLM_PAD=8", summary)
        self.assertIn("export STRATA_XE2_DUAL_COLUMN=1", summary)
        self.assertIn("export STRATA_XE2_CACHE_HINT=1", summary)

    def test_lunar_lake_report_calibration(self):
        real_lnl_path = Path(__file__).resolve().parent.parent / "bench_results/tier3_run/report.md"
        if not real_lnl_path.exists():
            self.skipTest("bench_results/tier3_run/report.md not found")

        content = real_lnl_path.read_text(encoding="utf-8")
        calibrator = ReportCalibrator(content)
        cal = calibrator.calibrate()

        self.assertFalse(cal["device"]["is_bmg"], "Lunar Lake iGPU should not be classified as BMG")
        self.assertEqual(cal["device"]["compute_units"], 64)
        params = cal["parameters"]
        self.assertEqual(params["STRATA_XE2_SPLIT_K"], 1)
        self.assertEqual(params["STRATA_XE2_SLM_PAD"], 0)


if __name__ == "__main__":
    unittest.main()
