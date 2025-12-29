# ⚡ SmartOps: Adaptive Accelerator Orchestration System (AAOS)

**SmartOps** is a software-defined orchestration layer that transforms static AI accelerators (GPUs) into workload-aware, self-optimizing compute engines. By dynamically adjusting batch size, precision, and memory allocation in real-time, SmartOps maximizes effective FLOPs and prevents OOM (Out-of-Memory) crashes for massive workloads.

---

## 🛑 The Problem

Modern AI accelerators are static. They do not adapt to the workload.

* **VRAM Fragmentation:** Static batch sizes leave gigabytes of VRAM unused or cause crashes.
* **Compute Idle Time:** Memory bandwidth bottlenecks starve tensor cores.
* **Multi-Tenant Chaos:** Running multiple models (LLM + Vision) leads to resource fighting and latency spikes.

## 💡 The Solution

SmartOps introduces a **Active Feedback Loop** that monitors hardware telemetry (Temperature, VRAM, Utilization) and actuates reconfiguration logic (Precision Switching, Gradient Accumulation, Prefetching) milliseconds before a bottleneck occurs.

---

## 🏗️ System Architecture (10 Modules)

SmartOps is composed of 10 distinct software-defined modules that manage the full lifecycle of an AI workload.

| Module | Status | Function |
| --- | --- | --- |
| **1. Smart Input** | 🚧 | Bottleneck-aware data prefetching pipeline. |
| **2. VRAM Engine** | ✅ | Real-time memory pressure monitoring & defragmentation. |
| **3. Tensor Engine** | ✅ | **Dynamic Precision Switching** (FP32  FP16). |
| **4. Bandwidth Opt.** | ✅ | Adaptive compression & prefetch scaling to prevent starvation. |
| **5. Scalability** | 🚧 | Multi-GPU gradient synchronization. |
| **6. Monitoring** | ✅ | **Thermal Heatmaps** & Core Utilization tracking. |
| **7. Mode Switcher** | ✅ | Toggle between **Training** (Throughput) & **Inference** (Latency). |
| **8. Amplification** | ✅ | **Virtual Batching** via Gradient Accumulation. |
| **9. Unified Scheduler** | ✅ | Multi-tenant priority queuing (LLM + Vision + Audio). |
| **10. Orchestrator** | ✅ | The "Brain" that executes decision policies based on feedback. |

---

## 📋 Benchmarks & Standards

SmartOps optimizes workload performance against industry-standard metrics defined by **NVIDIA**, **MLPerf**, and **Enterprise AIOps** frameworks.

| Module Scope | Metric Name | Standard Source | Optimization Goal |
| --- | --- | --- | --- |
| **Compute Engine** | **SM Occupancy** | NVIDIA Nsight | Maximize active warps per SM; eliminate "dead silicon" time. |
| **Tensor Logic** | **MFU (Model FLOPs Utilization)** | Google PaLM | Maximize ratio of *observed* vs *theoretical peak* FLOPs. |
| **Memory Manager** | **DRAM Bandwidth %** | NVIDIA DCGM | Sustain >80% bandwidth saturation during training bursts. |
| **Inference Mode** | **TTFT (Time To First Token)** | vLLM / MLPerf | Minimize latency to <50ms for real-time responsiveness. |
| **Scheduler** | **Job Completion Time (JCT)** | Kubernetes / HPC | Minimize total wait+run time for multi-tenant queues. |
| **Data Pipeline** | **PCIe Throughput** | Linux / PCIe | Detect and mitigate host-to-device transfer bottlenecks. |

---

## 🧠 Core Logic Examples

### 1. The "Safety Net" (Orchestrator)

SmartOps prevents crashes by proactively slashing batch sizes or switching precision when VRAM approaches critical limits.

```python
# Simplified Logic Snippet
if vram_usage > 0.95:
    print("🚨 CRITICAL: Switching to FP16 to prevent OOM")
    self.precision = "FP16"
    self.phys_batch_size = max(8, self.phys_batch_size // 2)

```

### 2. The "Amplifier" (Virtual Batching)

When hardware limits are reached, SmartOps trades VRAM for Compute Time by using Gradient Accumulation, effectively simulating a larger GPU.

```python
# If VRAM is full, freeze physical batch, increase virtual batch
if vram_full:
    self.accumulation_steps += 1
    print(f"🔋 AMPLIFYING: Effective Batch Size -> {phys * accum}")

```

---

## 🚀 Quick Start (Simulation Mode)

This project includes a **Streamlit Dashboard** that simulates the Orchestrator's behavior on heterogeneous workloads.

### Prerequisites

* Python 3.8+
* PyTorch
* Streamlit

### Installation

```bash
git clone https://github.com/kushal-mukhopadhyay/smartops-aaos.git
cd smartops-aaos
pip install -r requirements.txt

```

### Running the Dashboard

```bash
streamlit run app.py

```

*If running on Google Colab, use the Cloudflare Tunnel wrapper included in the notebook.*

---

## 🔮 Roadmap

* **Phase 1 (Current):** Simulation Dashboard & Logic Validation.
* **Phase 2:** Hardware Hook integration (NVML / `nvidia-smi` bindings).
* **Phase 3:** Custom PyTorch `DataLoader` wrapper for transparent integration.
* **Phase 4:** Kubernetes Operator for cluster-wide deployment.
