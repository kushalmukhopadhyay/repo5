import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython.display import display, update_display
import time
import numpy as np

plt.rcParams['figure.autolayout'] = False
# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 1. SETUP VISUALIZATION (Create Once)
# ----------------------------
plt.style.use('dark_background') # Easier on the eyes for dashboards
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.suptitle("SmartOps Training Monitor (Live)", fontsize=16, color='white')

# --- Plot 1: Throughput (Line) ---
ax_th = axes[0,0]
line_th, = ax_th.plot([], [], color='#00ff41', lw=2)
ax_th.set_title("System Throughput (img/s)")
ax_th.set_xlim(0, 50) # Window size
ax_th.set_ylim(0, 100) # Initial guess
ax_th.grid(True, alpha=0.3)

# --- Plot 2: Latency (Bar) ---
ax_lat = axes[0,1]
bars_lat = ax_lat.barh(['Data', 'Compute'], [0, 0], color=['orange', 'cyan'])
ax_lat.set_title("Step Latency (ms)")
ax_lat.set_xlim(0, 50)

# --- Plot 3: Loss (Line) ---
ax_loss = axes[1,0]
line_loss, = ax_loss.plot([], [], color='#ff0055', lw=2)
ax_loss.set_title("Training Loss")
ax_loss.set_xlim(0, 50)
ax_loss.set_ylim(0, 5)

# --- Plot 4: Stability (Fill) ---
ax_stab = axes[1,1]
x_data = np.arange(50)
y_data = np.zeros(50)
poly_stab = ax_stab.fill_between(x_data, y_data, color='purple', alpha=0.5)
ax_stab.set_title("Gradient Stability Score")
ax_stab.set_ylim(0, 1)

plt.tight_layout()
# create a unique display ID to update later
display_handle = display(fig, display_id=True)

# ----------------------------
# 2. LOGIC & DATA (Same "100/100" logic)
# ----------------------------
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
    nn.Linear(32, 10)
).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Data simulation
batch_size = 32
batches = 100
history_len = 50

# Pre-allocate arrays for smooth scrolling
hist_throughput = np.zeros(history_len)
hist_loss = np.zeros(history_len)
hist_stability = np.zeros(history_len)

print("Initializing smooth dashboard...")

# ----------------------------
# 3. THE SMOOTH LOOP
# ----------------------------
for i in range(batches):
    t0 = time.perf_counter()

    # -- Simulate Data Load --
    time.sleep(np.random.uniform(0.001, 0.01)) # Tiny sleep to simulate IO
    t_data = (time.perf_counter() - t0) * 1000

    # -- Simulate Training --
    x = torch.randn(batch_size, 3, 64, 64, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)

    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()

    # -- Metrics --
    dt = time.perf_counter() - t0
    t_compute = dt * 1000
    current_throughput = batch_size / dt

    # -- Update History Arrays (Scrolling Buffer) --
    hist_throughput = np.roll(hist_throughput, -1)
    hist_throughput[-1] = current_throughput

    hist_loss = np.roll(hist_loss, -1)
    hist_loss[-1] = loss.item()

    hist_stability = np.roll(hist_stability, -1)
    hist_stability[-1] = np.random.uniform(0.8, 1.0) # Simulated stability for demo

    # ----------------------------
    # 4. UPDATE VISUALS (No Clear Output!)
    # ----------------------------
    if i % 2 == 0: # Update every 2nd frame for 60fps feel

        # Update Throughput Line
        line_th.set_data(np.arange(history_len), hist_throughput)
        ax_th.set_ylim(0, np.max(hist_throughput) * 1.2) # Auto-scale Y

        # Update Latency Bars
        bars_lat[0].set_width(t_data)
        bars_lat[1].set_width(t_compute)

        # Update Loss Line
        line_loss.set_data(np.arange(history_len), hist_loss)
        ax_loss.set_ylim(0, np.max(hist_loss) + 0.5)

        # Update Stability Poly (Tricky: requires creating new collection path)
        # For speed, we just update the axes limits or use a line instead,
        # but here is how to update a fill:
        dummy = ax_stab.plot(np.arange(history_len), hist_stability, color='purple')
        # (Updating fill_between is complex in matplotlib, usually easier to clear just that ax
        # but for max speed we stick to lines usually. Here we just overlay lines for speed)

        # PUSH THE UPDATE
        update_display(fig, display_id=display_handle.display_id)

print("Training Complete.")
plt.close() # Clean up static image at the end

%%writefile app.py
import streamlit as st
import time
import numpy as np
import pandas as pd

# ----------------------------
# 1. MODULE 10 + MODULE 3 (Orchestrator + Tensor Engine)
# ----------------------------
class SmartOpsEngine:
    def __init__(self):
        self.batch_size = 32
        self.precision = "FP32" # Starts in High Precision
        self.cooldown = 0

    def decide(self, vram_pct):
        """
        Decides on Batch Size AND Precision based on VRAM pressure.
        """
        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        action = None

        # --- CRITICAL EMERGENCY LOGIC ---
        # If we are about to crash (>95%) and still in FP32,
        # SWITCH TO FP16 IMMEDIATELY.
        if vram_pct > 0.95 and self.precision == "FP32":
            self.precision = "FP16"
            action = "🚨 CRITICAL: SWITCHING TO FP16 (Mixed Precision)"
            self.cooldown = 8 # Give time to stabilize
            return action

        # --- STANDARD TUNING LOGIC ---

        # Rule A: PROTECT (High VRAM)
        if vram_pct > 0.85:
            new_bs = int(self.batch_size * 0.8)
            self.batch_size = max(new_bs, 8)
            action = f"⚠️ PROTECT: Lowering Batch Size -> {self.batch_size}"
            self.cooldown = 2

        # Rule B: AMPLIFY (Low VRAM)
        elif vram_pct < 0.70:
            new_bs = int(self.batch_size * 1.2)
            self.batch_size = min(new_bs, 512)
            action = f"🚀 AMPLIFY: Increasing Batch Size -> {self.batch_size}"
            self.cooldown = 2

        return action

# ----------------------------
# 2. UI SETUP
# ----------------------------
st.set_page_config(page_title="SmartOps Tensor Engine", page_icon="⚡", layout="wide")
st.markdown("<style>.stApp {background-color: #0e1117;}</style>", unsafe_allow_html=True)

st.title("⚡ SmartOps: Dynamic Tensor Engine")

# Top Metrics
m1, m2, m3, m4 = st.columns(4)
kpi_prec = m1.empty()
kpi_batch = m2.empty()
kpi_vram = m3.empty()
kpi_msg = m4.empty()

st.divider()

# Charts
c1, c2 = st.columns([3, 1])
chart_vram = c1.line_chart(None, height=300)
log_area = c2.empty()

# ----------------------------
# 3. SIMULATION LOOP
# ----------------------------
if st.button("Initialize Tensor Engine"):
    engine = SmartOpsEngine()

    # History buffers
    hist_vram = []
    logs = []

    # Hardware State
    current_vram = 0.5

    for step in range(200):
        # --- SIMULATE HARDWARE PHYSICS ---

        # Base VRAM cost depends on Batch Size
        base_load = engine.batch_size / 200

        # FP16 uses 40% less memory than FP32
        if engine.precision == "FP16":
            base_load = base_load * 0.6

        # Add noise and smoothing
        target = base_load + np.random.normal(0, 0.02)
        current_vram = (current_vram * 0.7) + (target * 0.3)
        current_vram = max(0, min(1.0, current_vram))

        # Slow down loop for visual effect
        time.sleep(0.05)

        # --- ORCHESTRATOR DECISION ---
        decision = engine.decide(current_vram)

        # --- UPDATE UI ---

        # 1. Precision Badge
        if engine.precision == "FP32":
            kpi_prec.metric("Precision Mode", "FP32 (High)", delta_color="off")
        else:
            kpi_prec.metric("Precision Mode", "FP16 (Turbo)", delta="2x Speed", delta_color="normal")

        kpi_batch.metric("Batch Size", engine.batch_size)
        kpi_vram.metric("VRAM Usage", f"{current_vram*100:.1f}%")

        if decision:
            kpi_msg.warning(decision)
            logs.append(f"[{step}] {decision}")
            log_area.text_area("System Logs", "\n".join(logs[-10:]), height=300)
        else:
            kpi_msg.success("Optimizing...")

        # Chart Update
        hist_vram.append(current_vram)
        chart_vram.line_chart(pd.DataFrame({"VRAM Usage": hist_vram[-60:]}))

    st.success("Optimization Cycle Complete.")

import os
import time
import subprocess

# 1. FORCE INSTALL DEPENDENCIES (Crucial if runtime reset)
print("📦 Installing dependencies...")
subprocess.run(["pip", "install", "streamlit", "GPUtil", "pandas", "numpy", "torch"], check=True)

# 2. RE-WRITE THE APP CODE (To ensure it's clean)
code = """
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import pandas as pd
import GPUtil

# --- 1. THE BRAIN: ORCHESTRATOR ---
class Orchestrator:
    def __init__(self, initial_batch_size=32):
        self.batch_size = initial_batch_size
        self.cooldown = 0

    def decide(self, metrics):
        if self.cooldown > 0:
            self.cooldown -= 1
            return None
        vram_pct = metrics.get('vram_pct', 0)
        action = None

        # Rule A: AMPLIFY
        if vram_pct < 0.70:
            new_bs = int(self.batch_size * 1.2)
            self.batch_size = min(new_bs, 512)
            action = f"AMPLIFY: Batch -> {self.batch_size}"
            self.cooldown = 5

        # Rule B: PROTECT
        elif vram_pct > 0.90:
            new_bs = int(self.batch_size * 0.8)
            self.batch_size = max(new_bs, 8)
            action = f"PROTECT: Batch -> {self.batch_size}"
            self.cooldown = 3
        return action

# --- 2. UI SETUP ---
st.set_page_config(page_title="SmartOps Orchestrator", page_icon="🧠", layout="wide")
st.markdown("<style>.stApp {background-color: #0e1117;}</style>", unsafe_allow_html=True)
st.title("🧠 SmartOps: Autonomous Orchestrator Mode")

col1, col2, col3, col4 = st.columns(4)
kpi_step = col1.empty()
kpi_batch = col2.empty()
kpi_vram = col3.empty()
kpi_action = col4.empty()
st.divider()
chart_vram = st.line_chart(None, height=300)

if st.button("Engage Auto-Pilot"):
    brain = Orchestrator(initial_batch_size=32)
    hist_vram = []
    hist_batch = []
    current_vram = 0.5

    for step in range(100):
        # Simulation
        target = (brain.batch_size / 200) + np.random.normal(0, 0.02)
        current_vram = (current_vram * 0.8) + (target * 0.2)
        current_vram = max(0, min(1.0, current_vram))
        time.sleep(0.05)

        # Logic
        decision = brain.decide({'vram_pct': current_vram})

        # UI
        kpi_step.metric("Step", step)
        kpi_batch.metric("Batch Size", brain.batch_size)
        kpi_vram.metric("VRAM", f"{current_vram*100:.1f}%")
        if decision: kpi_action.warning(decision)
        else: kpi_action.success("Stable")

        hist_vram.append(current_vram)
        chart_vram.line_chart(pd.DataFrame({"VRAM": hist_vram[-50:]}))
    st.success("Complete")
"""
with open("app.py", "w") as f:
    f.write(code)
print("✅ app.py written successfully.")

# 3. LAUNCH SERVER AND CAPTURE LOGS
print("🚀 Starting Streamlit...")
# Kill old
os.system("pkill streamlit")
os.system("pkill cloudflared")

# Start with log capture
process = subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# 4. HEALTH CHECK (Wait 10s)
print("⏳ Waiting for health check...", end="")
time.sleep(5)
if process.poll() is not None:
    # IT CRASHED! PRINT ERROR
    print("\n❌ CRASH DETECTED!")
    out, err = process.communicate()
    print("--- ERROR LOG ---")
    print(err.decode())
else:
    print("\n✅ Streamlit is HEALTHY!")

    # 5. START TUNNEL
    print("🔗 Starting Cloudflare Tunnel...")
    if not os.path.exists("cloudflared-linux-amd64"):
        os.system("wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64")
        os.system("chmod +x cloudflared-linux-amd64")

    !./cloudflared-linux-amd64 tunnel --url http://localhost:8501

%%writefile app.py
import streamlit as st
import time
import numpy as np
import pandas as pd

# ----------------------------
# LOGIC CORE: TRIPLE-LEVER ENGINE
# ----------------------------
class SmartOpsEngine:
    def __init__(self):
        # Hardware Levers
        self.phys_batch = 32      # Real data on GPU
        self.precision = "FP32"   # Quality
        self.accum_steps = 1      # Virtual Multiplier

        # Internal State
        self.cooldown = 0

    @property
    def virtual_batch_size(self):
        return self.phys_batch * self.accum_steps

    def decide(self, vram_pct):
        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        action = None

        # --- 1. EMERGENCY DEFENSE (VRAM > 95%) ---
        if vram_pct > 0.95:
            if self.precision == "FP32":
                self.precision = "FP16"
                action = "🚨 CRITICAL: SWITCH TO FP16"
            else:
                # If already FP16, we must slash physical batch
                self.phys_batch = max(8, int(self.phys_batch * 0.5))
                # BUT we increase accumulation to keep virtual batch high!
                self.accum_steps += 2
                action = "🛡️ OOM PREVENTED: Slash Physical, Boost Accumulation"
            self.cooldown = 5
            return action

        # --- 2. OPTIMIZATION (VRAM < 70%) ---
        if vram_pct < 0.70:
            # We have free VRAM! Fill it up physically (faster than accumulation)
            new_bs = int(self.phys_batch * 1.2)
            self.phys_batch = min(new_bs, 128)

            # If physical is high, reduce accumulation (optimization)
            if self.accum_steps > 1:
                self.accum_steps -= 1

            action = f"🚀 AMPLIFY: Phys Batch -> {self.phys_batch}"
            self.cooldown = 3

        return action

# ----------------------------
# UI SETUP
# ----------------------------
st.set_page_config(page_title="SmartOps Amplifier", page_icon="🔋", layout="wide")
st.markdown("<style>.stApp {background-color: #0e1117;}</style>", unsafe_allow_html=True)

st.title("🔋 SmartOps: Compute Amplification Layer")

# METRICS ROW
c1, c2, c3, c4 = st.columns(4)
kpi_virt = c1.empty()
kpi_phys = c2.empty()
kpi_acc = c3.empty()
kpi_vram = c4.empty()

st.divider()

# VISUALS
g1, g2 = st.columns([3, 1])
chart_perf = g1.line_chart(None, height=300)
log_box = g2.empty()

# ----------------------------
# SIMULATION LOOP
# ----------------------------
if st.button("Activate Amplification Layer"):
    engine = SmartOpsEngine()

    hist_virt = []
    logs = []
    current_vram = 0.5

    for step in range(200):
        # --- SIMULATE PHYSICS ---
        # VRAM depends ONLY on Physical Batch & Precision
        load_factor = 1.0 if engine.precision == "FP32" else 0.6
        target_vram = (engine.phys_batch / 150) * load_factor

        # Add realistic noise
        target_vram += np.random.normal(0, 0.03)
        current_vram = (current_vram * 0.7) + (target_vram * 0.3)
        current_vram = max(0, min(1.0, current_vram))

        time.sleep(0.05)

        # --- ENGINE DECISION ---
        decision = engine.decide(current_vram)

        # --- UI UPDATE ---

        # Metric 1: The "Effective" Power (Virtual Batch)
        kpi_virt.metric("Virtual Batch Size", engine.virtual_batch_size,
                        delta="Target Power", delta_color="normal")

        # Metric 2: The "Real" Load (Physical Batch)
        kpi_phys.metric("Physical Batch (VRAM)", engine.phys_batch)

        # Metric 3: The Amplifier (Accumulation)
        kpi_acc.metric("Accumulation Steps", f"{engine.accum_steps}x",
                       delta="Amplification" if engine.accum_steps > 1 else "Off")

        # Metric 4: Health
        kpi_vram.metric("VRAM Usage", f"{current_vram*100:.1f}%")

        if decision:
            logs.append(f"[{step}] {decision}")
            log_box.text_area("Orchestrator Logs", "\n".join(logs[-15:]), height=300)

        # Chart: Show Virtual vs Physical gap
        hist_virt.append(engine.virtual_batch_size)
        chart_perf.line_chart(pd.DataFrame({
            "Virtual Batch (Effective)": hist_virt[-60:],
            # Plot physical scaled so it's visible on same axis
            "Physical Batch (Hardware)": [engine.phys_batch for _ in range(len(hist_virt[-60:]))]
        }))

    st.success("Amplification Test Complete.")

%%writefile app.py
import streamlit as st
import time
import numpy as np
import pandas as pd

# ----------------------------
# LOGIC CORE: MULTI-MODE ENGINE
# ----------------------------
class SmartOpsEngine:
    def __init__(self):
        # Hardware State
        self.phys_batch = 32
        self.precision = "FP32"
        self.accum_steps = 1

        # Mode State
        self.mode = "TRAINING" # Default
        self.cooldown = 0

    def set_mode(self, new_mode):
        if self.mode != new_mode:
            self.mode = new_mode
            # RESET parameters for the new mode
            if new_mode == "INFERENCE":
                self.phys_batch = 1 # Start fast
                self.accum_steps = 0 # No accumulation needed
                self.precision = "FP16" # Usually default for inference
            else:
                self.phys_batch = 32
                self.accum_steps = 1
            return f"🔄 RECONFIGURING PIPELINE -> {new_mode}"
        return None

    def decide(self, vram_pct, latency_ms):
        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        action = None

        # ==========================================
        # LOGIC FOR INFERENCE (The Sprinter)
        # Goal: Keep Latency under 50ms
        # ==========================================
        if self.mode == "INFERENCE":
            # If too slow (>50ms), we must optimize
            if latency_ms > 50:
                if self.phys_batch > 1:
                    self.phys_batch = max(1, int(self.phys_batch * 0.8))
                    action = f"⚡ LATENCY SPIKE ({latency_ms:.1f}ms): Shrinking Batch"
                elif self.precision == "FP32":
                    self.precision = "FP16"
                    action = "⚡ LATENCY CRITICAL: Quantizing to FP16"

            # If very fast (<10ms) and low VRAM, we can handle more users (batching)
            elif latency_ms < 10 and vram_pct < 0.3:
                self.phys_batch += 1
                action = f"🟢 HEADROOM DETECTED: Increasing Concurrent Users -> {self.phys_batch}"

        # ==========================================
        # LOGIC FOR TRAINING (The Bodybuilder)
        # Goal: Fill VRAM, Don't Crash
        # ==========================================
        else:
            # (Previous Logic: VRAM Protection & Amplification)
            if vram_pct > 0.95:
                if self.precision == "FP32":
                    self.precision = "FP16"
                    action = "🚨 CRITICAL: SWITCH TO FP16"
                else:
                    self.phys_batch = max(8, int(self.phys_batch * 0.8))
                    self.accum_steps += 1
                    action = "🛡️ OOM DEFENSE: Amplifying Accumulation"
                self.cooldown = 5

            elif vram_pct < 0.70:
                new_bs = int(self.phys_batch * 1.2)
                self.phys_batch = min(new_bs, 128)
                action = f"🚀 AMPLIFY: Phys Batch -> {self.phys_batch}"
                self.cooldown = 2

        return action

# ----------------------------
# UI SETUP
# ----------------------------
st.set_page_config(page_title="SmartOps Modes", page_icon="🎛️", layout="wide")
st.markdown("<style>.stApp {background-color: #0e1117;}</style>", unsafe_allow_html=True)

st.title("🎛️ SmartOps: Accelerator Mode Switcher")

# CONTROL PANEL
col_mode, col_status = st.columns([1, 3])
with col_mode:
    target_mode = st.radio("System Objective:", ["TRAINING", "INFERENCE"], horizontal=True)

# METRICS
m1, m2, m3, m4 = st.columns(4)
kpi_prim = m1.empty() # Primary Metric (changes based on mode)
kpi_batch = m2.empty()
kpi_prec = m3.empty()
kpi_hw = m4.empty()

st.divider()

# CHART
c1, c2 = st.columns([3, 1])
chart_main = c1.line_chart(None, height=300)
log_box = c2.empty()

# ----------------------------
# SIMULATION LOOP
# ----------------------------
if st.button("Start System"):
    engine = SmartOpsEngine()
    logs = []
    hist_metric = []

    current_vram = 0.5
    current_latency = 20.0 # ms

    for step in range(300):
        # 1. HANDLE MODE SWITCHING
        mode_msg = engine.set_mode(target_mode)
        if mode_msg:
            logs.append(f"[{step}] {mode_msg}")
            # Reset simulated hardware
            current_vram = 0.1 if target_mode == "INFERENCE" else 0.5
            current_latency = 10.0

        # 2. SIMULATE PHYSICS
        # Training Physics: High Batch = High VRAM
        # Inference Physics: High Batch = High Latency

        if engine.mode == "TRAINING":
            target_vram = (engine.phys_batch / 150) * (0.6 if engine.precision=="FP16" else 1.0)
            current_vram = (current_vram * 0.8) + (target_vram * 0.2) + np.random.normal(0, 0.01)
            primary_val = engine.phys_batch * engine.accum_steps # Throughput
            primary_label = "Effective Batch Size"
        else:
            # Latency grows with batch size
            base_lat = 5.0 + (engine.phys_batch * 2.0)
            if engine.precision == "FP16": base_lat *= 0.6 # Faster

            # Add noise (network jitter)
            current_latency = (current_latency * 0.5) + (base_lat * 0.5) + np.random.normal(0, 2.0)
            current_vram = 0.1 + (engine.phys_batch * 0.05) # Inference uses less VRAM

            primary_val = current_latency
            primary_label = "Latency (ms)"

        time.sleep(0.05)

        # 3. ENGINE DECIDE
        decision = engine.decide(current_vram, current_latency)

        # 4. UPDATE UI
        kpi_prim.metric(primary_label, f"{primary_val:.1f}")
        kpi_batch.metric("Physical Batch", engine.phys_batch)
        kpi_prec.metric("Precision", engine.precision)
        kpi_hw.metric("VRAM Usage", f"{current_vram*100:.0f}%")

        if decision:
            logs.append(f"[{step}] {decision}")

        log_box.text_area("System Events", "\n".join(logs[-15:]), height=300)

        hist_metric.append(primary_val)
        chart_main.line_chart(pd.DataFrame({primary_label: hist_metric[-60:]}))

    st.success("Simulation Ended")

%%writefile app.py
import streamlit as st
import time
import numpy as np
import pandas as pd

# ----------------------------
# LOGIC CORE: BANDWIDTH ENGINE
# ----------------------------
class SmartOpsEngine:
    def __init__(self):
        # Hardware Config
        self.phys_batch = 32
        self.prefetch_factor = 0 # Starts Off
        self.compression = False # Starts Off

        # State
        self.cooldown = 0

    def decide(self, data_wait_ms, gpu_util_pct):
        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        action = None

        # --- BANDWIDTH BOTTLENECK DETECTED ---
        if data_wait_ms > 15 and gpu_util_pct < 0.50:
            if self.prefetch_factor < 4:
                self.prefetch_factor += 2
                action = f"🚚 STARVATION DETECTED: Boost Prefetch -> {self.prefetch_factor}x"
            elif not self.compression:
                self.compression = True
                action = "🗜️ BANDWIDTH CRITICAL: Enabling Tensor Compression"
            self.cooldown = 4
            return action

        # --- OPTIMIZATION ---
        if data_wait_ms < 5 and gpu_util_pct > 0.80:
            if self.compression:
                self.compression = False
                action = "✅ BANDWIDTH STABLE: Disabling Compression"
                self.cooldown = 4

        return action

# ----------------------------
# UI SETUP
# ----------------------------
st.set_page_config(page_title="SmartOps Bandwidth", page_icon="🚚", layout="wide")
st.markdown("<style>.stApp {background-color: #0e1117;}</style>", unsafe_allow_html=True)

st.title("🚚 SmartOps: Data Movement Optimizer")

# METRICS
m1, m2, m3, m4 = st.columns(4)
kpi_wait = m1.empty()
kpi_gpu = m2.empty()
kpi_pre = m3.empty()
kpi_comp = m4.empty()

st.divider()

# CHART
c1, c2 = st.columns([3, 1])
chart_wait = c1.line_chart(None, height=300)
log_box = c2.empty()

# ----------------------------
# SIMULATION LOOP
# ----------------------------
if st.button("Simulate Data Stream"):
    engine = SmartOpsEngine()
    logs = []
    hist_wait = []

    current_wait = 30.0
    current_gpu = 0.30

    for step in range(200):
        # 1. SIMULATE PHYSICS
        reduction = (engine.prefetch_factor * 5.0)
        if engine.compression: reduction += 8.0
        target_wait = max(2.0, 30.0 - reduction)
        current_wait = (current_wait * 0.7) + (target_wait * 0.3) + np.random.normal(0, 2.0)
        target_gpu = 1.0 - (current_wait / 40.0)
        current_gpu = max(0, min(1.0, target_gpu))

        time.sleep(0.05)

        # 2. ENGINE DECIDE
        decision = engine.decide(current_wait, current_gpu)

        # 3. UPDATE UI
        kpi_wait.metric("Data Wait Time", f"{current_wait:.1f} ms", delta_color="inverse")
        kpi_gpu.metric("GPU Utilization", f"{current_gpu*100:.0f}%")
        kpi_pre.metric("Prefetch Factor", f"{engine.prefetch_factor}x")
        kpi_comp.metric("Compression", "ON" if engine.compression else "OFF")

        if decision:
            logs.append(f"[{step}] {decision}")

        # FIX: Added unique key using 'step' to prevent DuplicateID error
        log_box.text_area("Orchestrator Logs", "\n".join(logs[-15:]), height=300, key=f"log_{step}")

        hist_wait.append(current_wait)
        chart_wait.line_chart(pd.DataFrame({"CPU Wait Latency (ms)": hist_wait[-60:]}))

    st.success("Bandwidth Optimization Complete.")

%%writefile app.py
import streamlit as st
import time
import numpy as np
import pandas as pd

# ----------------------------
# LOGIC CORE
# ----------------------------
class SmartOpsEngine:
    def __init__(self):
        self.mode = "TRAINING"
        self.phys_batch = 32
        self.cooldown = 0

    def generate_core_map(self):
        # Create an 8x8 grid representing 64 Tensor Cores
        if self.mode == "TRAINING":
            # Training uses ALL cores heavily
            base = 0.8
            noise = 0.2
        else:
            # Inference uses fewer cores (sparse activation)
            base = 0.1
            noise = 0.4

        # Generate grid with random heat
        cores = np.random.normal(base, 0.1, (8, 8))

        # Add "Hotspots" (random spikes)
        cores[np.random.randint(0,8), np.random.randint(0,8)] += noise

        # Clip to 0.0 - 1.0
        return np.clip(cores, 0, 1)

# ----------------------------
# UI SETUP
# ----------------------------
st.set_page_config(page_title="SmartOps Thermal Vision", page_icon="🔥", layout="wide")
st.markdown("<style>.stApp {background-color: #0e1117;}</style>", unsafe_allow_html=True)

st.title("🔥 SmartOps: GPU Core Thermal Vision")

# CONTROLS
mode = st.radio("Workload Type", ["TRAINING (Dense)", "INFERENCE (Sparse)"], horizontal=True)

# LAYOUT
col_heat, col_stats = st.columns([1, 1])

with col_heat:
    st.subheader("Tensor Core Activity (8x8 Block)")
    # We use an empty container to update the dataframe safely
    heatmap_container = st.empty()

with col_stats:
    st.subheader("Live Metrics")
    m1 = st.empty()
    m2 = st.empty()
    m3 = st.empty()
    log_box = st.empty()

# ----------------------------
# SIMULATION LOOP
# ----------------------------
if st.button("Activate Thermal Sensors"):
    engine = SmartOpsEngine()
    engine.mode = mode.split()[0] # Get "TRAINING" or "INFERENCE"

    logs = []

    for step in range(200):
        # 1. Generate Data
        core_grid = engine.generate_core_map()
        avg_temp = np.mean(core_grid) * 100
        peak_temp = np.max(core_grid) * 100

        # 2. Update Heatmap (The Cool Part)
        # We use Pandas Styler to color the grid
        df = pd.DataFrame(core_grid)
        styled_df = df.style.background_gradient(cmap='inferno', axis=None, vmin=0, vmax=1)
        heatmap_container.dataframe(styled_df, height=300, use_container_width=True)

        # 3. Update Metrics
        m1.metric("Average Core Load", f"{avg_temp:.1f}%")
        m2.metric("Peak Core Temp", f"{peak_temp:.1f}°C",
                  delta="THROTTLING" if peak_temp > 95 else "Nominal",
                  delta_color="inverse")
        m3.metric("Active Cores", f"{np.sum(core_grid > 0.3)} / 64")

        # 4. Logs
        if peak_temp > 95:
            logs.append(f"[{step}] ⚠️ THERMAL THROTTLING DETECTED on Core {np.argmax(core_grid)}")

        log_box.text_area("System Events", "\n".join(logs[-10:]), height=200, key=f"log_{step}")

        time.sleep(0.1) # Slightly slower to let you see the colors change

    st.success("Monitoring Session Ended.")

%%writefile app.py
import streamlit as st
import time
import numpy as np
import pandas as pd

# ----------------------------
# LOGIC CORE: UNIFIED SCHEDULER
# ----------------------------
class Job:
    def __init__(self, name, priority, vram_req):
        self.name = name
        self.priority = priority # 1 = High, 3 = Low
        self.vram_req = vram_req
        self.status = "PENDING" # PENDING, RUNNING, COMPLETED
        self.progress = 0

class Scheduler:
    def __init__(self):
        self.vram_capacity = 100 # Total "Units" of VRAM
        self.active_jobs = []
        self.queue = []

    def add_job(self, job):
        self.queue.append(job)

    def tick(self):
        # 1. Calculate Used VRAM
        used_vram = sum([j.vram_req for j in self.active_jobs])
        free_vram = self.vram_capacity - used_vram

        # 2. Try to Schedule Pending Jobs (Highest Priority First)
        self.queue.sort(key=lambda x: x.priority)

        for job in self.queue[:]: # Copy list to modify safely
            if job.status == "PENDING":
                if job.vram_req <= free_vram:
                    # Allocate!
                    job.status = "RUNNING"
                    self.active_jobs.append(job)
                    self.queue.remove(job)
                    free_vram -= job.vram_req
                else:
                    # Not enough space
                    job.status = "QUEUED (OOM)"

        # 3. Update Progress of Running Jobs
        completed_jobs = []
        for job in self.active_jobs:
            # Simulate work
            speed = np.random.randint(2, 5)
            job.progress += speed

            if job.progress >= 100:
                job.progress = 100
                job.status = "COMPLETED"
                completed_jobs.append(job)

        # 4. Cleanup Completed
        for job in completed_jobs:
            self.active_jobs.remove(job)

        return used_vram

# ----------------------------
# UI SETUP
# ----------------------------
st.set_page_config(page_title="SmartOps Scheduler", page_icon="🚥", layout="wide")
st.markdown("<style>.stApp {background-color: #0e1117;}</style>", unsafe_allow_html=True)

st.title("🚥 SmartOps: Multi-Tenant Scheduler")

# GLOBAL METRICS
g1, g2 = st.columns(2)
kpi_total_vram = g1.empty()
kpi_active = g2.empty()

st.divider()

# JOB COLUMNS
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("🤖 LLM-7B (Chat)")
    st.caption("Priority: HIGH | VRAM: 50%")
    stat_1 = st.empty()
    bar_1 = st.progress(0)

with c2:
    st.subheader("👁️ ViT-Huge (Vision)")
    stat_2 = st.empty()
    st.caption("Priority: MEDIUM | VRAM: 30%")
    bar_2 = st.progress(0)

with c3:
    st.subheader("🎙️ Whisper (Audio)")
    stat_3 = st.empty()
    st.caption("Priority: LOW | VRAM: 15%")
    bar_3 = st.progress(0)

# LOGS
st.divider()
log_box = st.empty()

# ----------------------------
# SIMULATION LOOP
# ----------------------------
if st.button("Start Scheduling Cycle"):
    sched = Scheduler()
    logs = []

    # Define Jobs
    job_llm = Job("LLM-7B", priority=1, vram_req=50)
    job_vis = Job("ViT-Huge", priority=2, vram_req=30)
    job_audio = Job("Whisper", priority=3, vram_req=15)

    # Add them to the system randomly over time
    all_jobs = [job_llm, job_vis, job_audio]
    current_time = 0

    # Map jobs to UI elements
    ui_map = {
        "LLM-7B": (stat_1, bar_1),
        "ViT-Huge": (stat_2, bar_2),
        "Whisper": (stat_3, bar_3)
    }

    # Add first job immediately
    sched.add_job(job_llm)
    logs.append("[0s] New Job Arrived: LLM-7B")

    for step in range(100):
        # Trigger new jobs arriving at different times
        if step == 10:
            sched.add_job(job_vis)
            logs.append(f"[{step}s] New Job Arrived: ViT-Huge")

        if step == 20:
            sched.add_job(job_audio)
            logs.append(f"[{step}s] New Job Arrived: Whisper")

        # Run Scheduler Logic
        used_vram = sched.tick()

        # UPDATE UI
        kpi_total_vram.metric("Total System VRAM Used", f"{used_vram}%", f"{100-used_vram}% Free")
        kpi_active.metric("Active Processes", len(sched.active_jobs))

        # Update Individual Job Cards
        for j in [job_llm, job_vis, job_audio]:
            ui_stat, ui_bar = ui_map[j.name]

            # Color logic based on status
            status_color = "red"
            if j.status == "RUNNING": status_color = "green"
            if j.status == "COMPLETED": status_color = "blue"

            ui_stat.markdown(f"Status: :{status_color}[**{j.status}**]")
            ui_bar.progress(j.progress)

        # Logs
        # FIX: Unique Key added
        log_box.text_area("Scheduler Event Log", "\n".join(logs[-8:]), height=150, key=f"log_{step}")

        time.sleep(0.15)

        if job_llm.status == "COMPLETED" and job_vis.status == "COMPLETED" and job_audio.status == "COMPLETED":
            break

    st.success("All Jobs Completed Successfully.")

import os
import time
import subprocess

print("🚀 Relaunching Dashboard...")
os.system("pkill streamlit")
os.system("pkill cloudflared")

# Start Streamlit
subprocess.Popen(["nohup", "streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true", "&"])

# Wait for it
time.sleep(5)

# Start Tunnel
print("🔗 Click the link below:")
if not os.path.exists("cloudflared-linux-amd64"):
    os.system("wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64")
    os.system("chmod +x cloudflared-linux-amd64")

!./cloudflared-linux-amd64 tunnel --url http://localhost:8501
