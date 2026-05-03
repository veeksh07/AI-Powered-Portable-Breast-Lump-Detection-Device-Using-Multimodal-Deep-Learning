"""
Pressure Sensor Data Simulator
Generates synthetic pressure sensor readings with trends, noise, spikes, and drift.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ── Core signal generators ──────────────────────────────────────────────────

def generate_pressure_signal(
    duration_s=60,
    sample_rate_hz=100,
    baseline_kpa=101.325,       # standard atmospheric pressure
    trend_kpa_per_s=0.0,        # slow linear drift
    sine_amp_kpa=2.0,           # periodic fluctuation amplitude
    sine_freq_hz=0.1,           # periodic fluctuation frequency
    noise_std_kpa=0.3,          # white Gaussian noise
    spike_rate=0.005,           # probability of a spike per sample
    spike_amp_kpa=10.0,         # spike magnitude
    drift_rate=0.002,           # random walk step size per sample
):
    """
    Simulate a realistic pressure sensor time-series.

    Returns
    -------
    t        : 1-D float array  — time axis in seconds
    pressure : 1-D float array  — pressure readings in kPa
    """
    n = int(duration_s * sample_rate_hz)
    t = np.linspace(0, duration_s, n)

    # 1. Baseline + linear trend
    signal = baseline_kpa + trend_kpa_per_s * t

    # 2. Periodic component (e.g. pump oscillation)
    signal += sine_amp_kpa * np.sin(2 * np.pi * sine_freq_hz * t)

    # 3. White noise
    signal += np.random.normal(0, noise_std_kpa, n)

    # 4. Random walk drift (sensor aging / temperature effect)
    drift = np.cumsum(np.random.normal(0, drift_rate, n))
    signal += drift

    # 5. Random spikes (transient events / EMI)
    spike_mask = np.random.rand(n) < spike_rate
    signs = np.random.choice([-1, 1], size=n)
    signal += spike_mask * signs * spike_amp_kpa

    return t, signal.astype(np.float32)


def add_sensor_fault(pressure, t, fault_start_s, fault_end_s, fault_type="stuck"):
    """
    Inject a sensor fault into a pressure signal.

    fault_type options
    ------------------
    'stuck'    : output freezes at the value at fault_start
    'dropout'  : output goes to 0 (disconnected)
    'noise'    : output becomes pure high-frequency noise
    """
    pressure = pressure.copy()
    sample_rate = len(t) / t[-1]
    i0 = int(fault_start_s * sample_rate)
    i1 = int(fault_end_s   * sample_rate)

    if fault_type == "stuck":
        pressure[i0:i1] = pressure[i0]
    elif fault_type == "dropout":
        pressure[i0:i1] = 0.0
    elif fault_type == "noise":
        pressure[i0:i1] = np.random.normal(pressure[i0], 20, i1 - i0)

    return pressure


# ── Multi-sensor array ──────────────────────────────────────────────────────

def generate_sensor_array(n_sensors=4, duration_s=60, sample_rate_hz=100, seed=42):
    """
    Simulate an array of pressure sensors with slight individual variations.

    Returns
    -------
    t        : 1-D float array of shape (N,)
    readings : 2-D float array of shape (n_sensors, N)
    """
    np.random.seed(seed)
    all_readings = []
    t = None
    for i in range(n_sensors):
        t, p = generate_pressure_signal(
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
            baseline_kpa=101.325 + np.random.uniform(-2, 2),
            sine_amp_kpa=np.random.uniform(1, 4),
            sine_freq_hz=np.random.uniform(0.05, 0.2),
            noise_std_kpa=np.random.uniform(0.1, 0.5),
            spike_rate=np.random.uniform(0.002, 0.01),
        )
        all_readings.append(p)
    return t, np.array(all_readings)


# ── Plotting helpers ────────────────────────────────────────────────────────

def plot_single_sensor(t, pressure, title="Pressure Sensor Simulation", save_path=None):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # --- Raw signal ---
    axes[0].plot(t, pressure, color="steelblue", linewidth=0.8, label="Pressure (kPa)")
    axes[0].set_ylabel("Pressure (kPa)")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.4)

    # --- Rolling statistics (window = 1 s worth of samples) ---
    window = max(1, len(t) // int(t[-1]))
    roll_mean = np.convolve(pressure, np.ones(window) / window, mode="same")
    roll_std  = np.array([pressure[max(0, i - window // 2):i + window // 2].std()
                          for i in range(len(pressure))])

    axes[1].plot(t, roll_mean,  color="darkorange", linewidth=1.2, label="Rolling mean")
    axes[1].fill_between(t, roll_mean - roll_std, roll_mean + roll_std,
                         alpha=0.3, color="darkorange", label="±1 std")
    axes[1].set_ylabel("kPa")
    axes[1].set_title("Rolling Mean ± Std Dev (1-second window)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.4)

    # --- Pressure histogram ---
    axes[2].hist(pressure, bins=80, color="mediumseagreen", edgecolor="none", alpha=0.85)
    axes[2].set_xlabel("Time (s)" if False else "Pressure (kPa)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Pressure Distribution")
    axes[2].grid(True, alpha=0.4)
    axes[2].set_xlabel("Pressure (kPa)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


def plot_sensor_array(t, readings, save_path=None):
    n = readings.shape[0]
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    for i, (ax, color) in enumerate(zip(axes, colors)):
        ax.plot(t, readings[i], color=color, linewidth=0.7)
        ax.set_ylabel("kPa")
        ax.set_title(f"Sensor {i + 1}")
        ax.grid(True, alpha=0.4)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Pressure Sensor Array", fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


def plot_with_fault(save_path=None):
    t, pressure = generate_pressure_signal(duration_s=60)
    faulty = add_sensor_fault(pressure, t, fault_start_s=20, fault_end_s=30,
                              fault_type="stuck")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, pressure, color="steelblue",  linewidth=0.8, label="Healthy signal",  alpha=0.6)
    ax.plot(t, faulty,   color="crimson",    linewidth=0.8, label="Faulty signal (stuck)")
    ax.axvspan(20, 30, alpha=0.15, color="crimson", label="Fault window")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (kPa)")
    ax.set_title("Pressure Sensor — Stuck Fault Injection (20 s – 30 s)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


def animate_realtime(duration_s=30, sample_rate_hz=50, window_s=5):
    """
    Real-time scrolling pressure sensor animation.
    Streams samples one-by-one like a live sensor feed.
    """
    t_full, p_full = generate_pressure_signal(
        duration_s=duration_s, sample_rate_hz=sample_rate_hz
    )
    window_samples = int(window_s * sample_rate_hz)

    fig, ax = plt.subplots(figsize=(10, 4))
    line, = ax.plot([], [], color="steelblue", linewidth=1.0)
    ax.set_xlim(0, window_s)
    ax.set_ylim(p_full.min() - 2, p_full.max() + 2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (kPa)")
    ax.set_title("Live Pressure Sensor Feed")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    def update(frame):
        i = frame + 1
        start = max(0, i - window_samples)
        t_win = t_full[start:i] - t_full[start]
        p_win = p_full[start:i]
        line.set_data(t_win, p_win)
        return (line,)

    ani = animation.FuncAnimation(
        fig, update, frames=len(t_full),
        interval=1000 / sample_rate_hz, blit=True
    )
    plt.show()
    return ani


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pressure Sensor Simulator")
    parser.add_argument(
        "--mode",
        choices=["single", "array", "fault", "animate"],
        default="single",
        help=(
            "single  : single sensor with stats\n"
            "array   : multi-sensor array\n"
            "fault   : stuck-fault injection demo\n"
            "animate : scrolling real-time feed"
        ),
    )
    parser.add_argument("--save", type=str, default=None,
                        help="File path to save the plot (PNG)")
    parser.add_argument("--duration", type=float, default=60,
                        help="Signal duration in seconds")
    args = parser.parse_args()

    if args.mode == "single":
        t, p = generate_pressure_signal(duration_s=args.duration)
        plot_single_sensor(t, p, save_path=args.save)

    elif args.mode == "array":
        t, readings = generate_sensor_array(n_sensors=4, duration_s=args.duration)
        plot_sensor_array(t, readings, save_path=args.save)

    elif args.mode == "fault":
        plot_with_fault(save_path=args.save)

    elif args.mode == "animate":
        animate_realtime(duration_s=int(args.duration))
