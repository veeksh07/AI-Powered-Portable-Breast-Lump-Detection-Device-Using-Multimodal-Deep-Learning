"""
Thermal Image Simulator
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter


def gaussian_blob(shape, center, sigma, amplitude):
    """Create a 2D Gaussian heat source."""
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = center
    blob = amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return blob


def generate_thermal_frame(shape=(256, 256), heat_sources=None, ambient_temp=20.0,
                            noise_std=0.5, blur_sigma=2.0):
    """
    Generate a single synthetic thermal image (temperature map in Celsius).

    Parameters
    ----------
    shape        : (H, W) image size in pixels
    heat_sources : list of dicts with keys: center (y,x), sigma, amplitude (°C above ambient)
    ambient_temp : background temperature in °C
    noise_std    : standard deviation of thermal sensor noise (°C)
    blur_sigma   : Gaussian blur to simulate sensor PSF / heat diffusion

    Returns
    -------
    temp_map : float32 array of shape (H, W) with temperatures in °C
    """
    temp_map = np.full(shape, ambient_temp, dtype=np.float32)

    if heat_sources is None:
        # Default: a few random-ish heat sources
        heat_sources = [
            {"center": (80, 100), "sigma": 30, "amplitude": 40},
            {"center": (180, 160), "sigma": 20, "amplitude": 25},
            {"center": (120, 200), "sigma": 15, "amplitude": 15},
        ]

    for src in heat_sources:
        temp_map += gaussian_blob(shape, src["center"], src["sigma"], src["amplitude"])

    # Sensor noise
    temp_map += np.random.normal(0, noise_std, shape).astype(np.float32)

    # Blur (heat diffusion / optics)
    temp_map = gaussian_filter(temp_map, sigma=blur_sigma).astype(np.float32)

    return temp_map


def render_thermal(temp_map, colormap="inferno", vmin=None, vmax=None, ax=None,
                   show_colorbar=True, title="Thermal Image"):
    """Render a temperature map as a false-color thermal image."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    im = ax.imshow(temp_map, cmap=colormap,
                   vmin=vmin if vmin is not None else temp_map.min(),
                   vmax=vmax if vmax is not None else temp_map.max(),
                   origin="upper")

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Temperature (°C)")

    ax.set_title(title)
    ax.axis("off")
    return fig, ax, im


def simulate_static(save_path=None):
    """Generate and display a single thermal frame."""
    temp_map = generate_thermal_frame()
    fig, _, _ = render_thermal(temp_map, colormap="inferno",
                               title="Simulated Thermal Image")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


def simulate_animation(n_frames=60, interval_ms=100, save_path=None):
    """
    Animate thermal image with moving/pulsing heat sources.

    Parameters
    ----------
    n_frames    : number of animation frames
    interval_ms : delay between frames in milliseconds
    save_path   : if given (e.g. 'thermal.gif'), save the animation
    """
    shape = (256, 256)
    ambient = 20.0

    fig, ax = plt.subplots(figsize=(6, 5))

    # Initial frame
    temp0 = generate_thermal_frame(shape, ambient_temp=ambient)
    vmin, vmax = ambient, ambient + 50  # fixed color scale

    im = ax.imshow(temp0, cmap="inferno", vmin=vmin, vmax=vmax, origin="upper")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (°C)")
    ax.set_title("Thermal Simulation (animated)")
    ax.axis("off")
    plt.tight_layout()

    def update(frame):
        t = frame / n_frames  # 0 → 1

        # Source 1: orbits in a circle
        angle1 = 2 * np.pi * t
        cy1 = int(128 + 60 * np.sin(angle1))
        cx1 = int(128 + 60 * np.cos(angle1))

        # Source 2: pulsing amplitude
        amp2 = 20 + 15 * np.sin(2 * np.pi * t * 2)

        heat_sources = [
            {"center": (cy1, cx1), "sigma": 25, "amplitude": 40},
            {"center": (180, 80),  "sigma": 20, "amplitude": amp2},
            {"center": (60, 200),  "sigma": 15, "amplitude": 15},
        ]
        temp = generate_thermal_frame(shape, heat_sources=heat_sources,
                                      ambient_temp=ambient)
        im.set_data(temp)
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=interval_ms, blit=True)

    if save_path:
        ani.save(save_path, writer="pillow", fps=1000 // interval_ms)
        print(f"Saved animation to {save_path}")

    plt.show()
    return ani


def multi_colormap_comparison():
    """Show the same thermal frame rendered with several common IR colormaps."""
    temp_map = generate_thermal_frame()
    colormaps = ["inferno", "hot", "plasma", "RdYlBu_r", "Spectral_r"]

    fig, axes = plt.subplots(1, len(colormaps), figsize=(4 * len(colormaps), 4))
    vmin, vmax = temp_map.min(), temp_map.max()

    for ax, cmap in zip(axes, colormaps):
        im = ax.imshow(temp_map, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(cmap)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Thermal Image — Colormap Comparison", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Thermal Image Simulator")
    parser.add_argument("--mode", choices=["static", "animate", "compare"],
                        default="static",
                        help="static: single frame | animate: animated | compare: colormaps")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save output (PNG for static, GIF for animate)")
    parser.add_argument("--frames", type=int, default=60,
                        help="Number of animation frames (animate mode only)")
    args = parser.parse_args()

    if args.mode == "static":
        simulate_static(save_path=args.save)
    elif args.mode == "animate":
        simulate_animation(n_frames=args.frames, save_path=args.save)
    elif args.mode == "compare":
        multi_colormap_comparison()
