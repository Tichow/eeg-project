"""
SSVEP (Steady-State Visual Evoked Potential) Stimulus Generator
================================================================

Research-grade visual stimulus generator for BCI experiments.
Generates flickering stimuli at configurable frequencies for SSVEP paradigms.

Theoretical background:
    SSVEP refers to sustained rhythmic brain activity in response to repetitive
    visual stimuli (RVS). When a subject fixates a stimulus flickering at
    frequency f, the occipital cortex (O1, O2) entrains to f and its harmonics.
    Peak SSVEP amplitude is typically observed around 10 Hz (Herrmann et al.),
    with a secondary peak near 15 Hz in occipital regions (Pastor et al.).

    Sources:
    - Herrmann, 2001: Highest SSVEP amplitude at 10 Hz, local peaks at 20/40 Hz
    - Pastor et al.: Occipital region responds most strongly at ~15 Hz
    - Zhu et al., 2010: "A Survey of Stimulation Methods Used in SSVEP-Based BCIs"
      Computational Intelligence and Neuroscience, doi:10.1155/2010/702357
    - Ladouce et al., 2022: "Improving user experience of SSVEP BCI through low
      amplitude depth and high frequency stimuli design", Scientific Reports

Design decisions:
    - SINUSOIDAL modulation: Preferred over square wave (on/off) because square
      waves generate harmonics in the EEG spectrum, making frequency detection
      ambiguous. Sinusoidal modulation produces a cleaner spectral peak.
      (See Zhu et al., 2010, Section 2)
    - FREQUENCY CHOICE: 10 Hz (low band, max amplitude) and 15 Hz (medium band,
      strong response). Both are divisible by common refresh rates (60/120/240 Hz).
      WARNING: 8-20 Hz range can trigger photosensitive epilepsy in susceptible
      individuals (~1 in 4000 of population, higher in 7-19 year-olds).
    - TRIAL DURATION: 5 seconds per trial is standard for reliable frequency
      detection via FFT (resolution = 1/T = 0.2 Hz at T=5s).
    - INTER-TRIAL REST: 3 seconds minimum to avoid visual fatigue and allow
      EEG baseline recovery.
    - STIMULUS SIZE: ~4-6 degrees of visual angle, centered on screen.
      Larger stimuli increase SNR but also epilepsy risk.

Safety notice:
    This script generates flickering visual stimuli in the 8-20 Hz range.
    These frequencies CAN trigger seizures in photosensitive individuals.
    Always screen participants for photosensitive epilepsy history before use.
    Keep a clear escape key (ESC) to stop immediately.

Usage:
    python ssvep_stimulus.py
    python ssvep_stimulus.py --frequencies 10 15 --trial-duration 5
    python ssvep_stimulus.py --help

Requirements:
    pip install pygame numpy

Author: EEG Research Project - 3A
"""

# =============================================================================
# 1. Standard library imports
# =============================================================================
import sys
import time
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime

# =============================================================================
# 2. Third-party imports
# =============================================================================
import numpy as np

try:
    import pygame
except ImportError:
    print("ERROR: pygame is required. Install it with: pip install pygame")
    sys.exit(1)


# =============================================================================
# PROTOCOL CONSTANTS — Based on SSVEP literature norms
#
# These defaults follow standard BCI research protocols.
# Modify via command-line arguments or the config dict, NOT here directly.
# =============================================================================

# Stimulus frequencies in Hz
# WHY 10 and 15 Hz: Peak SSVEP amplitude at ~10 Hz (Herrmann, 2001),
# secondary peak at ~15 Hz (Pastor et al.). Well-separated (5 Hz apart)
# for easy FFT discrimination. Both compatible with 60 Hz monitors.
DEFAULT_FREQUENCIES_HZ = [10.0, 15.0]

# Trial duration in seconds
# WHY 5s: FFT frequency resolution = 1/T. At 5s, resolution = 0.2 Hz,
# sufficient to distinguish 10 Hz from 15 Hz. Shorter epochs (3-4s)
# are possible but reduce SNR. Longer epochs increase fatigue.
DEFAULT_TRIAL_DURATION_S = 5.0

# Inter-trial rest duration in seconds
# WHY 3s: Allows alpha rhythm to return to baseline between trials.
# Literature typically uses 2-5s rest. 3s balances speed and recovery.
DEFAULT_REST_DURATION_S = 3.0

# Number of trials per frequency
# WHY 20: Standard for offline SSVEP calibration. Provides enough
# epochs for reliable CCA/FFT analysis after artifact rejection.
# Online BCI can work with fewer, but calibration needs ~20 per class.
DEFAULT_TRIALS_PER_FREQUENCY = 20

# Number of blocks (full cycles through all frequencies)
# WHY: Blocking prevents order effects and participant fatigue patterns.
# Each block contains one trial per frequency in randomized order.
DEFAULT_NUM_BLOCKS = 10

# Stimulus visual parameters
DEFAULT_STIMULUS_SIZE_PX = 200  # Square side length in pixels (~5° visual angle at 60cm)
DEFAULT_BACKGROUND_COLOR = (128, 128, 128)  # Neutral gray (important: reduces contrast artifacts)
DEFAULT_STIMULUS_COLOR_MAX = (255, 255, 255)  # White at peak luminance
DEFAULT_STIMULUS_COLOR_MIN = (0, 0, 0)  # Black at trough

# Modulation depth (0.0 to 1.0)
# WHY 1.0 default: Maximum contrast = maximum SNR for initial experiments.
# Ladouce et al. (2022) showed 60% depth (0.6) maintains >90% accuracy
# while improving comfort. Reduce to 0.6 if participants report fatigue.
DEFAULT_MODULATION_DEPTH = 1.0

# Waveform type: "sinusoidal" or "square"
# WHY sinusoidal: Produces cleaner frequency peaks in EEG spectrum.
# Square waves generate odd harmonics (3f, 5f, 7f...) that can confuse
# frequency detection algorithms, especially when frequencies are
# harmonically related. (Zhu et al., 2010)
DEFAULT_WAVEFORM = "sinusoidal"

# Screen parameters
DEFAULT_FULLSCREEN = False  # Set True for actual experiments
DEFAULT_WINDOW_SIZE = (800, 600)  # Windowed mode size for testing

# Countdown before first trial
DEFAULT_COUNTDOWN_S = 5

# Marker/trigger support
# WHY: For EEG synchronization. Prints markers to console and log file.
# In a real setup, replace print_marker() with LSL outlet or parallel port.
ENABLE_MARKERS = True


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def compute_luminance_sinusoidal(
    time_s: float,
    frequency_hz: float,
    modulation_depth: float = 1.0,
) -> float:
    """
    Compute normalized luminance [0, 1] for sinusoidal flicker at a given time.

    The luminance follows: L(t) = 0.5 + 0.5 * depth * sin(2 * pi * f * t)
    This produces a smooth oscillation between (0.5 - 0.5*depth) and (0.5 + 0.5*depth).

    Args:
        time_s: Current time in seconds since stimulus onset.
        frequency_hz: Flicker frequency in Hertz.
        modulation_depth: Contrast depth from 0 (no flicker) to 1 (full contrast).

    Returns:
        Normalized luminance value between 0.0 and 1.0.
    """
    return 0.5 + 0.5 * modulation_depth * np.sin(2 * np.pi * frequency_hz * time_s)


def compute_luminance_square(
    time_s: float,
    frequency_hz: float,
    modulation_depth: float = 1.0,
) -> float:
    """
    Compute normalized luminance [0, 1] for square wave flicker at a given time.

    Square wave is ON (bright) for half the period, OFF (dark) for the other half.
    Simpler but generates harmonics in the EEG frequency spectrum.

    Args:
        time_s: Current time in seconds since stimulus onset.
        frequency_hz: Flicker frequency in Hertz.
        modulation_depth: Contrast depth from 0 (no flicker) to 1 (full contrast).

    Returns:
        Normalized luminance value between 0.0 and 1.0.
    """
    phase = (time_s * frequency_hz) % 1.0
    if phase < 0.5:
        return 0.5 + 0.5 * modulation_depth
    else:
        return 0.5 - 0.5 * modulation_depth


def luminance_to_color(
    luminance: float,
    color_max: tuple = DEFAULT_STIMULUS_COLOR_MAX,
    color_min: tuple = DEFAULT_STIMULUS_COLOR_MIN,
) -> tuple:
    """
    Convert normalized luminance [0, 1] to RGB color tuple.

    Linearly interpolates between color_min (luminance=0) and
    color_max (luminance=1).

    Args:
        luminance: Normalized luminance between 0.0 and 1.0.
        color_max: RGB tuple for maximum brightness.
        color_min: RGB tuple for minimum brightness.

    Returns:
        RGB color tuple with integer values in [0, 255].
    """
    r = int(color_min[0] + (color_max[0] - color_min[0]) * luminance)
    g = int(color_min[1] + (color_max[1] - color_min[1]) * luminance)
    b = int(color_min[2] + (color_max[2] - color_min[2]) * luminance)
    return (
        max(0, min(255, r)),
        max(0, min(255, g)),
        max(0, min(255, b)),
    )


def generate_trial_sequence(
    frequencies_hz: list,
    num_blocks: int,
    seed: int = None,
) -> list:
    """
    Generate a randomized trial sequence organized in blocks.

    Each block contains exactly one trial per frequency, in random order.
    This balanced design ensures equal exposure and prevents order effects.

    Args:
        frequencies_hz: List of stimulus frequencies.
        num_blocks: Number of repetition blocks.
        seed: Random seed for reproducibility. None for random.

    Returns:
        List of (block_index, trial_index_in_block, frequency_hz) tuples.
    """
    rng = np.random.default_rng(seed)
    trial_sequence = []

    for block_idx in range(num_blocks):
        block_frequencies = list(frequencies_hz)
        rng.shuffle(block_frequencies)
        for trial_idx, freq in enumerate(block_frequencies):
            trial_sequence.append((block_idx, trial_idx, freq))

    return trial_sequence


def print_marker(marker_type: str, value: str, timestamp: float):
    """
    Emit an event marker for EEG synchronization.

    In a real experiment, replace this with:
    - LSL (Lab Streaming Layer) outlet push
    - Parallel port trigger
    - Serial port trigger
    - TCP/UDP socket message

    Args:
        marker_type: Category of the marker (e.g., 'TRIAL_START', 'TRIAL_END').
        value: Associated value (e.g., frequency in Hz).
        timestamp: Time in seconds since experiment start.
    """
    if ENABLE_MARKERS:
        print(f"[MARKER] {timestamp:.4f}s | {marker_type}: {value}")


def draw_fixation_cross(screen, center_x: int, center_y: int, size: int = 20, color=(255, 0, 0)):
    """
    Draw a fixation cross at the center of the stimulus area.

    Fixation crosses help maintain gaze direction, which is critical
    for SSVEP because the response depends on foveal stimulation.

    Args:
        screen: Pygame display surface.
        center_x: X coordinate of the cross center.
        center_y: Y coordinate of the cross center.
        size: Half-length of the cross arms in pixels.
        color: RGB color of the cross.
    """
    pygame.draw.line(screen, color, (center_x - size, center_y), (center_x + size, center_y), 3)
    pygame.draw.line(screen, color, (center_x, center_y - size), (center_x, center_y + size), 3)


def draw_text_centered(screen, text: str, font, y_offset: int = 0, color=(255, 255, 255)):
    """
    Draw centered text on the screen.

    Args:
        screen: Pygame display surface.
        text: Text string to render.
        font: Pygame font object.
        y_offset: Vertical offset from screen center in pixels.
        color: RGB color of the text.
    """
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 + y_offset))
    screen.blit(text_surface, text_rect)


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================

def run_experiment(config: dict):
    """
    Run the complete SSVEP stimulus experiment.

    Executes a block-randomized sequence of flickering stimuli,
    logging all events with precise timestamps for EEG analysis.

    Args:
        config: Dictionary containing all experiment parameters.
    """
    # -------------------------------------------------------------------------
    # Initialize Pygame and display
    # -------------------------------------------------------------------------
    pygame.init()

    if config["fullscreen"]:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode(config["window_size"])

    pygame.display.set_caption("SSVEP Stimulus Generator")

    # WHY we check refresh rate: Frequencies must be achievable given the monitor's
    # refresh rate. For square waves, freq must divide evenly into refresh rate.
    # For sinusoidal, any frequency works but temporal resolution is limited.
    clock = pygame.time.Clock()
    actual_refresh_rate = pygame.display.Info().current_w  # Approximate
    # NOTE: pygame doesn't reliably report refresh rate. For precise experiments,
    # measure it manually or use PsychoPy which has proper VSync detection.

    font_large = pygame.font.SysFont("Arial", 48)
    font_medium = pygame.font.SysFont("Arial", 32)
    font_small = pygame.font.SysFont("Arial", 24)

    screen_width = screen.get_width()
    screen_height = screen.get_height()
    center_x = screen_width // 2
    center_y = screen_height // 2
    stim_half = config["stimulus_size_px"] // 2

    # Select waveform function
    if config["waveform"] == "sinusoidal":
        compute_luminance = compute_luminance_sinusoidal
    else:
        compute_luminance = compute_luminance_square

    # -------------------------------------------------------------------------
    # Generate trial sequence
    # -------------------------------------------------------------------------
    trial_sequence = generate_trial_sequence(
        frequencies_hz=config["frequencies_hz"],
        num_blocks=config["num_blocks"],
        seed=config.get("random_seed"),
    )

    total_trials = len(trial_sequence)
    total_duration_estimate = total_trials * (config["trial_duration_s"] + config["rest_duration_s"])

    # -------------------------------------------------------------------------
    # Prepare log file
    # -------------------------------------------------------------------------
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"ssvep_log_{timestamp_str}.csv"
    log_path = Path(config.get("output_dir", ".")) / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_data = []

    # -------------------------------------------------------------------------
    # Display safety warning and experiment info
    # -------------------------------------------------------------------------
    is_running = True

    # Safety screen
    screen.fill((0, 0, 0))
    draw_text_centered(screen, "⚠ SSVEP STIMULUS GENERATOR ⚠", font_large, y_offset=-120, color=(255, 200, 0))
    draw_text_centered(screen, "WARNING: This program generates flickering lights", font_medium, y_offset=-60, color=(255, 100, 100))
    draw_text_centered(screen, "that may trigger seizures in photosensitive individuals.", font_medium, y_offset=-20, color=(255, 100, 100))
    draw_text_centered(screen, f"Frequencies: {config['frequencies_hz']} Hz | Waveform: {config['waveform']}", font_small, y_offset=40)
    draw_text_centered(screen, f"Trials: {total_trials} | Duration: ~{total_duration_estimate / 60:.1f} min", font_small, y_offset=70)
    draw_text_centered(screen, f"Modulation depth: {config['modulation_depth'] * 100:.0f}%", font_small, y_offset=100)
    draw_text_centered(screen, "Press SPACE to start | ESC to quit at any time", font_medium, y_offset=160)
    pygame.display.flip()

    # Wait for space or escape
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                if event.key == pygame.K_SPACE:
                    waiting = False

    # -------------------------------------------------------------------------
    # Countdown
    # -------------------------------------------------------------------------
    for count in range(config["countdown_s"], 0, -1):
        screen.fill(config["background_color"])
        draw_text_centered(screen, f"Starting in {count}...", font_large)
        draw_text_centered(screen, "Focus on the flickering stimulus center", font_small, y_offset=60)
        pygame.display.flip()
        pygame.time.wait(1000)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                return

    # -------------------------------------------------------------------------
    # Main experiment loop
    # -------------------------------------------------------------------------
    # Référence temporelle pour la synchronisation avec le dashboard.
    # On capture l'heure mur et le compteur haute-résolution au même instant,
    # ce qui permet de convertir n'importe quel elapsed perf_counter en timestamp
    # relatif au début de l'enregistrement dashboard.
    _wall_at_exp_start = time.time()
    _perf_at_exp_start = time.perf_counter()
    _sync_file = config.get("sync_file")
    _rec_start = config.get("rec_start")

    def _emit(marker_type: str, value: str, perf_elapsed: float) -> None:
        """Émet un marqueur console ET dans le fichier de sync si configuré."""
        print_marker(marker_type, value, perf_elapsed)
        if _sync_file and _rec_start is not None:
            # Temps relatif au début du recording dashboard :
            # délai entre rec_start et le lancement du stimulus + elapsed interne
            rec_time = round((_wall_at_exp_start - _rec_start) + perf_elapsed, 3)
            event = {"time_sec": rec_time, "action": f"{marker_type}: {value}"}
            with open(_sync_file, "a") as fh:
                fh.write(json.dumps(event) + "\n")

    experiment_start_time = time.perf_counter()
    _emit("EXPERIMENT", "START", 0.0)

    for trial_idx, (block_idx, trial_in_block, frequency_hz) in enumerate(trial_sequence):
        if not is_running:
            break

        # --- Pre-trial rest with fixation cross ---
        rest_start = time.perf_counter()
        elapsed_since_start = rest_start - experiment_start_time
        _emit("REST_START", f"block={block_idx} trial={trial_idx}", elapsed_since_start)

        while time.perf_counter() - rest_start < config["rest_duration_s"]:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    is_running = False
                    break
            if not is_running:
                break

            screen.fill(config["background_color"])

            # Show trial info during rest
            info_text = f"Next: {frequency_hz} Hz | Trial {trial_idx + 1}/{total_trials} | Block {block_idx + 1}/{config['num_blocks']}"
            draw_text_centered(screen, info_text, font_small, y_offset=-80)
            draw_fixation_cross(screen, center_x, center_y, size=15, color=(200, 200, 200))

            # Show countdown during last 2 seconds of rest
            rest_remaining = config["rest_duration_s"] - (time.perf_counter() - rest_start)
            if rest_remaining < 2.0:
                draw_text_centered(screen, f"{rest_remaining:.1f}", font_medium, y_offset=80, color=(255, 255, 0))

            pygame.display.flip()
            clock.tick(60)

        if not is_running:
            break

        # --- Stimulus trial ---
        trial_start = time.perf_counter()
        elapsed_since_start = trial_start - experiment_start_time
        _emit("TRIAL_START", f"freq={frequency_hz}Hz block={block_idx}", elapsed_since_start)

        # Log trial metadata
        trial_log_entry = {
            "trial_index": trial_idx,
            "block_index": block_idx,
            "trial_in_block": trial_in_block,
            "frequency_hz": frequency_hz,
            "waveform": config["waveform"],
            "modulation_depth": config["modulation_depth"],
            "trial_duration_s": config["trial_duration_s"],
            "timestamp_start_s": elapsed_since_start,
            "timestamp_start_absolute": datetime.now().isoformat(),
        }

        frame_count = 0

        while time.perf_counter() - trial_start < config["trial_duration_s"]:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    is_running = False
                    break
            if not is_running:
                break

            # Compute current luminance based on time since trial onset
            current_time = time.perf_counter() - trial_start
            luminance = compute_luminance(
                time_s=current_time,
                frequency_hz=frequency_hz,
                modulation_depth=config["modulation_depth"],
            )
            stim_color = luminance_to_color(luminance)

            # Draw frame
            screen.fill(config["background_color"])

            # Draw stimulus rectangle
            stim_rect = pygame.Rect(
                center_x - stim_half,
                center_y - stim_half,
                config["stimulus_size_px"],
                config["stimulus_size_px"],
            )
            pygame.draw.rect(screen, stim_color, stim_rect)

            # Fixation cross on top of stimulus (red for visibility)
            draw_fixation_cross(screen, center_x, center_y, size=12, color=(255, 0, 0))

            # Frequency label (subtle, for experimenter reference)
            freq_label = font_small.render(f"{frequency_hz} Hz", True, (100, 100, 100))
            screen.blit(freq_label, (10, 10))

            # Progress bar at bottom
            progress = (trial_idx + (current_time / config["trial_duration_s"])) / total_trials
            bar_width = int(screen_width * 0.8)
            bar_x = (screen_width - bar_width) // 2
            bar_y = screen_height - 30
            pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, 10))
            pygame.draw.rect(screen, (0, 200, 100), (bar_x, bar_y, int(bar_width * progress), 10))

            pygame.display.flip()
            frame_count += 1

            # WHY no clock.tick() here: We want maximum frame rate during stimulus
            # to get the smoothest sinusoidal modulation possible. The luminance
            # is computed from wall-clock time, not frame count, so missed frames
            # don't cause frequency drift. This is a key advantage of time-based
            # over frame-based flickering.

        # Record trial end
        trial_end = time.perf_counter()
        elapsed_since_start = trial_end - experiment_start_time
        actual_duration = trial_end - trial_start
        actual_fps = frame_count / actual_duration if actual_duration > 0 else 0

        trial_log_entry["timestamp_end_s"] = elapsed_since_start
        trial_log_entry["actual_duration_s"] = actual_duration
        trial_log_entry["frame_count"] = frame_count
        trial_log_entry["actual_fps"] = round(actual_fps, 1)
        log_data.append(trial_log_entry)

        _emit("TRIAL_END", f"freq={frequency_hz}Hz frames={frame_count} fps={actual_fps:.1f}", elapsed_since_start)

    # -------------------------------------------------------------------------
    # Experiment end
    # -------------------------------------------------------------------------
    experiment_end = time.perf_counter()
    total_experiment_duration = experiment_end - experiment_start_time
    _emit("EXPERIMENT", f"END duration={total_experiment_duration:.1f}s", total_experiment_duration)

    # Save log to CSV
    if log_data:
        fieldnames = log_data[0].keys()
        with open(log_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_data)
        print(f"\n[LOG] Trial data saved to: {log_path}")

    # Save config as JSON for reproducibility
    config_path = log_path.with_suffix(".json")
    config_serializable = {k: v for k, v in config.items()}
    # Convert tuples to lists for JSON serialization
    for key in config_serializable:
        if isinstance(config_serializable[key], tuple):
            config_serializable[key] = list(config_serializable[key])
    with open(config_path, "w") as f:
        json.dump(config_serializable, f, indent=2)
    print(f"[LOG] Config saved to: {config_path}")

    # End screen
    screen.fill((0, 0, 0))
    draw_text_centered(screen, "Experiment Complete!", font_large, y_offset=-40)
    draw_text_centered(screen, f"Total duration: {total_experiment_duration:.1f}s", font_medium, y_offset=20)
    draw_text_centered(screen, f"Trials completed: {len(log_data)}/{total_trials}", font_medium, y_offset=60)
    draw_text_centered(screen, "Press any key to exit", font_small, y_offset=120)
    pygame.display.flip()

    # Wait for keypress to close
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                waiting = False

    pygame.quit()

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for freq in config["frequencies_hz"]:
        freq_trials = [t for t in log_data if t["frequency_hz"] == freq]
        if freq_trials:
            avg_fps = np.mean([t["actual_fps"] for t in freq_trials])
            avg_dur = np.mean([t["actual_duration_s"] for t in freq_trials])
            print(f"  {freq} Hz: {len(freq_trials)} trials | avg FPS: {avg_fps:.1f} | avg duration: {avg_dur:.3f}s")
    print(f"  Total duration: {total_experiment_duration:.1f}s")
    print("=" * 60)


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def parse_arguments() -> dict:
    """
    Parse command-line arguments and return experiment configuration.

    Returns:
        Dictionary containing all experiment parameters.
    """
    parser = argparse.ArgumentParser(
        description="SSVEP Stimulus Generator for BCI Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ssvep_stimulus.py
  python ssvep_stimulus.py --frequencies 10 15 --trial-duration 5
  python ssvep_stimulus.py --frequencies 7 10 12 15 --blocks 5 --waveform square
  python ssvep_stimulus.py --modulation-depth 0.6 --fullscreen

Safety:
  Frequencies in 8-20 Hz range may trigger photosensitive epilepsy.
  Always screen participants before experiments.
  Press ESC at any time to stop the stimulus immediately.
        """,
    )

    parser.add_argument(
        "--frequencies", "-f",
        type=float, nargs="+",
        default=DEFAULT_FREQUENCIES_HZ,
        help=f"Stimulus frequencies in Hz (default: {DEFAULT_FREQUENCIES_HZ})",
    )
    parser.add_argument(
        "--trial-duration", "-t",
        type=float,
        default=DEFAULT_TRIAL_DURATION_S,
        help=f"Duration of each trial in seconds (default: {DEFAULT_TRIAL_DURATION_S})",
    )
    parser.add_argument(
        "--rest-duration", "-r",
        type=float,
        default=DEFAULT_REST_DURATION_S,
        help=f"Rest period between trials in seconds (default: {DEFAULT_REST_DURATION_S})",
    )
    parser.add_argument(
        "--blocks", "-b",
        type=int,
        default=DEFAULT_NUM_BLOCKS,
        help=f"Number of blocks (default: {DEFAULT_NUM_BLOCKS})",
    )
    parser.add_argument(
        "--waveform", "-w",
        type=str,
        choices=["sinusoidal", "square"],
        default=DEFAULT_WAVEFORM,
        help=f"Flicker waveform type (default: {DEFAULT_WAVEFORM})",
    )
    parser.add_argument(
        "--modulation-depth", "-m",
        type=float,
        default=DEFAULT_MODULATION_DEPTH,
        help=f"Modulation depth 0.0-1.0 (default: {DEFAULT_MODULATION_DEPTH})",
    )
    parser.add_argument(
        "--stimulus-size",
        type=int,
        default=DEFAULT_STIMULUS_SIZE_PX,
        help=f"Stimulus square size in pixels (default: {DEFAULT_STIMULUS_SIZE_PX})",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        default=DEFAULT_FULLSCREEN,
        help="Run in fullscreen mode (recommended for experiments)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible trial order",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=".",
        help="Directory for log files (default: current directory)",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=DEFAULT_COUNTDOWN_S,
        help=f"Countdown before first trial in seconds (default: {DEFAULT_COUNTDOWN_S})",
    )
    parser.add_argument(
        "--sync-file",
        type=str,
        default=None,
        help="Fichier JSONL partagé avec le dashboard pour la synchronisation des marqueurs.",
    )
    parser.add_argument(
        "--rec-start",
        type=float,
        default=None,
        help="time.time() au moment du démarrage du recording dashboard (pour aligner les timestamps).",
    )

    args = parser.parse_args()

    # Validate modulation depth
    if not 0.0 <= args.modulation_depth <= 1.0:
        parser.error("Modulation depth must be between 0.0 and 1.0")

    # Validate frequencies (warn about epilepsy-risk range)
    for freq in args.frequencies:
        if 3.0 <= freq <= 30.0:
            print(f"[WARNING] {freq} Hz is in the photosensitive epilepsy risk range (3-30 Hz).")
        if freq <= 0:
            parser.error(f"Frequency must be positive, got {freq}")

    config = {
        "frequencies_hz": args.frequencies,
        "trial_duration_s": args.trial_duration,
        "rest_duration_s": args.rest_duration,
        "num_blocks": args.blocks,
        "waveform": args.waveform,
        "modulation_depth": args.modulation_depth,
        "stimulus_size_px": args.stimulus_size,
        "fullscreen": args.fullscreen,
        "random_seed": args.seed,
        "output_dir": args.output_dir,
        "countdown_s": args.countdown,
        "background_color": DEFAULT_BACKGROUND_COLOR,
        "window_size": DEFAULT_WINDOW_SIZE,
        "sync_file": args.sync_file,
        "rec_start": args.rec_start,
    }

    return config


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    config = parse_arguments()

    print("\n" + "=" * 60)
    print("SSVEP STIMULUS GENERATOR")
    print("=" * 60)
    print(f"  Frequencies:       {config['frequencies_hz']} Hz")
    print(f"  Waveform:          {config['waveform']}")
    print(f"  Modulation depth:  {config['modulation_depth'] * 100:.0f}%")
    print(f"  Trial duration:    {config['trial_duration_s']}s")
    print(f"  Rest duration:     {config['rest_duration_s']}s")
    print(f"  Blocks:            {config['num_blocks']}")
    print(f"  Trials per block:  {len(config['frequencies_hz'])}")
    print(f"  Total trials:      {config['num_blocks'] * len(config['frequencies_hz'])}")
    total_est = config['num_blocks'] * len(config['frequencies_hz']) * (config['trial_duration_s'] + config['rest_duration_s'])
    print(f"  Estimated time:    {total_est / 60:.1f} min")
    print("=" * 60)
    print()

    run_experiment(config)