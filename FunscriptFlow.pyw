#!/usr/bin/env python3

import gc
import os, math, threading, concurrent.futures, json, argparse
import numpy as np
import cv2
from decord import VideoReader, cpu
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk
from multiprocessing import Pool


# ---------- Localization Strings ----------
def load_strings(filename="strings.json"):
    defaults = {
        "app_title": "Funscript Flow",
        "select_videos": "Select Videos",
        "select_folder": "Select Folder",
        "no_files_selected": "No files selected",
        "vr_mode": "VR Mode",
        "vr_mode_tooltip": ("Use this to improve accuracy for VR videos."),
        "overall_progress": "Overall Progress:",
        "current_video_progress": "Current Video Progress:",
        "advanced_settings": "Advanced Settings",
        "threads": "Threads:",
        "detrend_window": "Detrend window (sec):",
        
        "norm_window": "Norm window (sec):",
        "batch_size": "Batch size (frames):",
        "face_inversion": "Enable face‑based inversion",
        "show_preview": "Show Preview",
        "show_advanced": "Show Advanced Settings",
        "overwrite_files": "Overwrite existing files",
        "run": "Run",
        "cancel": "Cancel",
        "readme": "Readme",
        "config_saved": "Config saved to {config_path}",
        "config_load_error": "Error loading config: {error}",
        "no_files_warning": "Please select one or more video files or a folder.",
        "cancelled_by_user": "Processing cancelled by user.",
        "batch_processing_complete": "Batch processing complete.",
        "funscript_saved": "Funscript saved: {output_path}",
        "skipping_file_exists": "Skipping {video_path}: {output_path} exists.",
        "log_error": "ERROR: Could not write output: {error}",
        "found_files": "Found {n} file(s).",
        "processing_file": "--- Processing file {current}/{total}: {video_path} ---",
        "processing_completed_with_errors": "Processing completed with errors. See run.log for details.",
        "face_inversion_tooltip": "Uses face detection to try to determine the angle of motion, and adjust direction accordingly.",
        "pov_mode_tooltip": "Use this to improve stability for POV videos.",
    }
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return defaults

STRINGS = load_strings()

# ---------- Tooltip Implementation ----------
class ToolTip:
    """Simple tooltip for a widget."""
    def __init__(self, widget, text="widget info"):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.enter)
        widget.bind("<Leave>", self.leave)
    def enter(self, event=None):
        self.showtip()
    def leave(self, event=None):
        self.hidetip()
    def showtip(self):
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + cy + self.widget.winfo_rooty() + 25
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)
    def hidetip(self):
        if self.tipwindow:
            self.tipwindow.destroy()
        self.tipwindow = None

def detect_cut(pair, log_func=None, threshold=30):
    return False
    prev_frame, curr_frame = pair
    diff = np.mean(np.abs(curr_frame.astype(np.float32) - prev_frame.astype(np.float32)))
    if(log_func != None and diff > threshold):
        log_func(f"Found a cut at " + str(diff))
        
    return diff > threshold


# ---------- Original flow functions (for VR mode) ------------
def compute_flow(pair):
    """
    (VR Mode) Process a pair of consecutive 512x512 grayscale frames (cropped from left half).
    Computes optical flow on two regions (middle-center and bottom-center of a 3x3 grid)
    and returns a tuple (avg_flow_middle, avg_flow_bottom).
    """
    prev_frame, curr_frame = pair
    h, w = prev_frame.shape  # expected 512x512
    cell_h = h // 3
    cell_w = w // 3
    prev_middle = prev_frame[cell_h:2*cell_h, cell_w:2*cell_w]
    curr_middle = curr_frame[cell_h:2*cell_h, cell_w:2*cell_w]
    prev_bottom = prev_frame[2*cell_h:3*cell_h, cell_w:2*cell_w]
    curr_bottom = curr_frame[2*cell_h:3*cell_h, cell_w:2*cell_w]
    flow_middle = cv2.calcOpticalFlowFarneback(prev_middle, curr_middle, None,
                                               pyr_scale=0.5, levels=3, winsize=15,
                                               iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    flow_bottom = cv2.calcOpticalFlowFarneback(prev_bottom, curr_bottom, None,
                                               pyr_scale=0.5, levels=3, winsize=15,
                                               iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    avg_middle = np.mean(flow_middle[..., 1])
    avg_bottom = np.mean(flow_bottom[..., 1])
    return avg_middle, avg_bottom

def compute_flow_nonvr_invert(pair):
    prev_frame, curr_frame = pair
    flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    avg_flow = np.mean(-flow[..., 0] + flow[..., 1])
    return avg_flow

def compute_flow_nonvr(pair):
    prev_frame, curr_frame = pair
    flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    avg_flow = np.mean(flow[..., 0] + flow[..., 1])
    return avg_flow

# --- PSO mode ---

def center_of_mass_variance(flow, num_cells=32):
    """
    Splits the optical flow into a configurable grid (num_cells x num_cells), 
    computes the variance of the optical flow in each grid cell, 
    and returns the center of mass of the variance.
    """
    h, w, _ = flow.shape
    grid_h, grid_w = h // num_cells, w // num_cells

    variance_grid = np.zeros((num_cells, num_cells))
    y_coords, x_coords = np.meshgrid(np.arange(num_cells), np.arange(num_cells), indexing='ij')

    for i in range(num_cells):
        for j in range(num_cells):
            cell = flow[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
            magnitude = np.sqrt(cell[..., 0]**2 + cell[..., 1]**2)
            variance_grid[i, j] = np.var(magnitude)

    total_variance = np.sum(variance_grid)

    if total_variance == 0:
        return (w // 2, h // 2)  # Default to image center if no variance
    else:
        center_x = np.sum(x_coords * variance_grid) * grid_w / total_variance + grid_w / 2
        center_y = np.sum(y_coords * variance_grid) * grid_h / total_variance + grid_h / 2
        return (center_x, center_y)

def max_divergence(flow):
    """
    Computes the divergence of the optical flow over the whole image and returns
    the pixel (x, y) with the highest absolute divergence along with its value.
    """
    # No grid, just pure per-pixel divergence!
    div = np.gradient(flow[..., 0], axis=0) + np.gradient(flow[..., 1], axis=1)
    
    # Get the index (y, x) of the max abs divergence
    y, x = np.unravel_index(np.argmax(np.abs(div)), div.shape)
    return x, y, div[y, x]


def radial_motion_weighted(flow, center, is_cut, pov_mode=False):
    """
    Computes signed radial motion: positive for outward motion, negative for inward motion.
    Closer pixels have higher weight.
    """
    if(is_cut):
        return 0.0
    h, w, _ = flow.shape
    y, x = np.indices((h, w))
    dx = x - center[0]
    dy = y - center[1]

    dot = flow[..., 0] * dx + flow[..., 1] * dy

    # In POV mode, just return the mean dot product
    if(pov_mode):
        return np.mean(dot)
    
    #Cancel out global motion by balancing the averages
    # multiply products to the right of the center (w-x) / w and to the left by x / w
    weighted_dot = np.where(x > center[0], dot * (w - x) / w, dot * x / w)
    # multiply products below the center (h-y) / h and above by y / h
    weighted_dot = np.where(y > center[1], weighted_dot * (h - y) / h, weighted_dot * y / h)

    return np.mean(weighted_dot)



def largest_cluster_center(positions, threshold=10.0):
    """
    BFS to find the largest cluster of swarm positions, return its centroid.
    """
    num_particles = len(positions)
    adj = [[] for _ in range(num_particles)]
    for i in range(num_particles):
        for j in range(i+1, num_particles):
            if np.linalg.norm(positions[i] - positions[j]) < threshold:
                adj[i].append(j)
                adj[j].append(i)

    visited = set()
    clusters = []
    def bfs(start):
        queue, c = [start], []
        while queue:
            node = queue.pop()
            if node in visited: continue
            visited.add(node)
            c.append(node)
            for nei in adj[node]:
                if nei not in visited:
                    queue.append(nei)
        return c

    for i in range(num_particles):
        if i not in visited:
            group = bfs(i)
            clusters.append(group)

    biggest = max(clusters, key=len)
    return (np.mean(positions[biggest], axis=0), len(biggest))

def swarm_positions(flow, num_particles=30, iterations=50):
    """
    Moves 'num_particles' along 'flow' for 'iterations'. Return final positions for clustering.
    """
    h, w, _ = flow.shape
    positions = np.column_stack([
        np.random.uniform(0, w, num_particles),
        np.random.uniform(0, h, num_particles)
    ])
    for _ in range(iterations):
        for i in range(num_particles):
            x_i = int(np.clip(positions[i, 0], 0, w - 1))
            y_i = int(np.clip(positions[i, 1], 0, h - 1))
            vx = flow[y_i, x_i, 1]
            vy = flow[y_i, x_i, 0]
            positions[i, 0] = np.clip(positions[i, 0] + vx, 0, w - 1)
            positions[i, 1] = np.clip(positions[i, 1] + vy, 0, h - 1)
    return positions


def precompute_flow_info(p0, p1, config):
    """
    Concurrency-friendly function:
      - compute Farneback flow
      - swarm normal flow => pos_center => val_pos
      - swarm negative flow => neg_center => val_neg
      - detect cut
      - pick a center for cut jumps (could just use pos_center)
    Returns a dict with everything needed for final pass.
    """
    cut_threshold = config.get("cut_threshold", 7)
    
    flow = cv2.calcOpticalFlowFarneback(p0, p1, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    if(config.get("pov_mode")):
        # In pov mode, just use the center of the bottom edge of the frame
        max = (p0.shape[1] // 2, p0.shape[0] - 1, 0)
    else:
        max = max_divergence(flow)
    pos_center = max[0:2]
    val_pos = max[2]
        
    # Detect cut based on flow map
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    mean_mag = np.mean(mag)
    if mean_mag > cut_threshold:
        is_cut = True
    else:
        is_cut = False

    cut_center = pos_center[0]

    return {
        "flow": flow,
        "pos_center": pos_center,
        "neg_center": pos_center,
        "val_pos": val_pos,
        "val_neg": val_pos,
        "cut": is_cut,
        "cut_center": cut_center,
        "mean_mag": mean_mag
    }

def precompute_flow_info_gpu(p0, p1, cut_threshold=7):
    # Upload frames to GPU—time to let your GPU do the heavy lifting!
    gpu_p0 = cv2.cuda_GpuMat()
    gpu_p1 = cv2.cuda_GpuMat()
    gpu_p0.upload(p0)
    gpu_p1.upload(p1)

    # Compute optical flow on the GPU
    fb = cv2.cuda_FarnebackOpticalFlow.create(0.5, 3, 15, 3, 5, 1.2, 0)
    flow_gpu = fb.calc(gpu_p0, gpu_p1, None)
    
    # Grab flow back to CPU for any CPU-based ops (like max_divergence)
    flow = flow_gpu.download()
    max_val = max_divergence(flow)
    pos_center = max_val[:2]
    val_pos = max_val[2]

    # Calculate magnitude on GPU (splitting channels)
    channels = cv2.cuda.split(flow_gpu)
    mag_gpu = cv2.cuda.magnitude(channels[0], channels[1])
    mag = mag_gpu.download()
    mean_mag = float(np.mean(mag))
    is_cut = mean_mag > cut_threshold

    cut_center = pos_center[0]

    return {
        "flow": flow,
        "pos_center": pos_center,
        "neg_center": pos_center,
        "val_pos": val_pos,
        "val_neg": val_pos,
        "cut": is_cut,
        "cut_center": cut_center,
        "mean_mag": mean_mag
    }

def precompute_wrapper(p, params):
    return precompute_flow_info(
        p[0], p[1], params)

def fetch_frames(video_path, chunk, params):
    frames_gray = []
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=params["threads"], width=512 if params.get("vr_mode") else 256, height=512 if params.get("vr_mode") else 256)
        batch_frames = vr.get_batch(chunk).asnumpy()
    except Exception as e:
        return frames_gray
    vr = None
    gc.collect()

    for f in batch_frames:
        if params.get("vr_mode"):
            h, w, _ = f.shape
            gray = cv2.cvtColor(f[h // 2:, :w // 2], cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        frames_gray.append(gray)

    return frames_gray

# ---------- Main Processing Function ----------
def process_video(video_path, params, log_func, progress_callback=None, cancel_flag=None, preview_callback=None):
    """
    Example usage that:
      1) Reads frames in bracketed intervals
      2) For each consecutive pair, runs precompute_flow_info() in multiple threads
      3) After concurrency finishes, applies inertia in a single-thread pass
      4) In a final concurrency step, computes radial_flow with the final center
         (and sign) for each pair.
    """
    error_occurred = False
    base, _ = os.path.splitext(video_path)
    output_path = base + ".funscript"
    if os.path.exists(output_path) and not params["overwrite"]:
        log_func(f"Skipping: output file exists ({output_path})")
        return error_occurred

    # Attempt to open video
    try:
        log_func(f"Processing video: {video_path}")
        vr = VideoReader(video_path, ctx=cpu(0), width=1024, height=1024, num_threads=params["threads"])
    except Exception as e:
        log_func(f"ERROR: Unable to open video at {video_path}: {e}")
        return True

    # Basic video properties
    try:
        total_frames = len(vr)
        fps = vr.get_avg_fps()
    except Exception as e:
        log_func(f"ERROR: Unable to read video properties: {e}")
        return True

    step = max(1, int(math.ceil(fps / 15.0)))
    effective_fps = fps / step
    indices = list(range(0, total_frames, step))
    log_func(f"FPS: {fps:.2f}; downsampled to ~{effective_fps:.2f} fps; {len(indices)} frames selected.")

     
    step = max(1, int(math.ceil(fps / 30.0)))
    indices = list(range(0, total_frames, step))
    bracket_size = int(params.get("batch_size", 3000.0))

    center = None
    velocity = np.zeros(2, dtype=float)
    final_flow_list = []

    next_batch = None
    fetch_thread = None
    # We'll collect color frames for preview
    # (and grayscale for computing flow).
    final_con_list = []

    for chunk_start in range(0, len(indices), bracket_size):
        if cancel_flag and cancel_flag():
            log_func("User bailed.")
            return error_occurred

        chunk = indices[chunk_start:chunk_start + bracket_size]
        frame_indices = chunk[:-1]
        if len(chunk) < 2:
            continue

        # Start fetching the next batch while processing the current one
        if fetch_thread:
            fetch_thread.join()  # Ensure previous fetch is complete
            frames_gray = next_batch if next_batch is not None else fetch_frames(video_path, chunk, params)
            next_batch = None  # Clear next_batch after using it
        else:
            frames_gray = fetch_frames(video_path, chunk, params)

        if not frames_gray:
            log_func(f"ERROR: Unable to fetch frames for chunk {chunk_start} - skipping.")
            continue
        if chunk_start + bracket_size < len(indices):
            next_chunk = indices[chunk_start + bracket_size:chunk_start + 2 * bracket_size]
            def fetch_and_store():
                global next_batch
                next_batch = fetch_frames(video_path, next_chunk, params)

            fetch_thread = threading.Thread(target=fetch_and_store)
            fetch_thread.start()

        # Build consecutive pairs
        pairs = list(zip(frames_gray[:-1], frames_gray[1:]))

        with Pool(processes=params["threads"]) as pool:
            precomputed = pool.starmap(precompute_wrapper, [(p, params) for p in pairs])

        # import matplotlib.pyplot as plt
        # mean_mags = [abs(info["val_pos"]) for info in precomputed]
        # plt.plot(mean_mags)
        # plt.show()
        # Add values of val_pos to final_con_list for preview
        final_con_list.extend([abs(info["val_pos"] * 10) for info in precomputed])

        # 2) Single-thread pass to calculate centers based on the median center of the surrounding second, discarding outliers
        final_centers = []
        chosen_center = None
        for j, info in enumerate(precomputed):
            # Use the mean center of the 6 frames in each direction, discarding outliers
            center_list = [info["pos_center"]]
            for i in range(1, 7):
                if j - i >= 0:
                    center_list.append(precomputed[j - i]["pos_center"])
                if j + i < len(precomputed):
                    center_list.append(precomputed[j + i]["pos_center"])
            center_list = np.array(center_list)
            # Discard outliers from center list
            center = np.mean(center_list, axis=0)
            final_centers.append(center)
            
            # Show preview with the new center on the "next" color frame
            # e.g. color_pairs[j][1] is the "current" next frame
            # preview_frame = pairs[j][1].copy()
            # cv2.circle(preview_frame, (int(center[0]), int(center[1])), 6, (0,255,0), -1)
            # cv2.imshow("preview", preview_frame)
            # cv2.waitKey(30)

        # 3) Concurrency to compute final weighted dot products with actual final center
        results_in_bracket = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=params["threads"]) as ex:
            dot_futures = []
            for j, info in enumerate(precomputed):
                dot_futures.append(ex.submit(radial_motion_weighted, info["flow"], final_centers[j], info["cut"], params.get("pov_mode", False)))
            dot_vals = [f.result() for f in dot_futures]


        for j, dot_val in enumerate(dot_vals):
            #flow_val = dot_val * signs[j]
            is_cut   = precomputed[j]["cut"]
            final_flow_list.append((dot_val, is_cut, frame_indices[j]))

        # progress
        if progress_callback:
            prog = min(100, int(100 * (chunk_start + len(chunk)) / len(indices)))
            progress_callback(prog)

    # --- Piecewise Integration and Timestamping 
    cum_flow = [0]
    time_stamps = [final_flow_list[0][2]]

    for i in range(1, len(final_flow_list)):
        flow_prev, cut_prev, t_prev = final_flow_list[i - 1]
        flow_curr, cut_curr, t_curr = final_flow_list[i]

        if cut_curr:
            cum_flow.append(0)
        else:
            # Midpoint integration to reduce phase lag
            mid_flow = (flow_prev + flow_curr) / 2
            cum_flow.append(cum_flow[-1] + mid_flow)

        time_stamps.append(t_curr)

    # Optional: Shift the result back by half a time step to correct for residual phase offset
    cum_flow = [(cum_flow[i] + cum_flow[i-1]) / 2 if i > 0 else cum_flow[i] for i in range(len(cum_flow))]

    # --- Detrending & Normalization ---
    detrend_win = int(params["detrend_window"] * effective_fps)
    disc_threshold = 1000 #float(params.get("discontinuity_threshold", 0.1))  # tweak as needed

    detrended_data = np.zeros_like(cum_flow)
    weight_sum = np.zeros_like(cum_flow)

    # Find indices where a jump occurs
    disc_indices = np.where(np.abs(np.diff(cum_flow)) > disc_threshold)[0] + 1
    # Break data into continuous segments
    segment_boundaries = [0] + list(disc_indices) + [len(cum_flow)]

    overlap = detrend_win // 2

    for i in range(len(segment_boundaries) - 1):
        seg_start = segment_boundaries[i]
        seg_end = segment_boundaries[i + 1]
        seg_length = seg_end - seg_start

        # Just subtract the average  segments with too few points to converge with polypit
        if seg_length < 5:
            detrended_data[seg_start:seg_end] = cum_flow[seg_start:seg_end] - np.mean(cum_flow[seg_start:seg_end])
            continue
        if seg_length <= detrend_win:
            # If segment is too short, process it in one go
            segment = cum_flow[seg_start:seg_end]
            x = np.arange(len(segment))
            trend = np.polyfit(x, segment, 1)
            detrended_segment = segment - np.polyval(trend, x)
            weights = np.hanning(len(segment))
            detrended_data[seg_start:seg_end] += detrended_segment * weights
            weight_sum[seg_start:seg_end] += weights
        else:
            # Process long segments in overlapping windows
            for start in range(seg_start, seg_end - overlap, overlap):
                end = min(start + detrend_win, seg_end)
                segment = cum_flow[start:end]
                x = np.arange(len(segment))
                trend = np.polyfit(x, segment, 1)
                detrended_segment = segment - np.polyval(trend, x)
                weights = np.hanning(len(segment))
                detrended_data[start:end] += detrended_segment * weights
                weight_sum[start:end] += weights

    # Normalize by weight sum to blend overlapping windows
    detrended_data /= np.maximum(weight_sum, 1e-6)

    smoothed_data = np.convolve(detrended_data, [1/16, 1/4, 3/8, 1/4, 1/16], mode='same')
    # Normalize each window in normalization_window to 0-100
    norm_win = int(params["norm_window"] * effective_fps)
    if norm_win % 2 == 0:
        norm_win += 1
    half_norm = norm_win // 2
    norm_rolling = np.empty_like(smoothed_data)
    for i in range(len(smoothed_data)):
        start_idx = max(0, i - half_norm)
        end_idx = min(len(smoothed_data), i + half_norm + 1)
        local_window = smoothed_data[start_idx:end_idx]
        local_min = local_window.min()
        local_max = local_window.max()
        if local_max - local_min == 0:
            norm_rolling[i] = 50
        else:
            norm_rolling[i] = (smoothed_data[i] - local_min) / (local_max - local_min) * 100
    
    # Sine fit (aborted experiment, left here for reference)
    #sine = sine_fit(norm_rolling)

    #Plot sine against norm_rolling
    # import matplotlib.pyplot as plt
    # plt.plot(norm_rolling)
    # #Smooth final con
    # smoothed_final_con = np.convolve(final_con_list, [1/16, 1/4, 3/8, 1/4, 1/16], mode='same')
    # plt.plot(smoothed_final_con)
    # plt.show()

    # TEST: Raw data
    #norm_rolling = sine

    # 3. Keyframe Reduction. Just use slope inversions for now.
    if(params["keyframe_reduction"]):
        key_indices = [0]
        for i in range(1, len(norm_rolling) - 1):
            d1 = norm_rolling[i] - norm_rolling[i - 1]
            d2 = norm_rolling[i + 1] - norm_rolling[i]
            
            if (d1 < 0) != (d2 < 0):
                key_indices.append(i)
        key_indices.append(len(norm_rolling) - 1)
    else:
        key_indices = range(len(norm_rolling))
    actions = []
    for ki in key_indices:  
        try:
            timestamp_ms = int(((time_stamps[ki]) / fps) * 1000)
            pos = int(round(norm_rolling[ki]))
            actions.append({"at": timestamp_ms, "pos": 100-pos})
        except Exception as e:
            log_func(f"Error computing action at segment index {ki}: {e}")
            error_occurred = True
    final_actions = actions

    log_func(f"Keyframe reduction: {len(final_actions)} actions computed.")
    actions = final_actions

    funscript = {"version": "1.0", "actions": actions}
    try:
        with open(output_path, "w") as f:
            json.dump(funscript, f, indent=2)
        log_func(STRINGS["funscript_saved"].format(output_path=output_path))
    except Exception as e:
        log_func(STRINGS["log_error"].format(error=str(e)))
        error_occurred = True
    return error_occurred

# ---------- Sine Fit ------------

def sine_fit(data, error_threshold=5000.0, gain=2.0, min_points=3, max_points=30):
    """
    Fits half-wave sine segments (center=50) to `data` by testing candidate endpoints
    from min_points to max_points ahead. After segmentation, if two consecutive segments
    have the same sign, we split them with an inserted corrective half-wave (with inverted amplitude)
    to help catch missed alternations.

    Returns the fitted array.
    """
    n = len(data)
    segments = []  # each segment is a dict with {'start', 'end', 'A'}
    start = 0

    # --- First pass: Segment the data ---
    while start < n - 1:
        best_err = np.inf
        best_end = None
        best_A = 0.0

        for seg_len in range(min_points, max_points + 1):
            end = start + seg_len
            if end >= n:
                break

            T = seg_len  # segment length (points between endpoints)
            x = np.arange(T + 1)
            model = np.sin(np.pi * x / T)
            segment = data[start:end + 1]
            denom = np.sum(model**2)
            if denom == 0:
                continue
            # Linear LS solution for amplitude A.
            A = np.sum(model * (segment - 50)) / denom
            fit = 50 + A * model
            err = np.sqrt(np.mean((segment - fit) ** 2))

            if err < best_err:
                best_err = err
                best_end = end
                best_A = A

        if best_end is None:
            break

        # Error correction: if error too high, flatten the segment.
        if best_err > error_threshold:
            best_A = 0.0
        # Boost low amplitude segments (because sometimes they're just shy).
        #best_A = np.sign(best_A) * (abs(best_A) ** (1.0 / gain))

        segments.append({'start': start, 'end': best_end, 'A': best_A})
        start = best_end

    # --- Second pass: Correction for consecutive segments with the same sign ---
    corrected_segments = []
    i = 0
    while i < len(segments):
        # If the next segment exists and both segments have nonzero, same-signed amplitude...
        if (i < len(segments) - 1 and segments[i]['A'] != 0 and segments[i+1]['A'] != 0 and
            np.sign(segments[i]['A']) == np.sign(segments[i+1]['A'])):
            combined_start = segments[i]['start']
            combined_end = segments[i+1]['end']
            if (combined_end - combined_start) >= min_points*2:
                L = combined_end - combined_start
                # Split the combined region into three parts.
                mid1 = combined_start + L // 3
                mid2 = combined_start + 2 * L // 3

                # Re-fit first sub-segment.
                T1 = mid1 - combined_start
                if T1 < 2:
                    T1 = 2
                    mid1 = combined_start + T1
                x1 = np.arange(T1 + 1)
                model1 = np.sin(np.pi * x1 / T1)
                seg1 = data[combined_start:mid1 + 1]
                denom1 = np.sum(model1 ** 2)
                A1 = np.sum(model1 * (seg1 - 50)) / denom1 if denom1 != 0 else 0

                # Re-fit third sub-segment.
                T3 = combined_end - mid2
                if T3 < 2:
                    T3 = 2
                    mid2 = combined_end - T3
                x3 = np.arange(T3 + 1)
                model3 = np.sin(np.pi * x3 / T3)
                seg3 = data[mid2:combined_end + 1]
                denom3 = np.sum(model3 ** 2)
                A3 = np.sum(model3 * (seg3 - 50)) / denom3 if denom3 != 0 else 0

                # Corrective (middle) segment: force amplitude opposite in sign.
                A2 = -np.sign(segments[i]['A']) * (0.5 * (abs(A1) + abs(A3)))

                corrected_segments.append({'start': combined_start, 'end': mid1, 'A': A1})
                corrected_segments.append({'start': mid1, 'end': mid2, 'A': A2})
                corrected_segments.append({'start': mid2, 'end': combined_end, 'A': A3})
                i += 2  # skip the next segment; we've merged it
                continue
            else:
                #Comvine them into one segment
                combined_A = segments[i]['A'] + segments[i+1]['A']
                combined_start = segments[i]['start']
                combined_end = segments[i+1]['end']
                corrected_segments.append({'start': combined_start, 'end': combined_end, 'A': combined_A})
                i += 2
                continue

        corrected_segments.append(segments[i])
        i += 1

    # --- Third pass: Detect and fix missed periods ---
    final_segments = []
    for j in range(len(corrected_segments)):
        if j > 0 and j < len(corrected_segments) - 1:
            
            prev_L = corrected_segments[j-1]['end'] - corrected_segments[j-1]['start']
            curr_L = corrected_segments[j]['end'] - corrected_segments[j]['start']
            next_L = corrected_segments[j+1]['end'] - corrected_segments[j+1]['start']
            
            if curr_L > prev_L + next_L:
                # Split into a number of segments depending on the calculated number of missed periods
                missed_periods = round(curr_L / (prev_L + next_L))

                segment_splits = np.linspace(corrected_segments[j]['start'], corrected_segments[j]['end'], 2*missed_periods + 1, dtype=int)
                invert = False
                for split_idx in range(len(segment_splits) - 1):
                    split_segment = {'start': segment_splits[split_idx], 'end': segment_splits[split_idx + 1], 'A': corrected_segments[j]['A'] * (-1 if invert else 1)}
                    invert = not invert
                    final_segments.append(split_segment)
                continue
        final_segments.append(corrected_segments[j])
    #plot the rolling variance of segment lengths, with outliers flagged
    segment_lengths = [seg['end'] - seg['start'] for seg in final_segments]
    # Calculate the rolling variance of segment lengths in a window of 5 segments
    rolling_var = np.full(len(segment_lengths), np.nan)
    for i in range(2, len(segment_lengths) - 2):
        rolling_var[i] = np.var(segment_lengths[i-2:i+3])
    # Flag outliers (variance > 1.5 * median variance)
    var_threshold = 1.5 * np.nanmedian(rolling_var)
    for i in range(len(rolling_var)):
        if rolling_var[i] > var_threshold:
            final_segments[i]['outlier'] = True
    
    # plt.figure(figsize=(10, 5))
    # plt.plot(rolling_var, label='Variance', marker='o')
    # plt.axhline(y=np.mean(rolling_var), color='r', linestyle='--', label='Mean Segment Length')
    # plt.axhline(y=var_threshold, color='g', linestyle=':', label='Variance Threshold', xmin=0, xmax=len(segment_lengths)-1)
    # plt.title('Rolling Variance of Segment Lengths with Outliers Flagged')
    # plt.xlabel('Segment Index')
    # plt.ylabel('Length')
    # plt.legend()
    # plt.show()

    # --- Build the fitted curve from the corrected segments ---
    fitted = np.full(n, 50.0)
    for seg in final_segments:
        s, e = seg['start'], seg['end']
        T = e - s
        if T < 1:
            continue
        x_seg = np.arange(T + 1)
        fitted[s:e + 1] = 50 + seg['A'] * np.sin(np.pi * x_seg / T)

    return fitted

# ---------- Preview Helper ----------
def convert_frame_to_photo(frame):
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        retval, buffer = cv2.imencode('.png', rgb)
        if not retval:
            return None
        img_data = buffer.tobytes()
        return tk.PhotoImage(data=img_data)
    except Exception:
        return None

# ---------- GUI Code ----------
def disable_widgets_except(widget, exceptions):
    if widget not in exceptions:
        try:
            widget.configure(state="disabled")
        except tk.TclError:
            pass
    for child in widget.winfo_children():
        disable_widgets_except(child, exceptions)

def enable_widgets(widget):
    try:
        widget.configure(state="normal")
    except tk.TclError:
        pass
    for child in widget.winfo_children():
        enable_widgets(child)

class App:
    def __init__(self, master):
        self.master = master
        master.title(STRINGS["app_title"])
        icon = tk.PhotoImage(file="icon.png")
        master.iconphoto(False, icon)
        self.files = []
        self.cancel_event = threading.Event()
        self.error_occurred = False
        self.preview_window = None
        self.preview_label = None
        self.show_preview = tk.BooleanVar(value=False)
        self.show_adv = tk.BooleanVar(value=False)
        
        top_frame = tk.Frame(master)
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        btn_sel_files = tk.Button(top_frame, text=STRINGS["select_videos"], command=self.select_files)
        btn_sel_files.pack(side=tk.LEFT, padx=2)
        btn_sel_folder = tk.Button(top_frame, text=STRINGS["select_folder"], command=self.select_folder)
        btn_sel_folder.pack(side=tk.LEFT, padx=2)
        self.lbl_files = tk.Label(top_frame, text=STRINGS["no_files_selected"])
        self.lbl_files.pack(side=tk.LEFT, padx=10)
        btn_readme = tk.Button(top_frame, text=STRINGS["readme"], command=self.show_readme)
        btn_readme.pack(side=tk.RIGHT, padx=2)
        
        mode_frame = tk.Frame(master)
        mode_frame.pack(fill=tk.X, padx=5, pady=2)
        self.vr_mode = tk.BooleanVar(value=False)
        chk_vr = tk.Checkbutton(mode_frame, text=STRINGS["vr_mode"], variable=self.vr_mode)
        chk_vr.pack(side=tk.LEFT, padx=2)
        ToolTip(chk_vr, STRINGS["vr_mode_tooltip"])
        chk_preview = tk.Checkbutton(mode_frame, text=STRINGS["show_preview"], variable=self.show_preview)
        #chk_preview.pack(side=tk.LEFT, padx=2)
        self.pov_mode = tk.BooleanVar(value=False)
        chk_pov = tk.Checkbutton(mode_frame, text="POV Mode", variable=self.pov_mode)
        chk_pov.pack(side=tk.LEFT, padx=2)
        ToolTip(chk_pov, STRINGS["pov_mode_tooltip"])
        
        adv_toggle_frame = tk.Frame(master)
        adv_toggle_frame.pack(fill=tk.X, padx=5, pady=2)
        chk_adv = tk.Checkbutton(adv_toggle_frame, text=STRINGS["show_advanced"] if "show_advanced" in STRINGS else "Show Advanced Settings", variable=self.show_adv, command=self.toggle_advanced)
        chk_adv.pack(side=tk.LEFT, padx=2)
        
        prog_frame = tk.Frame(master)
        prog_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(prog_frame, text=STRINGS["overall_progress"]).pack(anchor=tk.W)
        self.overall_progress = ttk.Progressbar(prog_frame, orient="horizontal", mode="determinate", maximum=100)
        self.overall_progress.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(prog_frame, text=STRINGS["current_video_progress"]).pack(anchor=tk.W)
        self.video_progress = ttk.Progressbar(prog_frame, orient="horizontal", mode="determinate", maximum=100)
        self.video_progress.pack(fill=tk.X, padx=5, pady=2)
        
        self.adv_frame = tk.LabelFrame(master, text=STRINGS["advanced_settings"])
        self.adv_frame.pack(fill=tk.X, padx=5, pady=5)
        self.params = {}
        # Get the number of cores available
        num_cores = os.cpu_count()
        self.create_param(self.adv_frame, STRINGS["threads"], "threads", str(num_cores), "Number of threads used for optical flow computation.")
        self.create_param(self.adv_frame, STRINGS["detrend_window"], "detrend_window", "1.5", "Controls the aggressiveness of drift removal. See readme for detail.  Recommended: 1-10, higher values for more stable cameras.")
        
        self.create_param(self.adv_frame, STRINGS["norm_window"], "norm_window", "4", "Time window to calibrate motion range (seconds). Shorter values amplify motion, but also cause artifacts in long thrusts.")
        self.create_param(self.adv_frame, STRINGS["batch_size"], "batch_size", "3000", "Number of frames to process per batch (Higher values will be faster, but also take more RAM).")
        
        self.keyframe_reduction = tk.BooleanVar(value=True)
        chk_keyframe = tk.Checkbutton(self.adv_frame, text=STRINGS["chk_keyframe"] if "chk_keyframe" in STRINGS else "Enable keyframe reduction", variable=self.keyframe_reduction)
        chk_keyframe.pack(anchor=tk.W, padx=5, pady=2)
        #ToolTip(chk_keyframe, STRINGS["keyframe_tooltip"])
        
        self.overwrite = tk.BooleanVar(value=False)
        chk_overwrite = tk.Checkbutton(self.adv_frame, text=STRINGS["overwrite_files"], variable=self.overwrite)
        chk_overwrite.pack(anchor=tk.W, padx=5, pady=2)
        
        btn_frame = tk.Frame(master)
        btn_frame.pack(padx=5, pady=5)
        btn_run = tk.Button(btn_frame, text=STRINGS["run"], command=self.run_batch)
        btn_run.pack(side=tk.LEFT, padx=5)
        self.btn_cancel = tk.Button(btn_frame, text=STRINGS["cancel"], command=self.cancel_run)
        self.btn_cancel.pack(side=tk.LEFT, padx=5)
        
        self.log_file = None
        self.error_occurred = False
        self.load_config()
        self.toggle_advanced()
    
    def create_param(self, parent, label_text, key, default, tooltip_text):
        frm = tk.Frame(parent)
        frm.pack(fill=tk.X, padx=5, pady=2)
        lbl = tk.Label(frm, text=label_text, width=25, anchor=tk.W)
        lbl.pack(side=tk.LEFT)
        entry = tk.Entry(frm)
        entry.insert(0, default)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.params[key] = entry
        ToolTip(entry, tooltip_text)
    
    def select_files(self):
        files = filedialog.askopenfilenames(title=STRINGS["select_videos"],
                    filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All Files", "*.*")])
        if files:
            self.files = list(files)
            self.lbl_files.config(text=f"{len(self.files)} file(s) selected")
        else:
            self.files = []
            self.lbl_files.config(text=STRINGS["no_files_selected"])
    
    def select_folder(self):
        folder = filedialog.askdirectory(title=STRINGS["select_folder"])
        if folder:
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
            found = []
            for root, dirs, files in os.walk(folder):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in video_exts:
                        found.append(os.path.join(root, f))
            self.files = found
            self.lbl_files.config(text=f"{len(self.files)} file(s) found in folder")
    
    def show_readme(self):
        try:
            with open("readme.txt", "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            content = f"Error reading readme.txt: {e}"
        win = tk.Toplevel(self.master)
        win.title("Readme")
        txt = tk.Text(win, wrap=tk.WORD)
        txt.insert(tk.END, content)
        txt.pack(fill=tk.BOTH, expand=True)
    
    def update_preview(self, frame):
        photo = convert_frame_to_photo(frame)
        if photo is None:
            return
        if self.preview_window is None:
            self.preview_window = tk.Toplevel(self.master)
            self.preview_window.title("Preview")
            self.preview_label = tk.Label(self.preview_window)
            self.preview_label.pack()
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo
    
    def save_config(self):
        config = { key: self.params[key].get() for key in self.params }
        config["overwrite"] = self.overwrite.get()
        config["vr_mode"] = self.vr_mode.get()
        config["pov_mode"] = self.pov_mode.get()
        
        config_path = "config.json"
        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            messagebox.showinfo("Config Saved", STRINGS["config_saved"].format(config_path=config_path))
        except Exception as e:
            messagebox.showerror("Error", f"Could not save config: {e}")
    
    def load_config(self):
        config_path = "config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                for key in self.params:
                    if key in config:
                        self.params[key].delete(0, tk.END)
                        self.params[key].insert(0, str(config[key]))
                self.overwrite.set(config.get("overwrite", False))
                self.vr_mode.set(config.get("vr_mode", False))
                self.pov_mode.set(config.get("pov_mode", False))
                
            except Exception as e:
                messagebox.showwarning("Config Load", STRINGS["config_load_error"].format(error=str(e)))
    
    def toggle_advanced(self):
        if self.show_adv.get():
            self.adv_frame.pack(fill=tk.X, padx=5, pady=5)
        else:
            self.adv_frame.forget()
    
    def update_video_progress(self, prog):
        self.master.after(0, lambda: self.video_progress.configure(value=prog))
    
    def update_overall_progress(self, prog):
        self.master.after(0, lambda: self.overall_progress.configure(value=prog))
    
    def cancel_run(self):
        self.cancel_event.set()
    
    def log(self, msg):
        self.log_file.write(msg + "\n")
        self.log_file.flush()
        
    def run_batch(self):
        if not self.files:
            messagebox.showwarning("No files", STRINGS["no_files_warning"])
            return
        try:
            settings = {
                "threads": int(self.params["threads"].get()),
                "detrend_window": float(self.params["detrend_window"].get()),
                
                "norm_window": float(self.params["norm_window"].get()),
                "batch_size": int(self.params["batch_size"].get()),
                "overwrite": self.overwrite.get(),
                "keyframe_reduction": self.keyframe_reduction.get()
            }
            settings["vr_mode"] = self.vr_mode.get()
            settings["pov_mode"] = self.pov_mode.get()
        except Exception as e:
            messagebox.showerror("Parameter Error", f"Invalid parameters: {e}")
            return
        
        self.cancel_event.clear()
        try:
            # Open a log file in appdata
            log_path = os.path.join(os.getenv("APPDATA"), "FunscriptFlow")
            os.makedirs(log_path, exist_ok=True)
            log_filename = os.path.join(log_path, "run.log")
            self.log_file = open(log_filename, "w")
        except Exception as e:
            messagebox.showerror("Log Error", f"Cannot open log file: {e}")
            return
        self.overall_progress.configure(value=0)
        self.video_progress.configure(value=0)
        disable_widgets_except(self.master, [self.btn_cancel])
        total_files = len(self.files)
        self.error_occurred = False
        def worker():
            for idx, video in enumerate(self.files):
                if self.cancel_event.is_set():
                    self.log(STRINGS["cancelled_by_user"])
                    break
                self.update_video_progress(0)
                err = process_video(video, settings, self.log,
                              progress_callback=lambda prog: self.update_video_progress(prog),
                              cancel_flag=lambda: self.cancel_event.is_set(),
                              preview_callback=lambda frame: self.update_preview(frame) if self.show_preview.get() else None)
                if err:
                    self.error_occurred = True
                overall = int(100 * (idx + 1) / total_files)
                self.update_overall_progress(overall)
            self.log(STRINGS["batch_processing_complete"])
            self.log_file.close()
            enable_widgets(self.master)
            if self.error_occurred:
                if messagebox.askyesno("Run Finished", STRINGS["processing_completed_with_errors"] + "\nWould you like to open the log?"):
                    open_log(log_filename)
            else:
                if messagebox.askyesno("Run Finished", "Batch processing complete.\nSee run.log for details.\nWould you like to open the log?"):
                    open_log(log_filename)
        threading.Thread(target=worker, daemon=True).start()

def open_log(log_filename):
    os.startfile(log_filename)
    
def disable_widgets_except(widget, exceptions):
    if widget not in exceptions:
        try:
            widget.configure(state="disabled")
        except tk.TclError:
            pass
    for child in widget.winfo_children():
        disable_widgets_except(child, exceptions)

def enable_widgets(widget):
    try:
        widget.configure(state="normal")
    except tk.TclError:
        pass
    for child in widget.winfo_children():
        enable_widgets(child)

# ---------- Headless Mode ----------
def run_headless(input_path, settings):
    log_filename = "run.log"
    try:
        logf = open(log_filename, "w")
    except Exception as e:
        print(f"Error opening log file: {e}")
        return
    def log_func(msg):
        logf.write(msg + "\n")
        logf.flush()
        print(msg)
    if os.path.isdir(input_path):
        video_exts = {".mp4", ".avi", ".mov", ".mkv"}
        files = []
        for root, dirs, files_in in os.walk(input_path):
            for f in files_in:
                ext = os.path.splitext(f)[1].lower()
                if ext in video_exts:
                    files.append(os.path.join(root, f))
    else:
        files = [input_path]
    if not files:
        print("No video files found.")
        logf.write("No video files found.\n")
        logf.close()
        return
    total_files = len(files)
    log_func(STRINGS["found_files"].format(n=total_files))
    for idx, video in enumerate(files):
        log_func(STRINGS["processing_file"].format(current=idx+1, total=total_files, video_path=video))
        process_video(video, settings, log_func, progress_callback=lambda prog: print(f"Video progress: {prog}%"))
    log_func(STRINGS["batch_processing_complete"])
    logf.close()
    print("Done. See run.log for details.")

# ---------- Main ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optical Flow to Funscript")
    parser.add_argument("input", nargs="?", help="Input video file or folder")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads (default: 8)")
    parser.add_argument("--detrend_window", type=float, default=2.0, help="Detrend window in seconds (default: 2.0)")
    parser.add_argument("--norm_window", type=float, default=3.0, help="Normalization window in seconds (default: 3.0)")
    parser.add_argument("--batch_size", type=int, default=3000, help="Batch size in frames (default: 3000)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--vr_mode", action="store_true", help="Enable VR Mode (if not set, non‑VR mode is used)")
    parser.add_argument("--pov_mode", action="store_true", help="Enable POV Mode (improves stability for POV videos)")
    parser.add_argument("--disable_keyframe_reduction", action="store_false", help="Disable keyframe reduction")
    args = parser.parse_args()
    settings = {
        "threads": args.threads,
        "detrend_window": args.detrend_window,
        "norm_window": args.norm_window,
        "batch_size": args.batch_size,
        "overwrite": args.overwrite,
        "vr_mode": args.vr_mode,
        "pov_mode": args.pov_mode,
        "keyframe_reduction": not args.disable_keyframe_reduction
    }
    if args.input:
        run_headless(args.input, settings)
    else:
        root = tk.Tk()
        app = App(root)
        root.mainloop()
