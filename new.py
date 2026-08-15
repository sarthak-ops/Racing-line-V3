import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import time
import os
import csv

# Config
TRACK_FILE = "berlin.csv"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_NODES = 300
ITERATIONS = 300
ALPHA = 0.2

# 1. LOAD TRACK 
data = np.loadtxt(
    TRACK_FILE,
    delimiter=",",
    comments="#"
)

x_raw = data[:, 0]
y_raw = data[:, 1]
w_right_raw = data[:, 2]
w_left_raw = data[:, 3]

print(f"Loaded {len(x_raw)} raw points from {TRACK_FILE}")

# 2. CLOSE THE TRACK
first_last_gap = np.hypot(
    x_raw[-1] - x_raw[0],
    y_raw[-1] - y_raw[0]
)

print(f"Gap between first and last raw point: {first_last_gap:.4f} m")

# Append the first point so the spline can be periodic.
x = np.append(x_raw, x_raw[0])
y = np.append(y_raw, y_raw[0])

w_right = np.append(w_right_raw, w_right_raw[0])
w_left = np.append(w_left_raw, w_left_raw[0])

# 3. ARC-LENGTH PARAMETERIZATION
s = np.zeros(len(x))

for i in range(1, len(x)):
    segment_length = np.hypot(
        x[i] - x[i - 1],
        y[i] - y[i - 1]
    )

    s[i] = s[i - 1] + segment_length

track_length = s[-1]

print(f"Track length: {track_length:.1f} m")

# 4. PERIODIC SPLINE
cs_x = CubicSpline(
    s,
    x,
    bc_type="periodic"
)

cs_y = CubicSpline(
    s,
    y,
    bc_type="periodic"
)

# 5. SAMPLE TRACK
# endpoint=False prevents duplicating the first point.
nodes = np.linspace(
    0,
    track_length,
    N_NODES,
    endpoint=False
)

center_x = cs_x(nodes)
center_y = cs_y(nodes)

w_right_s = np.interp(
    nodes,
    s,
    w_right
)

w_left_s = np.interp(
    nodes,
    s,
    w_left
)

# 6. CALCULATE TANGENTS AND NORMALS
dx = cs_x(nodes, 1)
dy = cs_y(nodes, 1)

magnitude = np.hypot(dx, dy)

tx = dx / magnitude
ty = dy / magnitude

# Left-facing normal.
nx = -ty
ny = tx

# 7. DISPLACEMENT-BASED RACING LINE OPTIMIZATION
n = len(center_x)

# Displacement from centerline.
d = np.zeros(n)

print(
    f"Running geometric optimization "
    f"({ITERATIONS} iterations, {N_NODES} nodes)..."
)

start_time = time.perf_counter()

for iteration in range(ITERATIONS):

    # Current racing line.
    racing_x = center_x + d * nx
    racing_y = center_y + d * ny

    for i in range(n):

        previous = (i - 1) % n
        next_point = (i + 1) % n

        # Vector toward previous point.
        v1 = np.array([
            racing_x[previous] - racing_x[i],
            racing_y[previous] - racing_y[i]
        ])

        # Vector toward next point.
        v2 = np.array([
            racing_x[next_point] - racing_x[i],
            racing_y[next_point] - racing_y[i]
        ])

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 < 1e-9 or norm_v2 < 1e-9:
            continue

        v1 /= norm_v1
        v2 /= norm_v2

        # Approximate direction of the local curve bisector.
        bisector = v1 + v2

        norm_bisector = np.linalg.norm(bisector)

        if norm_bisector < 1e-9:
            continue

        bisector /= norm_bisector

        normal = np.array([
            nx[i],
            ny[i]
        ])

        # Determine whether moving along the normal
        # would improve the local geometry.
        dp = np.dot(
            bisector,
            normal
        )

        # Update displacement.
        d[i] += ALPHA * dp

        # Keep the racing line inside the track.
        d[i] = np.clip(
            d[i],
            -w_left_s[i],
            w_right_s[i]
        )


elapsed = time.perf_counter() - start_time

print(f"Optimization runtime: {elapsed:.4f} seconds")

# 8. RECOMPUTE FINAL RACING LINE
racing_x = center_x + d * nx
racing_y = center_y + d * ny

# 9. TRACK BOUNDARIES
left_x = center_x - nx * w_left_s
left_y = center_y - ny * w_left_s

right_x = center_x + nx * w_right_s
right_y = center_y + ny * w_right_s

# 10. SAVE RACING LINE
output_file = os.path.join(
    RESULTS_DIR,
    "berlin_geometric_line.csv"
)

with open(
    output_file,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "s_m",
        "x_m",
        "y_m",
        "d_offset_m",
        "width_left_m",
        "width_right_m"
    ])

    for i in range(n):

        writer.writerow([
            nodes[i],
            racing_x[i],
            racing_y[i],
            d[i],
            w_left_s[i],
            w_right_s[i]
        ])

print(f"Saved racing line: {output_file}")

# 11. SAVE RUN METADATA
metadata_file = os.path.join(
    RESULTS_DIR,
    "berlin_geometric_line_meta.txt"
)

with open(metadata_file, "w") as f:

    f.write(f"track: {TRACK_FILE}\n")
    f.write("method: geometric racing line\n")
    f.write(f"nodes: {N_NODES}\n")
    f.write(f"iterations: {ITERATIONS}\n")
    f.write(f"alpha: {ALPHA}\n")
    f.write(f"track_length_m: {track_length:.4f}\n")
    f.write(f"runtime_s: {elapsed:.6f}\n")

print(f"Saved metadata: {metadata_file}")

# 12. PLOT
plt.figure(figsize=(12, 10))

plt.plot(
    left_x,
    left_y,
    "k",
    linewidth=1.5,
    label="left boundary"
)

plt.plot(
    right_x,
    right_y,
    "k",
    linewidth=1.5,
    label="right boundary"
)

plt.plot(
    center_x,
    center_y,
    "--",
    color="gray",
    linewidth=1,
    label="centerline"
)

plt.plot(
    racing_x,
    racing_y,
    "r",
    linewidth=2,
    label="racing line"
)

plt.legend()
plt.axis("equal")

plt.title(
    f"Racing line optimization — {TRACK_FILE}"
)

plt.grid(alpha=0.3)

plot_file = os.path.join(
    RESULTS_DIR,
    "berlin_geometric_line.png"
)

plt.savefig(
    plot_file,
    dpi=150,
    bbox_inches="tight"
)

print(f"Saved plot: {plot_file}")
plt.show()
