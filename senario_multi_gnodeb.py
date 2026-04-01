import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path

# =========================
# Your scenarios
# =========================

scenario_1 = {
    'n_gnbs': 20,
    'max_prbs_per_gnb': [150] * 20,
    'gnb_positions': [
        (0.0, 0.0), (300.0, 0.0), (600.0, 0.0), (900.0, 0.0), (1200.0, 0.0),
        (0.0, 300.0), (300.0, 300.0), (600.0, 300.0), (900.0, 300.0), (1200.0, 300.0),
        (0.0, 600.0), (300.0, 600.0), (600.0, 600.0), (900.0, 600.0), (1200.0, 600.0),
        (0.0, 900.0), (300.0, 900.0), (600.0, 900.0), (900.0, 900.0), (1200.0, 900.0),
    ],
    'coverage_radius': [250.0] * 20,
    'carrier_ids': [0, 1, 0, 1, 0,
                    1, 0, 1, 0, 1,
                    0, 1, 0, 1, 0,
                    1, 0, 1, 0, 1],
    'n_ues': 400,
    'ue_distribution': 'clustered',
    'slices': [
        {'type': 'eMBB', 'count': 4},
        {'type': 'mMTC', 'count': 3},
        {'type': 'URLLC', 'count': 1},
    ]
}

scenario_2 = {
    'n_gnbs': 25,
    'max_prbs_per_gnb': [150] * 25,
    'gnb_positions': [
        (0.0, 0.0), (250.0, 0.0), (500.0, 0.0), (750.0, 0.0), (1000.0, 0.0),
        (0.0, 250.0), (250.0, 250.0), (500.0, 250.0), (750.0, 250.0), (1000.0, 250.0),
        (0.0, 500.0), (250.0, 500.0), (500.0, 500.0), (750.0, 500.0), (1000.0, 500.0),
        (0.0, 750.0), (250.0, 750.0), (500.0, 750.0), (750.0, 750.0), (1000.0, 750.0),
        (0.0, 1000.0), (250.0, 1000.0), (500.0, 1000.0), (750.0, 1000.0), (1000.0, 1000.0),
    ],
    'coverage_radius': [220.0] * 25,
    'carrier_ids': [
        0, 1, 2, 0, 1,
        1, 2, 0, 1, 2,
        2, 0, 1, 2, 0,
        0, 1, 2, 0, 1,
        1, 2, 0, 1, 2
    ],
    'n_ues': 600,
    'ue_distribution': 'hotspot',
    'slices': [
        {'type': 'eMBB', 'count': 5},
        {'type': 'mMTC', 'count': 3},
        {'type': 'URLLC', 'count': 2},
    ]
}

scenario_3 = {
    'n_gnbs': 20,
    'max_prbs_per_gnb': [
        200, 150, 150, 200, 150,
        150, 100, 100, 150, 100,
        150, 100, 200, 150, 100,
        200, 150, 150, 200, 150
    ],
    'gnb_positions': [
        (0.0, 0.0), (300.0, 0.0), (600.0, 0.0), (900.0, 0.0), (1200.0, 0.0),
        (0.0, 300.0), (300.0, 300.0), (600.0, 300.0), (900.0, 300.0), (1200.0, 300.0),
        (0.0, 600.0), (300.0, 600.0), (600.0, 600.0), (900.0, 600.0), (1200.0, 600.0),
        (0.0, 900.0), (300.0, 900.0), (600.0, 900.0), (900.0, 900.0), (1200.0, 900.0),
    ],
    'coverage_radius': [
        260.0, 240.0, 240.0, 260.0, 240.0,
        240.0, 220.0, 220.0, 240.0, 220.0,
        240.0, 220.0, 260.0, 240.0, 220.0,
        260.0, 240.0, 240.0, 260.0, 240.0
    ],
    'carrier_ids': [0, 1, 0, 1, 0,
                    1, 0, 1, 0, 1,
                    0, 1, 0, 1, 0,
                    1, 0, 1, 0, 1],
    'n_ues': 500,
    'ue_distribution': 'clustered',
    'slices': [
        {'type': 'eMBB', 'count': 4},
        {'type': 'mMTC', 'count': 4},
        {'type': 'URLLC', 'count': 1},
    ]
}

scenario_dense_20 = {
    'n_gnbs': 20,
    'max_prbs_per_gnb': [150] * 20,
    'gnb_positions': [
        (0.0, 0.0), (300.0, 0.0), (600.0, 0.0), (900.0, 0.0), (1200.0, 0.0),
        (0.0, 300.0), (300.0, 300.0), (600.0, 300.0), (900.0, 300.0), (1200.0, 300.0),
        (0.0, 600.0), (300.0, 600.0), (600.0, 600.0), (900.0, 600.0), (1200.0, 600.0),
        (0.0, 900.0), (300.0, 900.0), (600.0, 900.0), (900.0, 900.0), (1200.0, 900.0),
    ],
    'coverage_radius': [250.0] * 20,
    'carrier_ids': [0, 1, 0, 1, 0,
                    1, 0, 1, 0, 1,
                    0, 1, 0, 1, 0,
                    1, 0, 1, 0, 1],
    'n_ues': 400,
    'ue_distribution': 'hotspot',
    'slices': [
        {'type': 'eMBB', 'count': 4},
        {'type': 'mMTC', 'count': 3},
        {'type': 'URLLC', 'count': 1},
    ]
}

scenarios = {
    "scenario_1": scenario_1,
    "scenario_2": scenario_2,
    "scenario_3": scenario_3,
    "scenario_dense_20": scenario_dense_20,
}

# =========================
# Hexagon plotting helpers
# =========================

def hexagon_vertices(x, y, radius):
    """
    Same orientation as your NodeB._calculate_hexagon_vertices():
    angle = 0, 60, 120, 180, 240, 300 degrees
    """
    vertices = []
    for i in range(6):
        angle_deg = 60 * i
        angle_rad = math.radians(angle_deg)
        xv = x + radius * math.cos(angle_rad)
        yv = y + radius * math.sin(angle_rad)
        vertices.append((xv, yv))
    return vertices


def plot_scenario_hex(name, scenario, save_dir="scenario_maps_hex", show=False):
    positions = scenario["gnb_positions"]
    radii = scenario["coverage_radius"]
    carriers = scenario["carrier_ids"]
    prbs = scenario["max_prbs_per_gnb"]

    fig, ax = plt.subplots(figsize=(12, 9))

    all_x = []
    all_y = []

    for i, ((x, y), r, c, p) in enumerate(zip(positions, radii, carriers, prbs)):
        verts = hexagon_vertices(x, y, r)
        poly = Polygon(verts, closed=True, fill=False, linewidth=1.2, alpha=0.7)
        ax.add_patch(poly)

        ax.scatter([x], [y], s=35)
        ax.text(
            x + 8, y + 8,
            f"gNB {i}\nC{c}\n{p} PRB",
            fontsize=7
        )

        all_x.extend(v[0] for v in verts)
        all_y.extend(v[1] for v in verts)

    margin = 80
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        f"{name}\n"
        f"gNBs={scenario['n_gnbs']} | "
        f"UEs={scenario['n_ues']} | "
        f"distribution={scenario['ue_distribution']} | "
        f"slices={scenario['slices']}"
    )

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(save_dir) / f"{name}_hex.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


# =========================
# Plot all scenarios
# =========================

if __name__ == "__main__":
    for name, sc in scenarios.items():
        path = plot_scenario_hex(name, sc, save_dir="scenario_maps_hex", show=False)
        print(f"Saved: {path}")