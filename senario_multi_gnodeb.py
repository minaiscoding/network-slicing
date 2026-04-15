import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


DEFAULT_SLICES = [
    {"type": "eMBB", "count": 2},
    {"type": "mMTC", "count": 1},
    {"type": "URLLC", "count": 1},
]

DEFAULT_SLICE_PRIORITIES = {
    "eMBB": 1.0,
    "mMTC": 1.0,
    "URLLC": 1.0,
}


def make_slice_priorities(overrides=None):
    priorities = DEFAULT_SLICE_PRIORITIES.copy()
    if overrides:
        priorities.update(overrides)
    return priorities


def make_scenario(
    n_gnbs,
    gnb_positions,
    coverage_radius,
    carrier_ids,
    max_prbs_per_gnb,
    n_ues,
    ue_distribution,
    slices=None,
    slice_priorities=None,
):
    return {
        "n_gnbs": n_gnbs,
        "gnb_positions": gnb_positions,
        "coverage_radius": coverage_radius,
        "carrier_ids": carrier_ids,
        "max_prbs_per_gnb": max_prbs_per_gnb,
        "n_ues": n_ues,
        "ue_distribution": ue_distribution,
        "slices": slices if slices is not None else DEFAULT_SLICES,
        "slice_priorities": slice_priorities or {},
    }


# ============================================================
# Small / focused scenarios
# ============================================================

scenario_3gnb_overlap = make_scenario(
    n_gnbs=3,
    gnb_positions=[
        (0.0, 0.0),
        (400.0, 0.0),
        (200.0, 350.0),
    ],
    coverage_radius=[350.0, 350.0, 350.0],
    carrier_ids=[0, 0, 0],
    max_prbs_per_gnb=[150, 150, 150],
    n_ues=200,
    ue_distribution="clustered",
    slice_priorities={
        0: make_slice_priorities({"mMTC": 0.4, "URLLC": 0.3}),
        1: make_slice_priorities({"eMBB": 0.4, "mMTC": 0.5}),
        2: make_slice_priorities({"eMBB": 0.7, "URLLC": 0.6}),
    },
)

scenario_3gnb_overlap_large = make_scenario(
    n_gnbs=3,
    gnb_positions=[
        (0.0, 0.0),
        (1200.0, 0.0),
        (600.0, 1040.0),
    ],
    coverage_radius=[800.0, 800.0, 800.0],
    carrier_ids=[0, 0, 0],
    max_prbs_per_gnb=[150, 150, 150],
    n_ues=200,
    ue_distribution="clustered",
)

scenario_4gnb_mixed = make_scenario(
    n_gnbs=4,
    gnb_positions=[
        (0.0, 0.0),
        (900.0, 0.0),
        (450.0, 780.0),
        (1650.0, 250.0),
    ],
    coverage_radius=[700.0, 700.0, 700.0, 650.0],
    carrier_ids=[0, 0, 0, 0],
    max_prbs_per_gnb=[150, 150, 150, 150],
    n_ues=200,
    ue_distribution="clustered",
)


# ============================================================
# Larger grid scenarios
# ============================================================

scenario_1 = make_scenario(
    n_gnbs=20,
    gnb_positions=[
        (0.0, 0.0), (300.0, 0.0), (600.0, 0.0), (900.0, 0.0), (1200.0, 0.0),
        (0.0, 300.0), (300.0, 300.0), (600.0, 300.0), (900.0, 300.0), (1200.0, 300.0),
        (0.0, 600.0), (300.0, 600.0), (600.0, 600.0), (900.0, 600.0), (1200.0, 600.0),
        (0.0, 900.0), (300.0, 900.0), (600.0, 900.0), (900.0, 900.0), (1200.0, 900.0),
    ],
    coverage_radius=[250.0] * 20,
    carrier_ids=[
        0, 1, 0, 1, 0,
        1, 0, 1, 0, 1,
        0, 1, 0, 1, 0,
        1, 0, 1, 0, 1,
    ],
    max_prbs_per_gnb=[150] * 20,
    n_ues=400,
    ue_distribution="clustered",
    slices=[
        {"type": "eMBB", "count": 4},
        {"type": "mMTC", "count": 3},
        {"type": "URLLC", "count": 1},
    ],
)

scenario_2 = make_scenario(
    n_gnbs=25,
    gnb_positions=[
        (0.0, 0.0), (250.0, 0.0), (500.0, 0.0), (750.0, 0.0), (1000.0, 0.0),
        (0.0, 250.0), (250.0, 250.0), (500.0, 250.0), (750.0, 250.0), (1000.0, 250.0),
        (0.0, 500.0), (250.0, 500.0), (500.0, 500.0), (750.0, 500.0), (1000.0, 500.0),
        (0.0, 750.0), (250.0, 750.0), (500.0, 750.0), (750.0, 750.0), (1000.0, 750.0),
        (0.0, 1000.0), (250.0, 1000.0), (500.0, 1000.0), (750.0, 1000.0), (1000.0, 1000.0),
    ],
    coverage_radius=[220.0] * 25,
    carrier_ids=[
        0, 1, 2, 0, 1,
        1, 2, 0, 1, 2,
        2, 0, 1, 2, 0,
        0, 1, 2, 0, 1,
        1, 2, 0, 1, 2,
    ],
    max_prbs_per_gnb=[150] * 25,
    n_ues=600,
    ue_distribution="hotspot",
    slices=[
        {"type": "eMBB", "count": 5},
        {"type": "mMTC", "count": 3},
        {"type": "URLLC", "count": 2},
    ],
)

scenario_3 = make_scenario(
    n_gnbs=20,
    gnb_positions=[
        (0.0, 0.0), (300.0, 0.0), (600.0, 0.0), (900.0, 0.0), (1200.0, 0.0),
        (0.0, 300.0), (300.0, 300.0), (600.0, 300.0), (900.0, 300.0), (1200.0, 300.0),
        (0.0, 600.0), (300.0, 600.0), (600.0, 600.0), (900.0, 600.0), (1200.0, 600.0),
        (0.0, 900.0), (300.0, 900.0), (600.0, 900.0), (900.0, 900.0), (1200.0, 900.0),
    ],
    coverage_radius=[
        260.0, 240.0, 240.0, 260.0, 240.0,
        240.0, 220.0, 220.0, 240.0, 220.0,
        240.0, 220.0, 260.0, 240.0, 220.0,
        260.0, 240.0, 240.0, 260.0, 240.0,
    ],
    carrier_ids=[
        0, 1, 0, 1, 0,
        1, 0, 1, 0, 1,
        0, 1, 0, 1, 0,
        1, 0, 1, 0, 1,
    ],
    max_prbs_per_gnb=[
        200, 150, 150, 200, 150,
        150, 100, 100, 150, 100,
        150, 100, 200, 150, 100,
        200, 150, 150, 200, 150,
    ],
    n_ues=500,
    ue_distribution="clustered",
    slices=[
        {"type": "eMBB", "count": 4},
        {"type": "mMTC", "count": 4},
        {"type": "URLLC", "count": 1},
    ],
)

scenario_dense_20 = make_scenario(
    n_gnbs=20,
    gnb_positions=[
        (0.0, 0.0), (300.0, 0.0), (600.0, 0.0), (900.0, 0.0), (1200.0, 0.0),
        (0.0, 300.0), (300.0, 300.0), (600.0, 300.0), (900.0, 300.0), (1200.0, 300.0),
        (0.0, 600.0), (300.0, 600.0), (600.0, 600.0), (900.0, 600.0), (1200.0, 600.0),
        (0.0, 900.0), (300.0, 900.0), (600.0, 900.0), (900.0, 900.0), (1200.0, 900.0),
    ],
    coverage_radius=[250.0] * 20,
    carrier_ids=[
        0, 1, 0, 1, 0,
        1, 0, 1, 0, 1,
        0, 1, 0, 1, 0,
        1, 0, 1, 0, 1,
    ],
    max_prbs_per_gnb=[150] * 20,
    n_ues=400,
    ue_distribution="hotspot",
    slices=[
        {"type": "eMBB", "count": 4},
        {"type": "mMTC", "count": 3},
        {"type": "URLLC", "count": 1},
    ],
)


scenarios = {
    "scenario_3gnb_overlap": scenario_3gnb_overlap,
    "scenario_3gnb_overlap_large": scenario_3gnb_overlap_large,
    "scenario_4gnb_mixed": scenario_4gnb_mixed,
    "scenario_1": scenario_1,
    "scenario_2": scenario_2,
    "scenario_3": scenario_3,
    "scenario_dense_20": scenario_dense_20,
}


def hexagon_vertices(x, y, radius):
    vertices = []
    for i in range(6):
        angle_rad = math.radians(60 * i)
        vertices.append((
            x + radius * math.cos(angle_rad),
            y + radius * math.sin(angle_rad),
        ))
    return vertices


def format_slices(slices):
    return ", ".join(f"{s['type']}:{s['count']}" for s in slices)


def plot_scenario_hex(name, scenario, save_dir="scenario_maps_hex", show=False):
    positions = scenario["gnb_positions"]
    radii = scenario["coverage_radius"]
    carriers = scenario["carrier_ids"]
    prbs = scenario["max_prbs_per_gnb"]
    slices = scenario.get("slices", [])

    fig, ax = plt.subplots(figsize=(12, 9))

    all_x = []
    all_y = []

    for idx, ((x, y), radius, carrier_id, prb_count) in enumerate(
        zip(positions, radii, carriers, prbs)
    ):
        verts = hexagon_vertices(x, y, radius)
        polygon = Polygon(verts, closed=True, fill=False, linewidth=1.2, alpha=0.7)
        ax.add_patch(polygon)

        ax.scatter([x], [y], s=35)
        ax.text(
            x + 8,
            y + 8,
            f"gNB {idx}\nC{carrier_id}\n{prb_count} PRB",
            fontsize=7,
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
        f"slices={format_slices(slices)}"
    )

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    output_path = save_path / f"{name}_hex.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


if __name__ == "__main__":
    for scenario_name, scenario_cfg in scenarios.items():
        output = plot_scenario_hex(
            scenario_name,
            scenario_cfg,
            save_dir="scenario_maps_hex",
            show=False,
        )
        print(f"Saved: {output}")