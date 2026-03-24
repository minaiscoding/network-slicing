#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz
Enhanced with hexagonal coverage
"""
import numpy as np


class NodeB():
    def __init__(self, id, x, y, slices_l1, slots_per_step, n_prbs,
                 coverage_radius=500, slot_length=1e-3):
        """
        Initialize gNodeB with hexagonal coverage

        Args:
            id: gNodeB identifier
            x, y: center coordinates of the hexagon
            slices_l1: list of L1 slice objects
            slots_per_step: number of time slots per step
            n_prbs: total number of Physical Resource Blocks
            coverage_radius: radius of the circumscribed circle (distance from center to vertices)
            slot_length: duration of each time slot
        """
        self.id = id  # gNodeB identifier
        self.x = x  # x coordinate (center of hexagon)
        self.y = y  # y coordinate (center of hexagon)
        self.coverage_radius = coverage_radius  # radius of circumscribed circle
        self.slices_l1 = slices_l1
        self.n_slices_l1 = len(self.slices_l1)
        self.slots_per_step = slots_per_step
        self.n_prbs = n_prbs
        self.slot_length = slot_length

        # Pre-calculate hexagon vertices for visualization
        self.vertices = self._calculate_hexagon_vertices()

        # Calculate coverage area
        self.coverage_area = self._calculate_hexagon_area()

        self.reset()

    def _calculate_hexagon_vertices(self):
        """
        Calculate the six vertices of the regular hexagon
        Returns: list of (x, y) tuples for each vertex
        """
        vertices = []
        for i in range(6):
            # 60-degree increments (π/3 radians)
            angle_deg = 60 * i
            angle_rad = np.radians(angle_deg)

            # Regular hexagon vertices at distance coverage_radius from center
            x_vertex = self.x + self.coverage_radius * np.cos(angle_rad)
            y_vertex = self.y + self.coverage_radius * np.sin(angle_rad)
            vertices.append((x_vertex, y_vertex))

        return vertices

    def _calculate_hexagon_area(self):
        """
        Calculate the area of the regular hexagon
        Area = (3√3/2) * r² where r is the circumradius
        """
        return (3 * np.sqrt(3) / 2) * (self.coverage_radius ** 2)

    def is_point_in_coverage(self, ue_x, ue_y):
        """
        Check if a point (UE) is inside the hexagonal coverage area
        Uses ray casting algorithm for point-in-polygon

        Args:
            ue_x, ue_y: coordinates of the user equipment

        Returns:
            bool: True if UE is within coverage, False otherwise
        """
        # Fast check: if distance from center is greater than coverage_radius, definitely outside
        distance = self.distance_to_ue(ue_x, ue_y)
        if distance > self.coverage_radius:
            return False

        # Point-in-polygon test for hexagon
        # Ray casting algorithm: count intersections of horizontal ray with polygon edges
        x, y = ue_x, ue_y
        inside = False

        for i in range(len(self.vertices)):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % len(self.vertices)]

            # Check if the point is exactly on a vertex or edge
            if self._point_on_line_segment(x, y, x1, y1, x2, y2):
                return True

            # Check if the horizontal ray intersects the edge
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                inside = not inside

        return inside

    def _point_on_line_segment(self, x, y, x1, y1, x2, y2):
        """
        Check if a point lies on a line segment
        """
        # Check if point is within bounding box of segment
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)

        if x < min_x - 1e-10 or x > max_x + 1e-10 or y < min_y - 1e-10 or y > max_y + 1e-10:
            return False

        # Check collinearity using cross product
        cross_product = (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)
        if abs(cross_product) > 1e-10:
            return False

        return True

    def get_coverage_boundaries(self):
        """
        Get the bounding box of the coverage area
        Returns: (x_min, x_max, y_min, y_max)
        """
        x_coords = [v[0] for v in self.vertices]
        y_coords = [v[1] for v in self.vertices]

        return min(x_coords), max(x_coords), min(y_coords), max(y_coords)

    def get_ue_signal_strength(self, ue_x, ue_y):
        """
        Calculate signal strength for a UE based on distance
        Simple path loss model: signal strength decreases with distance

        Args:
            ue_x, ue_y: coordinates of user equipment

        Returns:
            float: normalized signal strength (0 to 1), 0 if out of coverage
        """
        if not self.is_point_in_coverage(ue_x, ue_y):
            return 0.0

        distance = self.distance_to_ue(ue_x, ue_y)

        # Simple path loss model: signal = 1 / (1 + (distance/coverage_radius)^2)
        # This gives 1 at center, ~0.5 at edge
        normalized_distance = distance / self.coverage_radius
        signal_strength = 1.0 / (1.0 + normalized_distance ** 2)

        return signal_strength

    def get_ue_snr(self, ue_x, ue_y, noise_power=1e-9):
        """
        Calculate SNR for a UE
        More realistic model with noise

        Args:
            ue_x, ue_y: coordinates of user equipment
            noise_power: noise power in watts

        Returns:
            float: SNR in dB, -inf if out of coverage
        """
        if not self.is_point_in_coverage(ue_x, ue_y):
            return -np.inf

        distance = self.distance_to_ue(ue_x, ue_y)

        # Free space path loss model
        # Assumes transmission power = 1W, frequency = 2.4GHz
        wavelength = 0.125  # 2.4GHz
        received_power = (wavelength / (4 * np.pi * max(distance, 1))) ** 2

        snr_linear = received_power / noise_power
        snr_db = 10 * np.log10(snr_linear)

        return snr_db

    def get_overlapping_coverage(self, other_nodeb):
        """
        Check if coverage areas overlap with another gNodeB
        Simple check based on distance between centers

        Args:
            other_nodeb: another NodeB object

        Returns:
            bool: True if coverage areas overlap
        """
        distance = self.distance_to_ue(other_nodeb.x, other_nodeb.y)
        # Overlap if distance < sum of radii
        return distance < (self.coverage_radius + other_nodeb.coverage_radius)

    def get_overlap_area_estimate(self, other_nodeb):
        """
        Rough estimate of overlap area with another gNodeB
        Useful for handover planning and interference management

        Returns:
            float: approximate overlap area
        """
        distance = self.distance_to_ue(other_nodeb.x, other_nodeb.y)

        if distance >= (self.coverage_radius + other_nodeb.coverage_radius):
            return 0.0  # No overlap

        if distance <= abs(self.coverage_radius - other_nodeb.coverage_radius):
            # One hexagon completely contains the other
            return min(self.coverage_area, other_nodeb.coverage_area)

        # Approximate overlap as circle-circle overlap
        # This is an approximation since we're dealing with hexagons
        r1 = self.coverage_radius
        r2 = other_nodeb.coverage_radius

        # Circle-circle overlap area formula
        part1 = r1 ** 2 * np.arccos((distance ** 2 + r1 ** 2 - r2 ** 2) / (2 * distance * r1))
        part2 = r2 ** 2 * np.arccos((distance ** 2 + r2 ** 2 - r1 ** 2) / (2 * distance * r2))
        part3 = 0.5 * np.sqrt((-distance + r1 + r2) * (distance + r1 - r2) *
                              (distance - r1 + r2) * (distance + r1 + r2))

        overlap_area = part1 + part2 - part3

        return max(0, overlap_area)

    def visualize_coverage(self, ax=None, show_slices=False, color='lightblue', alpha=0.3,
                           edge_color='blue', linewidth=2, show_sector_ids=False):
        """
        Plot the hexagonal coverage area with customizable visualization options

        Args:
            ax: matplotlib axis object (optional)
            show_slices: whether to show slice divisions (default False)
            color: face color of the hexagon (default 'lightblue')
            alpha: transparency level (default 0.3)
            edge_color: color of hexagon edges (default 'blue')
            linewidth: width of hexagon edge lines (default 2)
            show_sector_ids: whether to show sector/slice IDs (default False)
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon

            if ax is None:
                fig, ax = plt.subplots()

            # Create hexagon patch with custom parameters
            hex_patch = Polygon(self.vertices, closed=True,
                                edgecolor=edge_color if edge_color != 'none' else 'none',
                                facecolor=color, alpha=alpha, linewidth=linewidth)
            ax.add_patch(hex_patch)

            # Plot center
            marker_color = 'red' if edge_color != 'none' else 'blue'
            ax.plot(self.x, self.y, '^', color=marker_color, markersize=10,
                    label=f'gNodeB {self.id}', zorder=5)

            # Show slice divisions if requested
            if show_slices and self.n_slices_l1 > 1:
                self._draw_slice_divisions(ax)

            # Show sector IDs if requested
            if show_sector_ids:
                self._draw_sector_ids(ax)

            # Add radius indicator (only if not suppressed)
            if edge_color != 'none':
                ax.annotate(f'R={self.coverage_radius}',
                            xy=(self.x, self.y + self.coverage_radius),
                            xytext=(self.x + 10, self.y + self.coverage_radius + 10),
                            arrowprops=dict(arrowstyle='->'), fontsize=8)

            ax.set_aspect('equal')

            # Set axis limits with some margin
            x_coords = [v[0] for v in self.vertices]
            y_coords = [v[1] for v in self.vertices]
            margin = self.coverage_radius * 0.1
            ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
            ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)

            return ax

        except ImportError:
            print("Matplotlib not available for visualization")
            return None

    def _draw_slice_divisions(self, ax):
        """Draw divisions between slices in the hexagon"""
        try:
            n_slices = self.n_slices_l1
            if n_slices <= 1:
                return

            # Draw radial lines from center to divide slices
            angle_step = 360 / n_slices
            for i in range(n_slices):
                angle_rad = np.radians(i * angle_step)
                x_end = self.x + self.coverage_radius * np.cos(angle_rad)
                y_end = self.y + self.coverage_radius * np.sin(angle_rad)
                ax.plot([self.x, x_end], [self.y, y_end], 'k--', alpha=0.5, linewidth=1)
        except Exception as e:
            print(f"Error drawing slice divisions: {e}")

    def _draw_sector_ids(self, ax):
        """Draw sector/slice IDs on the hexagon"""
        try:
            n_slices = self.n_slices_l1
            angle_step = 360 / n_slices if n_slices > 0 else 360

            for i in range(n_slices):
                angle_rad = np.radians(i * angle_step + angle_step / 2)
                # Position text at 2/3 of radius
                text_radius = self.coverage_radius * 0.65
                x_text = self.x + text_radius * np.cos(angle_rad)
                y_text = self.y + text_radius * np.sin(angle_rad)
                ax.text(x_text, y_text, f'S{i}', ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='circle', facecolor='white', alpha=0.7))
        except Exception as e:
            print(f"Error drawing sector IDs: {e}")

    def reset(self):
        self.steps = 0
        for slice_l1 in self.slices_l1:
            slice_l1.reset()
        state = self.get_state()
        return state

    def get_n_variables(self):
        n_variables = 0
        for slice_l1 in self.slices_l1:
            n_variables += slice_l1.get_n_variables()
        return n_variables

    def reset_info(self):
        ''' Reset the info of the l1 slices for SLA assessment'''
        for l1 in self.slices_l1:
            l1.reset_info()

    def slot(self):
        ''' runs the system just for one time-slot '''
        for slice_l1 in self.slices_l1:
            slice_l1.slot()

    def get_state(self):
        state = np.array([self.x, self.y], dtype=float)
        for l1 in self.slices_l1:
            state = np.concatenate((state, l1.get_state()), axis=None)
        return state

    def get_info(self, violations=0, SLA_labels=0):
        info = {'l1_info': [l1.get_info() for l1 in self.slices_l1], 'SLA_labels': SLA_labels, \
                'violations': violations, 'n_prbs': [l1.n_prbs for l1 in self.slices_l1]}
        return info

    def distance_to_ue(self, ue_x, ue_y):
        '''Calculate distance from this gNodeB to a UE'''
        return np.sqrt((self.x - ue_x) ** 2 + (self.y - ue_y) ** 2)

    def __repr__(self):
        return f'NodeB {self.id} at ({self.x:.2f}, {self.y:.2f}), radius={self.coverage_radius}'

    def compute_reward(self):
        '''checks if the SLA is fulfilled for each slice'''
        SLA_labels = np.zeros(self.n_slices_l1, dtype=int)
        violations = np.zeros(self.n_slices_l1, dtype=int)
        for i, l1 in enumerate(self.slices_l1):
            SLA_labels[i], violations[i] = l1.compute_reward()
        return SLA_labels, violations

    def step(self, action):
        '''
        move a step forward using the selected action
        each step consists of a number of time slots
        '''
        self.reset_info()

        if len(action) != len(self.slices_l1):
            print('The action must contain as many elements as slices!')
            return self.get_state, self.get_info()

        # configure slices
        i_prb = 0
        for slice_l1, prbs in zip(self.slices_l1, action):
            slice_l1.set_prbs(i_prb, prbs)
            i_prb += prbs

        # run a step
        for _ in range(self.slots_per_step):
            self.slot()

        # get the node state
        state = self.get_state()

        # check the SLAs of each slice_l1
        SLA_labels, violations = self.compute_reward()

        # the info is a dict
        info = self.get_info(SLA_labels=SLA_labels, violations=violations)

        self.steps += 1

        return state, info


if __name__ == '__main__':
    # Import required modules for testing
    from numpy.random import default_rng
    from slice_l1 import SliceL1eMBB, SliceL1mMTC
    from slice_ran import SliceRANeMBB, SliceRANmMTC
    from channel_models import SINRSelectiveFading, MCSCodeset
    from schedulers import ProportionalFair
    from itertools import count
    import matplotlib.pyplot as plt

    # Initialize random number generator
    rng = default_rng(seed=42)

    # Create basic parameters for slices
    CBR_description = {'lambda': 2.0 / 60.0, 't_mean': 30.0, 'bit_rate': 500000}
    VBR_description = {'lambda': 5.0 / 60.0, 't_mean': 30.0, 'p_size': 1000, 'b_size': 500, 'b_rate': 1}
    SLA_embb = {'cbr_th': 10e6, 'cbr_prb': 20, 'cbr_queue': 10e4, 'vbr_th': 15e6, 'vbr_prb': 30, 'vbr_queue': 15e4}
    state_variables_embb = ['cbr_traffic', 'cbr_th', 'cbr_prb', 'cbr_queue', 'cbr_snr', 'vbr_traffic', 'vbr_th',
                            'vbr_prb', 'vbr_queue', 'vbr_snr']
    norm_const_embb = {'cbr_traffic': 5e6 * 0.05, 'cbr_th': 10e6 * 0.05, 'cbr_prb': 25 * 50, 'cbr_queue': 10e4 * 50,
                       'cbr_snr': 35 * 50,
                       'vbr_traffic': 5e6 * 0.05, 'vbr_th': 10e6 * 0.05, 'vbr_prb': 35 * 50, 'vbr_queue': 10e4 * 50,
                       'vbr_snr': 35 * 50}

    # Create channel model and scheduler
    snr_generator = SINRSelectiveFading(rng, 'macro_cell_urban_2GHz', n_prbs=100)
    mcs_codeset = MCSCodeset()
    scheduler = ProportionalFair(mcs_codeset)
    user_counter = count()

    # Create eMBB slice
    slice_ran_embb = SliceRANeMBB(rng, user_counter, 0, SLA_embb, CBR_description, VBR_description,
                                  state_variables_embb, norm_const_embb, 50)
    slice_l1_embb = SliceL1eMBB(rng, snr_generator, 50, [slice_ran_embb], scheduler)

    # Create mMTC slice
    MTC_description = {'n_devices': 1000, 'repetition_set': [2, 4, 8, 16, 32, 64, 128],
                       'period_set': [1000, 50000, 10000, 15000, 20000, 25000, 50000, 100000]}
    state_variables_mmtc = ['devices', 'avg_rep', 'delay']
    SLA_mmtc = {'delay': 300}
    norm_const_mmtc = {'devices': 100 * 50, 'avg_rep': 100 * 50, 'delay': 100 * 50}
    slice_ran_mmtc = SliceRANmMTC(rng, 1, SLA_mmtc, MTC_description, state_variables_mmtc, norm_const_mmtc, 50)
    slice_l1_mmtc = SliceL1mMTC(50, [slice_ran_mmtc])

    # Create slices list
    my_slices = [slice_l1_embb, slice_l1_mmtc]

    # Create gNodeBs
    gnb1 = NodeB(id=1, x=100, y=100, slices_l1=my_slices, slots_per_step=10, n_prbs=100, coverage_radius=500)

    slice_ran_embb2 = SliceRANeMBB(rng, user_counter, 2, SLA_embb, CBR_description, VBR_description,
                                   state_variables_embb, norm_const_embb, 50)
    slice_l1_embb2 = SliceL1eMBB(rng, snr_generator, 50, [slice_ran_embb2], scheduler)
    gnb2 = NodeB(id=2, x=400, y=300, slices_l1=[slice_l1_embb2], slots_per_step=10, n_prbs=100, coverage_radius=500)

    # Create test UEs
    test_ues = [
        (150, 200, 1),  # UE1 inside first cell
        (400, 300, 2),  # UE2 inside second cell
        (250, 250, 3),  # UE3 in overlap region
        (600, 600, 4),  # UE4 outside both cells
    ]

    # OPTIMIZED VISUALIZATION - Single figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Subplot 1: Basic coverage with UEs
    ax1 = fig.add_subplot(gs[0, 0])
    gnb1.visualize_coverage(ax=ax1, show_slices=False, color='lightblue', alpha=0.3, edge_color='blue')
    gnb2.visualize_coverage(ax=ax1, show_slices=False, color='lightgreen', alpha=0.3, edge_color='green')

    # Add UEs to subplot 1
    for ue_x, ue_y, ue_id in test_ues:
        in_coverage = gnb1.is_point_in_coverage(ue_x, ue_y) or gnb2.is_point_in_coverage(ue_x, ue_y)
        color = 'green' if in_coverage else 'red'
        ax1.plot(ue_x, ue_y, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=1)
        ax1.annotate(f'UE{ue_id}', (ue_x, ue_y), xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax1.set_title('Basic Coverage with UE Positions', fontweight='bold')
    ax1.legend(['gNodeB 1', 'gNodeB 2'], loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Subplot 2: Slice distribution view
    ax2 = fig.add_subplot(gs[0, 1])
    gnb1.visualize_coverage(ax=ax2, show_slices=True, color='lightblue', alpha=0.2,
                            edge_color='blue', show_sector_ids=True)
    gnb2.visualize_coverage(ax=ax2, show_slices=True, color='lightgreen', alpha=0.2,
                            edge_color='green', show_sector_ids=True)
    ax2.set_title('Slice Distribution & Sector View', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Subplot 3: Signal strength heatmap for first gNodeB
    ax3 = fig.add_subplot(gs[1, 0])

    # Create heatmap data
    x_min, x_max, y_min, y_max = gnb1.get_coverage_boundaries()
    x = np.linspace(x_min - 50, x_max + 50, 30)
    y = np.linspace(y_min - 50, y_max + 50, 30)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(len(x)):
        for j in range(len(y)):
            if gnb1.is_point_in_coverage(X[j, i], Y[j, i]):
                Z[j, i] = gnb1.get_ue_signal_strength(X[j, i], Y[j, i])

    # Plot heatmap
    contour = ax3.contourf(X, Y, Z, levels=20, cmap='YlOrRd', alpha=0.7)
    plt.colorbar(contour, ax=ax3, label='Signal Strength')

    # Overlay hexagon
    gnb1.visualize_coverage(ax=ax3, show_slices=False, color='none', edge_color='blue', alpha=0)
    ax3.plot(gnb1.x, gnb1.y, 'b^', markersize=10, label='gNodeB 1')
    ax3.set_title('Signal Strength Heatmap', fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend()

    # Subplot 4: Network view with overlap analysis
    ax4 = fig.add_subplot(gs[1, 1])

    # Plot both cells
    gnb1.visualize_coverage(ax=ax4, show_slices=False, color='lightblue', alpha=0.3,
                            edge_color='blue', linewidth=2)
    gnb2.visualize_coverage(ax=ax4, show_slices=False, color='lightgreen', alpha=0.3,
                            edge_color='green', linewidth=2)

    # Highlight overlap zone
    if gnb1.get_overlapping_coverage(gnb2):
        overlap = gnb1.get_overlap_area_estimate(gnb2)
        mid_x = (gnb1.x + gnb2.x) / 2
        mid_y = (gnb1.y + gnb2.y) / 2

        # Create overlap indicator
        ax4.annotate(f'Overlap Area:\n{overlap:.0f} m²',
                     xy=(mid_x, mid_y),
                     xytext=(mid_x + 100, mid_y + 100),
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='red', linewidth=2))

        # Add distance indicator
        distance = gnb1.distance_to_ue(gnb2.x, gnb2.y)
        ax4.plot([gnb1.x, gnb2.x], [gnb1.y, gnb2.y], 'r--', linewidth=2, alpha=0.5)
        mid_dist_x = (gnb1.x + gnb2.x) / 2
        mid_dist_y = (gnb1.y + gnb2.y) / 2
        ax4.text(mid_dist_x, mid_dist_y - 50, f'Distance: {distance:.0f}m',
                 ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax4.set_title('Network View with Overlap Analysis', fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(['gNodeB 1', 'gNodeB 2'], loc='upper right')

    # Main title
    fig.suptitle('5G gNodeB Coverage Analysis - Hexagonal Cells', fontsize=16, fontweight='bold', y=0.98)

    # Save the figure
    plt.tight_layout()
    plt.savefig('gnb_coverage_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("=" * 60)
    print("OPTIMIZED VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Coverage Analysis Summary:")
    print(f"  - gNodeB 1: Center ({gnb1.x}, {gnb1.y}), Radius {gnb1.coverage_radius}m, Area {gnb1.coverage_area:.0f}m²")
    print(f"  - gNodeB 2: Center ({gnb2.x}, {gnb2.y}), Radius {gnb2.coverage_radius}m, Area {gnb2.coverage_area:.0f}m²")
    print(f"  - Distance between centers: {gnb1.distance_to_ue(gnb2.x, gnb2.y):.2f}m")
    print(f"  - Overlap exists: {gnb1.get_overlapping_coverage(gnb2)}")
    print(f"  - Overlap area estimate: {gnb1.get_overlap_area_estimate(gnb2):.2f}m²")
    print("=" * 60)
    print("Figure saved as 'gnb_coverage_optimized.png'")