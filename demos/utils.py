# Copyright (C) 2026 Antonio Baiano Svizzero and Jørgen S. Dokken
#
# This file is part of FEniCSx_vib_Shells (https://github.com/bayswiss/fenicsx_vib_shells)
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import numpy.typing as npt
from dolfinx import geometry, default_scalar_type
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class FieldProbe:
    def __init__(self, domain, position):
        """Init probe(s).

        Args:
            domain: Mesh to probe on.
            position: Probe location(s). For multiple points, ordered as
                ``(p0_x, p1_x, ..., p0_y, p1_y, ..., p0_z, p1_z, ...)``.
        """
        self._domain = domain
        self._position = np.asarray(
            position, dtype=domain.geometry.x.dtype
        ).reshape(3, -1)
        self._local_cells, self._local_position = self._compute_local_probes()

    def _compute_local_probes(
        self,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.floating]]:
        """Find local cell ownership on a distributed mesh."""
        points = self._position.T
        bb_tree = geometry.bb_tree(self._domain, self._domain.topology.dim)

        cells = []
        points_on_proc = []

        cell_candidates = geometry.compute_collisions_points(bb_tree, points)
        colliding_cells = geometry.compute_colliding_cells(
            self._domain, cell_candidates, points
        )

        for i, point in enumerate(points):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])

        return np.asarray(cells, dtype=np.int32), np.asarray(
            points_on_proc, dtype=self._domain.geometry.x.dtype
        )

    def sample(
        self,
        uh,
        recompute_collisions: bool = False,
    ) -> npt.NDArray[np.complexfloating]:
        """Eval uh at probe points. Shape matches uh's value shape."""
        if recompute_collisions:
            self._local_cells, self._local_position = self._compute_local_probes()
        if len(self._local_cells) > 0:
            return uh.eval(self._local_position, self._local_cells)
        # No local probes: empty array, value shape matching uh
        vshape = uh.function_space.value_shape
        return np.zeros((0, *vshape) if vshape else (0,), dtype=default_scalar_type)


def plot_complex_spectra(x_axis, p_spectra_list, labels=None, title=None, plot_db=False):
    """
    Plot amplitude and phase of one or more complex spectra.

    Arguments:
    - p_spectra_list: 1D complex array, list of 1D arrays, or 2D array (rows = signals).
    - x_axis: x-axis values. Defaults to indices.
    - labels: legend names.
    - title: title for the top plot.
    - plot_db: amplitude in dB if True.
    """
    # Standardize to list
    if isinstance(p_spectra_list, np.ndarray) and p_spectra_list.ndim == 1:
        p_spectra_list = [p_spectra_list]
    elif isinstance(p_spectra_list, np.ndarray) and p_spectra_list.ndim == 2:
        p_spectra_list = list(p_spectra_list)

    if x_axis is None:
        x_axis = np.arange(len(p_spectra_list[0]))
        x_label = 'Index'
    else:
        x_label = 'Frequency'

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    ax_amp   = fig.add_subplot(gs[0:2, :])
    ax_phase = fig.add_subplot(gs[2, :])

    for i, p_complex in enumerate(p_spectra_list):
        amplitude = np.abs(p_complex)
        if plot_db:
            amplitude = 20 * np.log10(np.maximum(amplitude, 1e-12))
        phase = np.angle(p_complex)

        current_label = (
            labels[i] if (labels is not None and i < len(labels)) else f'Signal {i+1}'
        )

        ax_amp.plot(x_axis, amplitude, linewidth=1.5, label=current_label)
        ax_phase.plot(x_axis, phase, linewidth=1.5, label=current_label, alpha=0.8)

    # Amplitude
    ax_amp.set_title(title, fontsize=14, fontweight='bold')
    amp_ylabel = 'Amplitude (dB)' if plot_db else 'Amplitude (Linear)'
    ax_amp.set_ylabel(amp_ylabel, fontsize=12)
    ax_amp.grid(True, linestyle='--', alpha=0.7)
    ax_amp.set_xlim([min(x_axis), max(x_axis)])
    ax_amp.tick_params(labelbottom=False)
    ax_amp.legend()

    # Phase
    ax_phase.set_xlabel(x_label, fontsize=12)
    ax_phase.set_ylabel('Phase (Rad)', fontsize=12)
    ax_phase.grid(True, linestyle='--', alpha=0.7)
    ax_phase.set_xlim([min(x_axis), max(x_axis)])
    ax_phase.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax_phase.set_yticklabels(['-$\pi$', '-$\pi/2$', '0', '$\pi/2$', '$\pi$'])

    plt.tight_layout()
    plt.show()

    return fig, (ax_amp, ax_phase)