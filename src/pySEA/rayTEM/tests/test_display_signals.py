"""Tests for the derived display Signals: ray diagram, beam envelope, wave cross-section.

Each view is a calibrated sea-eco Signal built from a stored propagation
result on ``basic_column``, carrying its section, element and plane markers as
PLOT-8 annotations. The tests pin dimensions and units, aperture ghosting,
marker counts against the column's own description, the section-level views,
``.sea`` round trips, the no-sea-eco fallback, that no view ever copies the
column, and that ``show`` still draws the cross-section the old code drew.
See ``notes/eric/TODO_ACTIVE_2026-10-12_live-model-plots.md``.
"""

import numpy as np
import pytest

from pySEA.rayTEM import seashells
from pySEA.rayTEM.elements import columnByName
from pySEA.rayTEM.microscopes.basic_column import build_basic_column
from pySEA.rayTEM.seashells import read_display_signal


def _kinds(markers):
	"""Count marker records by kind.

	Parameters
	----------
	markers : Sequence[Mapping]
		Records from :func:`seashells.read_display_signal`.

	Returns
	-------
	dict
		``{kind: count}``.
	"""
	counts = {}
	for marker in markers:
		counts[marker["kind"]] = counts.get(marker["kind"], 0) + 1
	return counts


def _named_elements(positions):
	"""The named-element entries a display Signal marks (blank names are not).

	Parameters
	----------
	positions : Mapping
		A ``named_positions`` dict.

	Returns
	-------
	dict
		The entries with a non-empty name.
	"""
	return {k: v for k, v in positions.items() if k}


def _stored_planes(scope, axis="x"):
	"""Plane positions plot2D finds in a column's stored rays, per family.

	Parameters
	----------
	scope : Microscope
		A column with stored rays.
	axis : str, optional
		Axis to search, by default ``"x"``.

	Returns
	-------
	dict
		``{"diff": ndarray, "image": ndarray}`` in metres.
	"""
	from pySEA.rayTEM.postprocessing import findPlanes, zFromFractional
	zs = np.asarray(scope.rays)[:, 0, columnByName("z")]
	found = findPlanes(scope.rays, axis=axis)[axis]
	return {family: np.array([zFromFractional(zs, min(f, len(zs) - 1 - 1e-9))
							  for f in found[family]["z"]]) for family in ("diff", "image")}


def _blocked_column():
	"""The basic column with its condenser aperture closed to 2 µm, rays traced.

	The stock 10 µm aperture passes every ray of the 5x5 grid, so ghosting
	needs a tighter one to have something to show.

	Returns
	-------
	Microscope
		Propagated column.
	"""
	scope = build_basic_column()
	scope["CA"].radius = 2e-6
	scope.propagate_ray()
	return scope


# --- ray diagram --------------------------------------------------------------

def test_ray_diagram_dims_units_and_values():
	"""The Microscope ray diagram is [ray, plane_z] in metres with the stored z."""
	scope = build_basic_column()
	scope.propagate_ray()
	sig = scope.ray_diagram_signal()
	rays = np.asarray(scope.rays)
	assert sig.dimensions.get_names() == ["ray", "plane_z"]
	assert sig.data.shape == (rays.shape[1], rays.shape[0])
	data, axes, markers, meta = read_display_signal(sig)
	assert [a[2] for a in axes] == ["", "m"]
	assert np.array_equal(axes[1][1], rays[:, 0, columnByName("z")])		# repeats kept
	assert sig.signal_quantities.dimensions[0].units == "m"
	assert meta["kind"] == "ray-diagram" and meta["axis"] == "x" and meta["frame"] == "rotating"
	assert meta["source"] == "basic column" and sig.name == "ray diagram (x)"


def test_ray_diagram_frames_match_plot2D_and_stored_columns():
	"""'rotating' converts like plot2D does; 'lab' is the stored column."""
	scope = build_basic_column()
	scope.propagate_ray()
	rotated = np.asarray(scope.rays.convert_to_rotating_reference_frame())
	assert np.allclose(scope.ray_diagram_signal().data, rotated[:, :, columnByName("x")].T)
	assert np.allclose(scope.ray_diagram_signal(axis="y").data, rotated[:, :, columnByName("y")].T)
	lab = scope.ray_diagram_signal(axis="y", frame="lab")
	assert np.array_equal(lab.data, np.asarray(scope.rays)[:, :, columnByName("y")].T)
	assert not np.allclose(lab.data, scope.ray_diagram_signal(axis="y").data)	# the column rotates
	with pytest.raises(ValueError, match="frame"):
		scope.ray_diagram_signal(frame="larmor")
	with pytest.raises(ValueError, match="axis"):
		scope.ray_diagram_signal(axis="z")


def test_ray_diagram_z_monotone_within_each_section():
	"""plane_z never decreases inside a section, and each section starts at its position."""
	scope = build_basic_column()
	scope.propagate_ray()
	z = read_display_signal(scope.ray_diagram_signal())[1][1][1]
	start = 0
	for sec in scope.sections:
		n = len(sec.rays)
		chunk = z[start:start + n]
		assert np.all(np.diff(chunk) >= 0), sec.name
		assert np.isclose(chunk[0], sec.position) and np.isclose(chunk[-1], sec.position + sec.length)
		start += n
	assert start == len(z)


def test_ray_diagram_ghosts_rays_an_aperture_blocked():
	"""A ray with zero intensity is NaN from the blocking plane on; live rays stay finite."""
	scope = _blocked_column()
	data = scope.ray_diagram_signal().data
	dead = np.asarray(scope.I).T <= 0
	assert dead.any() and not dead.all(axis=0).any()		# some, never all, rays blocked
	assert np.array_equal(np.isnan(data), dead)			# exactly plot2D's ghosting mask
	first = dead.any(axis=0).argmax()						# the aperture's plane
	assert np.allclose(scope.rays[first, 0, columnByName("z")], scope.named_positions["CA"])
	blocked = dead[:, first]
	assert np.isnan(data[blocked, first:]).all() and np.isfinite(data[:, :first]).all()
	assert np.isfinite(data[~blocked]).all()


def test_ray_diagram_markers_match_sections_elements_and_planes():
	"""One Span per section, one Line per named element and per plane plot2D finds."""
	scope = build_basic_column()
	scope.propagate_ray()
	sig = scope.ray_diagram_signal()
	from pySEA.sea_eco.annotations import get_display_annotations
	annotations = list(get_display_annotations(sig)[0])
	planes = _stored_planes(scope)
	# this source launches findPlanes' reference pairs, so the stored-trace
	# planes are the column entrance's conjugates
	for family in ("diff", "image"):
		assert np.allclose(planes[family], scope.conjugate_planes()[family], atol=1e-9)
	elements = _named_elements(scope.named_positions)
	assert [a.annotation_type for a in annotations].count("Span") == len(scope.named_sections)
	assert len(annotations) == (len(scope.named_sections) + len(elements)
								+ len(planes["diff"]) + len(planes["image"]))
	markers = read_display_signal(sig)[2]
	assert _kinds(markers) == {"section": len(scope.named_sections), "element": len(elements),
							   "diff": len(planes["diff"]), "image": len(planes["image"])}
	spans = [m for m in markers if m["kind"] == "section"]
	assert [(m["label"], [m["z0"], m["z1"]]) for m in spans] == list(scope.named_sections.items())
	assert {m["label"]: m["z"] for m in markers if m["kind"] == "element"} == elements
	assert np.allclose([m["z"] for m in markers if m["kind"] == "image"], planes["image"])
	# labelled like plot2D: family, z, magnification and rotation
	assert all(m["label"].startswith("image @ z=") and ", M=" in m["label"] and ", R=" in m["label"]
			   for m in markers if m["kind"] == "image")
	# lines span the plotted values; every annotation is calibrated and named raytem.<kind>.
	lines = [a for a in annotations if a.annotation_type == "Line"]
	assert np.isclose(lines[0].coordinates["y0"], np.nanmin(sig.data))
	assert np.isclose(lines[0].coordinates["y1"], np.nanmax(sig.data))
	# PLOT-8: the column's marks are overlays drawn for the plane-z axis.
	assert all(a.coordinate_reference == {"kind": "calibrated", "x_dim": "plane_z"}
			   for a in annotations)
	assert all(a.role == "overlay" for a in annotations)
	assert all(a.name.startswith("raytem.") for a in annotations)


def test_section_ray_diagram_marks_its_own_span_and_elements():
	"""A section's view covers its own planes, span and elements, in absolute z."""
	scope = build_basic_column()
	scope.propagate_ray()
	sec = scope["O"]
	sig = sec.ray_diagram_signal()
	data, axes, markers, meta = read_display_signal(sig)
	assert data.shape == (np.asarray(sec.rays).shape[1], len(sec.rays))
	assert np.array_equal(axes[1][1], np.asarray(sec.rays)[:, 0, columnByName("z")])
	assert meta["source"] == "O"
	assert _kinds(markers) == {"section": 1, "element": len(_named_elements(sec.named_positions))}
	assert np.isclose(markers[0]["z0"], sec.position)
	assert np.isclose(markers[0]["z1"], sec.position + sec.length)
	whole = scope.named_positions
	for m in markers[1:]:
		assert np.isclose(m["z"], whole[m["label"]])		# same place the column marks it


def test_display_signals_refuse_to_propagate():
	"""Nothing stored -> an actionable RuntimeError, and nothing gets propagated."""
	scope = build_basic_column()
	with pytest.raises(RuntimeError, match="propagate_ray"):
		scope.ray_diagram_signal()
	with pytest.raises(RuntimeError, match="propagate_ray"):
		scope["C"].ray_diagram_signal()
	assert scope.rays is None and scope["C"].rays is None


def test_ray_diagram_survives_sea_round_trips(tmp_path):
	"""The Signal's markers serialize with it; a reloaded column still derives it."""
	from pySEA.sea_eco.io import load_sea
	scope = _blocked_column()
	sig = scope.ray_diagram_signal()
	path = str(tmp_path / "ray_diagram.sea")
	sig.to_sea(path)
	data, axes, markers, meta = read_display_signal(load_sea(path))
	expected = read_display_signal(sig)
	assert np.array_equal(data, expected[0], equal_nan=True)
	assert markers == expected[2] and meta["kind"] == "ray-diagram"
	# a column reloaded from .sea stores plain arrays, not Rays
	from pySEA.rayTEM.assemblies import Microscope
	scope_path = str(tmp_path / "column.sea")
	scope.to_sea(scope_path)
	reloaded = Microscope() ; reloaded.from_sea(scope_path)
	assert np.array_equal(reloaded.ray_diagram_signal().data, sig.data, equal_nan=True)


def test_ray_diagram_falls_back_without_sea_eco(monkeypatch):
	"""Without sea_eco the same data and markers come back in the fallback container."""
	scope = _blocked_column()
	expected = read_display_signal(scope.ray_diagram_signal())
	monkeypatch.setattr(seashells, "sea_available", False)
	with pytest.warns(UserWarning, match="sea_eco is not installed"):
		fallback = scope.ray_diagram_signal()
	assert isinstance(fallback, seashells._DisplaySignal)
	data, axes, markers, meta = read_display_signal(fallback)
	assert np.array_equal(data, expected[0], equal_nan=True)
	assert [a[0] for a in axes] == ["ray", "plane_z"] and np.array_equal(axes[1][1], expected[1][1][1])
	assert markers == expected[2] and meta == expected[3]


# --- beam envelope ------------------------------------------------------------

def test_envelope_signal_rows_units_and_z():
	"""[quantity, z]: sigma_x/y (m), sigma_xt/yt (rad), on the covariance's own z."""
	from pySEA.rayTEM.postprocessing import beam_widths
	scope = build_basic_column()
	scope.propagate_moments()
	sig = scope.envelope_signal()
	data, axes, markers, meta = read_display_signal(sig)
	assert sig.dimensions.get_names() == ["quantity", "z"]
	assert list(axes[0][1]) == ["sigma_x", "sigma_y", "sigma_xt", "sigma_yt"]
	assert meta["kind"] == "beam-envelope" and list(meta["quantity_units"]) == ["m", "m", "rad", "rad"]
	assert axes[1][2] == "m" and np.array_equal(axes[1][1], scope.mu[:, columnByName("z")])
	cov = scope.covariance_matrix.data
	assert data.shape == (4, len(cov))
	assert np.allclose(data[:2], beam_widths(scope.covariance_matrix).T)	# the x/y rows
	for row, name in zip(data, ("x", "y", "xt", "yt")):
		assert np.allclose(row, np.sqrt(cov[:, columnByName(name), columnByName(name)]))
	assert sig.data[2, 0] == pytest.approx(1e-4)		# the source's RMS divergence
	assert sig.dimensions["quantity"].values[3] == "sigma_yt"


def test_envelope_signal_carries_the_ray_diagram_section_and_element_markers():
	"""The ray diagram's section/element markers; no planes, even with rays stored."""
	scope = build_basic_column()
	scope.propagate_ray()
	scope.propagate_moments()
	envelope = read_display_signal(scope.envelope_signal())[2]
	rays = read_display_signal(scope.ray_diagram_signal())[2]
	assert envelope == [m for m in rays if m["kind"] in ("section", "element")]
	assert _kinds(envelope) == {"section": len(scope.named_sections),
								"element": len(_named_elements(scope.named_positions))}


def test_section_envelope_signal_and_errors():
	"""A section's envelope covers its own planes; nothing stored raises."""
	scope = build_basic_column()
	with pytest.raises(RuntimeError, match="propagate_moments"):
		scope.envelope_signal()
	with pytest.raises(RuntimeError, match="propagate_moments"):
		scope["P"].envelope_signal()
	scope.propagate_moments()
	sec = scope["P"]
	data, axes, markers, meta = read_display_signal(sec.envelope_signal())
	assert data.shape == (4, len(sec.mu)) and meta["source"] == "P"
	assert np.array_equal(axes[1][1], sec.mu[:, columnByName("z")])
	assert _kinds(markers) == {"section": 1, "element": len(_named_elements(sec.named_positions))}
	assert np.isclose(markers[0]["z0"], sec.position)


# --- scaled-wave cross-section ------------------------------------------------

@pytest.fixture(scope="module")
def hybrid_column():
	"""The basic column after one hybrid scaled-wave run (shared, read-only).

	Returns
	-------
	Microscope
		Column with ``_wave_scaled_planes`` and ``crossovers`` stored.
	"""
	scope = build_basic_column()
	scope.propagate_wave(mode="hybrid")
	return scope


def _cross_section_before_refactor(planes):
	"""The cross-section exactly as the pre-2026-10-12 ``_scaled_wave_cross_section`` drew it.

	A frozen copy of the old computation, kept so the display Signal -- and
	the figure ``show`` now draws from it -- can be checked against what the
	figure used to plot, whatever the wave engine produces.

	Parameters
	----------
	planes : Sequence
		Per-plane scaled wavefields.

	Returns
	-------
	tuple
		``(z_edges_mm, x_edges_um, plotted)``: the pcolormesh arguments,
		``plotted`` being the ``prof.T`` array it was given.
	"""
	from pySEA.rayTEM.seashells import read_scaled_wavefield
	from pySEA.rayTEM.waveoptics import reconstruct_physical_wave
	recon = []
	for p in planes:
		U, dxi, deta, lam, s, R, tau, z = read_scaled_wavefield(p)
		psi, dx, dy = reconstruct_physical_wave(U, dxi, deta, lam, s, R)
		recon.append((z if z is not None else 0.0, psi, dx))
	recon.sort(key=lambda r: r[0])
	zs = np.array([r[0] for r in recon])
	z_edges = np.concatenate([[zs[0] - 1e-4], (zs[:-1] + zs[1:]) / 2, [zs[-1] + 1e-4]]) * 1e3
	n = recon[0][1].shape[1]
	half = max(abs(r[2]) * n / 2 for r in recon)
	x_common = np.linspace(-half, half, 600)
	prof = np.zeros((len(recon), x_common.size))
	for i, (z, psi, dx) in enumerate(recon):
		x = (np.arange(n) - n // 2) * dx
		row = np.abs(psi[psi.shape[0] // 2, :])
		prof[i] = np.interp(x_common, x, row / row.max(), left=0, right=0)
	x_edges = np.linspace(-half, half, x_common.size + 1) * 1e6
	return z_edges, x_edges, prof.T


def _shown_cross_section(scope, **kwargs):
	"""Run ``show(kind='wave-hybrid')`` headless and return what it drew.

	Parameters
	----------
	scope : Microscope
		Column to show (its stored run is reused).
	**kwargs
		Passed to :meth:`Microscope.show`.

	Returns
	-------
	tuple
		``(mesh_array, mesh_coordinates, line_z_mm, texts)``.
	"""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	scope.show(kind="wave-hybrid", plt_ax=ax, regenerate=False, **kwargs)
	mesh = ax.collections[0]
	shown = (np.asarray(mesh.get_array()), np.asarray(mesh.get_coordinates()),
			 [float(line.get_xdata()[0]) for line in ax.lines], [t.get_text() for t in ax.texts])
	plt.close(fig)
	return shown


def test_wave_cross_section_signal_dims_units_and_values(hybrid_column):
	"""[x, z], x uniform and symmetric in m, z ascending in m, values = the old figure's array."""
	sig = hybrid_column.wave_cross_section_signal()
	data, axes, markers, meta = read_display_signal(sig)
	assert sig.dimensions.get_names() == ["x", "z"] and sig.signal_type == "Image"
	assert [a[2] for a in axes] == ["m", "m"] and meta["kind"] == "wave-cross-section"
	x, z = axes[0][1], axes[1][1]
	assert data.shape == (600, len(hybrid_column._wave_scaled_planes)) == (x.size, z.size)
	assert np.all(np.diff(z) >= 0) and np.isclose(x[0], -x[-1]) and np.allclose(np.diff(x), x[1] - x[0])
	z_edges, x_edges, plotted = _cross_section_before_refactor(hybrid_column._wave_scaled_planes)
	assert np.array_equal(data, plotted)
	# normalized to each plane's peak BEFORE resampling, so nothing exceeds 1
	# (a narrow focus undersampled by the common grid need not reach it)
	assert data.min() >= 0 and data.max() <= 1 + 1e-12 and np.isclose(data.max(), 1.0, atol=1e-3)


def test_show_draws_the_cross_section_unchanged(hybrid_column):
	"""show() plots the Signal's array, on the same cell edges the old code used."""
	array, coords, line_z, texts = _shown_cross_section(hybrid_column)
	z_edges, x_edges, plotted = _cross_section_before_refactor(hybrid_column._wave_scaled_planes)
	assert np.array_equal(array, plotted)
	assert np.array_equal(array, hybrid_column.wave_cross_section_signal().data)
	assert np.array_equal(coords[0, :, 0], z_edges) and np.array_equal(coords[:, 0, 1], x_edges)
	# overlays: every named element, every crossover, every image plane -- in that order
	elements = _named_elements(hybrid_column.named_positions)
	images = hybrid_column.image_planes["x"]
	expected = ([z * 1e3 for z in elements.values()] + [z * 1e3 for z in hybrid_column.crossovers]
				+ [z * 1e3 for z in images])
	assert np.allclose(line_z, expected)
	# the run's logged image planes are where the old figure put them (it traced them)
	assert np.allclose(images, hybrid_column.conjugate_planes(axis="x")["image"], rtol=0, atol=1e-12)
	assert texts == list(elements) + ["crossover"] * len(hybrid_column.crossovers) + ["image"] * len(images)


def test_wave_cross_section_markers(hybrid_column):
	"""Element, crossover and image-plane Lines; conjugates=False drops the image planes."""
	from pySEA.sea_eco.annotations import get_display_annotations
	sig = hybrid_column.wave_cross_section_signal()
	markers = read_display_signal(sig)[2]
	images = hybrid_column.image_planes["x"]
	assert _kinds(markers) == {"element": len(_named_elements(hybrid_column.named_positions)),
							   "crossover": len(hybrid_column.crossovers), "image": len(images)}
	lines = list(get_display_annotations(sig)[0])
	assert {a.annotation_type for a in lines} == {"Line"}
	x = read_display_signal(sig)[1][0][1]
	assert all((a.coordinates["y0"], a.coordinates["y1"]) == (x[0], x[-1]) for a in lines)
	without = read_display_signal(hybrid_column.wave_cross_section_signal(conjugates=False))[2]
	assert "image" not in _kinds(without) and len(without) == len(markers) - len(images)
	assert len(_shown_cross_section(hybrid_column, conjugates=False)[2]) == len(without)


def test_wave_cross_section_requires_a_scaled_run_and_falls_back(hybrid_column, monkeypatch):
	"""No stored planes raises; without sea_eco the fallback draws the identical figure."""
	with pytest.raises(RuntimeError, match="propagate_wave"):
		build_basic_column().wave_cross_section_signal()
	expected = read_display_signal(hybrid_column.wave_cross_section_signal())
	shown = _shown_cross_section(hybrid_column)
	monkeypatch.setattr(seashells, "sea_available", False)
	with pytest.warns(UserWarning, match="sea_eco is not installed"):
		fallback = hybrid_column.wave_cross_section_signal()
	assert isinstance(fallback, seashells._DisplaySignal)
	data, axes, markers, meta = read_display_signal(fallback)
	assert np.array_equal(data, expected[0]) and markers == expected[2] and meta == expected[3]
	with pytest.warns(UserWarning):
		fallback_shown = _shown_cross_section(hybrid_column)
	assert np.array_equal(fallback_shown[0], shown[0]) and np.array_equal(fallback_shown[1], shown[1])
	assert fallback_shown[2:] == shown[2:]


# --- never copy the column ----------------------------------------------------

def test_display_signals_never_copy_the_column(hybrid_column, monkeypatch):
	"""A column that refuses deepcopy -- as one bound to a live backend does -- still derives every view.

	A hosted sea-pearl model holds its backend, whose transports (sockets,
	locks) cannot be deep-copied; a display Signal must come from the stored
	result alone, with no reference trace on a copy.
	"""
	import copy
	import threading
	scope = build_basic_column()
	scope.propagate_ray()
	scope.propagate_moments()
	for column in (scope, hybrid_column):
		monkeypatch.setattr(column, "_backend_lock", threading.Lock(), raising=False)
		with pytest.raises(TypeError):
			copy.deepcopy(column)
	assert read_display_signal(scope.ray_diagram_signal())[2]
	assert read_display_signal(scope.ray_diagram_signal(axis="y", frame="lab"))[2]
	assert read_display_signal(scope.envelope_signal())[2]
	assert read_display_signal(scope["O"].ray_diagram_signal())[2]
	assert read_display_signal(scope["O"].envelope_signal())[2]
	assert _kinds(read_display_signal(hybrid_column.wave_cross_section_signal())[2])["image"]
	assert _shown_cross_section(hybrid_column)[0].shape[0] == 600		# show() too


def test_envelope_and_cross_section_survive_sea_round_trips(hybrid_column, tmp_path):
	"""The categorical quantity axis, per-row units and every marker reload intact."""
	from pySEA.sea_eco.io import load_sea
	scope = build_basic_column()
	scope.propagate_moments()
	for sig in (scope.envelope_signal(), hybrid_column.wave_cross_section_signal()):
		path = str(tmp_path / (sig.name.replace(" ", "_") + ".sea"))
		sig.to_sea(path)
		reloaded = load_sea(path)
		before, after = read_display_signal(sig), read_display_signal(reloaded)
		assert np.array_equal(after[0], before[0])
		assert [(n, u) for n, v, u in after[1]] == [(n, u) for n, v, u in before[1]]
		assert all(np.array_equal(a[1], b[1]) for a, b in zip(after[1], before[1]))
		assert after[2] == before[2] and after[3] == before[3]
		assert reloaded.signal_type == sig.signal_type
