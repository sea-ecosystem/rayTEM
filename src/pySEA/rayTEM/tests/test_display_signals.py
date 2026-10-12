# TESTS for the derived display Signals: ray diagram, beam envelope and the
# scaled-wave cross-section, each a calibrated sea-eco Signal carrying its
# section/element/plane markers as PLOT-8 annotations.
# See notes/eric/TODO_ACTIVE_2026-10-12_live-model-plots.md.

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
	"""One Span per section, one Line per named element and per conjugate plane."""
	scope = build_basic_column()
	scope.propagate_ray()
	sig = scope.ray_diagram_signal()
	from pySEA.sea_eco.annotations import get_display_annotations
	annotations = list(get_display_annotations(sig)[0])
	planes = scope.conjugate_planes()
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
	assert all(a.coordinate_reference == {"kind": "calibrated"} for a in annotations)
	assert all(a.name.startswith("raytem.") for a in annotations)


def test_conjugate_planes_carry_magnification_and_rotation():
	"""conjugate_planes adds M/R aligned with each family: measured by 'ray', NaN by 'frame'."""
	scope = build_basic_column()
	ray = scope.conjugate_planes()
	frame = scope.conjugate_planes(method="frame")
	for family in ("diff", "image"):
		assert len(ray[family + "_M"]) == len(ray[family]) == len(ray[family + "_R"])
		assert np.isfinite(ray[family + "_M"]).all() and np.isfinite(ray[family + "_R"]).all()
		assert np.isnan(frame[family + "_M"]).all() and len(frame[family + "_R"]) == len(frame[family])


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
