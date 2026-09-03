"""Quick visual check of a MACSTEM condenser/projector calibration. written by Codex"""

import argparse
import csv
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEV = Path(__file__).resolve().parents[1]
CAL = DEV.parent / "macstem_calibration"
sys.path.insert(0, str(DEV / "src"))

from pySEA.rayTEM import Source, load_microscope
from pySEA.rayTEM.postprocessing import diffraction_bundles_at_z, helper_focus_to


def rows(path, dicts=False):
	if not path.exists():
		warnings.warn(f"skipping missing file: {path}")
		return []
	with path.open(newline="") as f:
		lines = (line for line in f if line.strip() and not line.lstrip().startswith("#"))
		return list(csv.DictReader(lines) if dicts else csv.reader(lines))


def load(path):
	path = Path(path).expanduser().resolve()
	if path.suffix == ".json":
		path = path.with_suffix("")
	return load_microscope(str(path))


def note_missing(ax, title):
	ax.set_title(title)
	ax.text(.5, .5, "data unavailable", ha="center", va="center", transform=ax.transAxes)
	ax.set_axis_off()


def focus_cases(path, model_path, family):
	out = []
	for row in rows(path):
		try:
			m = load(model_path)
			if family == "P" and "DQCM" in m.keys():
				m["DQCM"].strength = 0
			l1, l2, current, *extra = row
			names = [f"{family}{i}" for i in range(1, 4 if family == "C" else 5)]
			settings = {name:{"strength":0.0} for name in names}
			settings[l1] = {"strength":float(current)}
			if extra:
				name, value = extra[0].split("=")
				settings[name] = {"strength":float(value)}
			m.update_with_settings(settings)
			backward = int(l1[1:]) > int(l2[1:])
			start = l2 if backward else 0
			end = ("sample" if family == "C" else "CCD") if backward else l2
			if start == "C1":
				start = "VOA"
			scope = m[:"P1"] if family == "C" else m["sample":]
			radius = np.sqrt(helper_focus_to(scope, start, end))
			out.append((f"{l1}→{l2}\n{float(current):.3f} A", float(radius)))
		except Exception as exc:
			warnings.warn(f"skipping focus condition {row}: {exc}")
	return out


def plot_focus(ax, data_dir, model_path):
	cl = focus_cases(data_dir / "CLs_critical.csv", model_path, "C")
	pl = focus_cases(data_dir / "PLs_critical.csv", model_path, "P")
	checks = cl + pl
	if not checks:
		return note_missing(ax, "Focusing conditions")
	labels, values = zip(*checks)
	x = np.arange(len(values))
	ax.bar(x, values, color=["tab:blue"] * len(cl) + ["tab:orange"] * len(pl))
	ax.set_xticks(x, labels, rotation=90, fontsize=7)
	ax.set_ylabel("radius at target plane (model units)")
	ax.set_title(f"Focusing conditions; RMS={np.sqrt(np.mean(np.square(values))):.3g}")
	ax.grid(axis="y", alpha=.25)


def current_curve(path, model_path):
	data = rows(path)
	if not data:
		return None
	measured = np.asarray(data, dtype=float)
	m = load(model_path)[:"P1"]
	z = m.get_element_position("VOA")
	model = []
	for current in measured[:,0]:
		try:
			m["C1"].strength = current
			rays = m.propagate_ray()
			model.append(m["VOA"].transmitted_fraction(rays.at_z(z)))
		except Exception as exc:
			warnings.warn(f"C1={current:g} A current calculation failed: {exc}")
			model.append(np.nan)
	model = np.asarray(model, dtype=float)
	measured[:,1] /= np.nanmax(measured[:,1])
	model /= np.nanmax(model)
	return measured[:,0], measured[:,1], model


def plot_current(ax, data_dir, model_path):
	try:
		curve = current_curve(data_dir / "C1_vs_beamcurrent.csv", model_path)
	except Exception as exc:
		warnings.warn(f"beam-current check failed: {exc}")
		curve = None
	if curve is None:
		return note_missing(ax, "Beam current through VOA")
	x, measured, model = curve
	ax.plot(x, measured, ".", label="measured")
	ax.plot(x, model, "-", label="model")
	ax.set(xlabel="C1 current (A)", ylabel="normalized beam current", title="Beam current through VOA")
	ax.grid(alpha=.25)
	ax.legend()


def rotation_values(path, model_path):
	measured = {row[0]:float(row[1]) for row in rows(path)}
	try:
		m = load(model_path)
	except Exception as exc:
		warnings.warn(f"rotation check failed: {exc}")
		return []
	out = []
	for name in ("P1", "P2", "P3", "P4"):
		try:
			lens = m[name]
			old = lens.strength
			eps, current = 1e-6, .25
			lens.strength = current + eps
			r1 = -lens.calibrated_strength * lens.length
			lens.strength = current - eps
			r0 = -lens.calibrated_strength * lens.length
			lens.strength = old
			out.append((name, measured.get(name, np.nan), (r1-r0)/(2*eps)))
		except Exception as exc:
			warnings.warn(f"skipping rotation for {name}: {exc}")
	print("\nLens rotation (radians per amp at 0.25 A)")
	print(f"{'lens':<6}{'measured':>14}{'model':>14}{'delta':>14}")
	for name, measured_rate, model_rate in out:
		print(f"{name:<6}{measured_rate:14.6g}{model_rate:14.6g}{model_rate-measured_rate:14.6g}")
	return out


def plot_rotation(ax, values):
	if not values:
		return note_missing(ax, "Lens rotation")
	names = [v[0] for v in values]
	measured = np.asarray([v[1] for v in values])
	model = np.asarray([v[2] for v in values])
	x = np.arange(len(names))
	ax.bar(x-.2, measured, .4, label="measured")
	ax.bar(x+.2, model, .4, label="model")
	ax.set_xticks(x, names)
	ax.set_ylabel("rotation per amp (rad/A)")
	ax.set_title("Lens rotation")
	ax.grid(axis="y", alpha=.25)
	ax.legend()


def diffraction_values(path, model_path):
	data = rows(path, dicts=True)
	out = []
	for row in data:
		try:
			m = load(model_path)
			for i in range(1,5):
				m[f"P{i}"].strength = float(row[f"P{i}_mA"])/1000
			scope = m["sample":]
			angle = 1e-3
			scope[0].insert(0, Source(size=(1e-4,1e-4), np_xy=(3,3), angle=(angle,angle), na_xy=(3,3)))
			rays = scope.propagate_ray()
			result = diffraction_bundles_at_z(scope.get_element_position("CCD"), rays)
			length = np.mean([abs(result["bundle_spread"][axis]) for axis in ("x", "y")])/angle
			blur = np.mean([abs(result["bundle_size"][axis]) for axis in ("x", "y")])
			actual = float(row["camera_length_actual_mm"]) if row["camera_length_actual_mm"] else np.nan
			deviation = 100*(length/actual-1) if np.isfinite(actual) else np.nan
			out.append((row["setting"], actual, length, deviation, blur))
		except Exception as exc:
			warnings.warn(f"skipping diffraction setting {row.get('setting', '?')}: {exc}")
	print("\nTable 2 diffraction check (camera lengths and bundle blur in model mm)")
	print(f"{'state':<7}{'actual':>12}{'model':>12}{'delta %':>12}{'blur':>12}")
	for state, actual, model, deviation, blur in out:
		print(f"{state:<7}{actual:12.5g}{model:12.5g}{deviation:12.3g}{blur:12.5g}")
	return out


def plot_diffraction(ax, values):
	if not values:
		return note_missing(ax, "Table 2 diffraction conditions")
	names = [v[0] for v in values]
	actual = np.asarray([v[1] for v in values])
	model = np.asarray([v[2] for v in values])
	x = np.arange(len(names))
	ax.bar(x-.2, actual, .4, label="measured")
	ax.bar(x+.2, model, .4, label="model")
	for i, (_, _, _, deviation, blur) in enumerate(values):
		label = f"{deviation:+.1f}%\nblur {blur:.3g}" if np.isfinite(deviation) else f"blur {blur:.3g}"
		ax.text(i+.2, model[i], label, ha="center", va="bottom", fontsize=7)
	ax.set_xticks(x, names)
	ax.set_ylabel("camera length (mm)")
	ax.set_title("Table 2 diffraction conditions")
	ax.grid(axis="y", alpha=.25)
	ax.legend()


def main():
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("model", nargs="?", default=CAL / "microscope", type=Path)
	p.add_argument("--data-dir", default=CAL, type=Path)
	p.add_argument("--save", type=Path)
	p.add_argument("--no-show", action="store_true")
	a = p.parse_args()
	table = a.data_dir / "table2.csv"
	if not table.exists():
		table = a.data_dir / "codex_attempts" / "table2.csv"
	fig, axes = plt.subplots(2, 2, figsize=(14,9), constrained_layout=True)
	plot_focus(axes[0,0], a.data_dir, a.model)
	plot_current(axes[0,1], a.data_dir, a.model)
	plot_rotation(axes[1,0], rotation_values(a.data_dir / "rotations.csv", a.model))
	plot_diffraction(axes[1,1], diffraction_values(table, a.model))
	fig.suptitle(f"MACSTEM calibration sanity check: {a.model}")
	if a.save:
		fig.savefig(a.save, dpi=180)
	if not a.no_show:
		plt.show()


if __name__ == "__main__":
	main()
