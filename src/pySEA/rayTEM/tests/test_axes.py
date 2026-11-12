"""A scope's axes (``pySEA.rayTEM.axes``): the golden cases, the twin's own, and a round trip.

The golden cases are sea-pearl's scope-map schema's ``fixtures/axes.json``
(SM-7), which this implementation and sea-pearl's copy both satisfy; they run
when sea-pearl is installed or checked out beside rayTEM.
"""

from __future__ import annotations

import json
import math
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pySEA.rayTEM.assemblies import Microscope, MicroscopeSection
from pySEA.rayTEM.axes import Axes, Axis
from pySEA.rayTEM.elements import Lens
from pySEA.rayTEM.seashells import sea_available


def _golden_cases() -> list[dict[str, Any]]:
    """Return the schema's golden cases for the classes, or none when the schema is not here.

    Returns
    -------
    list[dict[str, Any]]
        ``fixtures/axes.json``'s cases.
    """

    relative = ("ai_wiki", "sea_pearl", "schema", "scope-map", "fixtures", "axes.json")
    candidates = []
    try:
        candidates.append(files("pySEA").joinpath(*relative))
    except (ModuleNotFoundError, TypeError):
        pass
    candidates.append(Path(__file__).resolve().parents[5] / "sea-pearl" / "src" / "pySEA" / Path(*relative))
    for candidate in candidates:
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))["cases"]
        except (FileNotFoundError, OSError):
            continue
    return []


GOLDEN = _golden_cases()


def _axis(spec: dict[str, Any]) -> Axis:
    """Build an axis from a golden spec.

    Parameters
    ----------
    spec : dict[str, Any]
        ``name``, ``components`` and ``rotation``/``flipped`` or ``matrix``.

    Returns
    -------
    Axis
        The axis.
    """

    return Axis(spec["name"], spec["components"], matrix=spec.get("matrix"),
                rotation=spec.get("rotation", 0.0), flipped=spec.get("flipped", False))


def _axes(spec: dict[str, Any]) -> Axes:
    """Build an axes from a golden spec.

    Parameters
    ----------
    spec : dict[str, Any]
        ``primary`` and ``axes``.

    Returns
    -------
    Axes
        The axes.
    """

    return Axes.from_mapping(spec)


@pytest.mark.skipif(not GOLDEN, reason="sea-pearl's scope-map fixtures are not installed or checked out")
@pytest.mark.parametrize("case", GOLDEN, ids=lambda case: case["id"])
def test_golden_case(case: dict[str, Any]) -> None:
    """Every golden case of the scope-map schema's axes holds here."""
    error = {"ValueError": ValueError, "KeyError": KeyError}.get(case.get("error", ""))
    if error is not None:
        with pytest.raises(error):
            _axis(case["axis"]) if "axis" in case else _axes(case["axes"])
        return
    if "axis" in case:
        axis = _axis(case["axis"])
        np.testing.assert_allclose(axis.matrix, case["matrix"], atol=1e-12)
        assert axis.rotation == pytest.approx(case["rotation"], abs=1e-12)
        assert axis.flipped is case["flipped"]
        return
    axes = _axes(case["axes"])
    if "set_primary" in case:
        axes.set_primary(case["set_primary"])
        assert axes.primary == case["set_primary"]
        for name, matrix in case["matrices"].items():
            np.testing.assert_allclose(axes[name].matrix, matrix, atol=1e-12)
        for name, rotation in case["rotations"].items():
            assert axes[name].rotation == pytest.approx(rotation, abs=1e-12)
    for transform in case.get("transforms", []):
        np.testing.assert_allclose(axes.matrix(transform["from"], transform["to"]), transform["matrix"], atol=1e-12)
        np.testing.assert_allclose(axes.transform(transform["vector"], transform["from"], transform["to"]),
                                   transform["result"], atol=1e-12)
    for call in case.get("raises", []):
        with pytest.raises(KeyError):
            if call["call"] == "matrix":
                axes.matrix(call["from"], call["to"])
            else:
                axes.transform(call["vector"], call["from"], call["to"])
    for key, name in case.get("lookups", {}).items():
        found = axes.axis_of(key)
        assert (found.name if found is not None else None) == name, key
    for pair, name in case.get("pair_lookups", []):
        found = axes.axis_of(pair)
        assert (found.name if found is not None else None) == name, pair


def test_a_fresh_axes_holds_the_corrector_alone() -> None:
    """Nothing is assumed: the primary is stated, every other axis has to be."""
    axes = Axes()
    assert axes.primary == "corrector" and axes.names == ("corrector",)
    assert axes["ax"].components == ("ax", "ay")
    with pytest.raises(KeyError):
        axes.transform([1.0, 0.0], "camera", "corrector")


def test_a_primary_that_is_not_the_identity_is_refused() -> None:
    """The primary is the reference; a turned primary would contradict every other statement."""
    with pytest.raises(ValueError):
        Axes([Axis("corrector", ("ax", "ay"), rotation=0.1)])


def test_the_primary_is_never_removed() -> None:
    """Withdrawing the reference would leave every matrix stated against nothing."""
    axes = Axes([Axis("corrector", ("ax", "ay")), Axis("camera", ("x", "y"), rotation=0.2)])
    with pytest.raises(ValueError):
        axes.remove("corrector")
    axes.remove("camera")
    assert axes.names == ("corrector",)


def test_a_stack_transforms_row_by_row() -> None:
    """A stack of vectors is each vector transformed."""
    axes = Axes([Axis("corrector", ("ax", "ay")), Axis("camera", ("x", "y"), rotation=math.pi / 2)])
    np.testing.assert_allclose(axes.transform([[1.0, 0.0], [0.0, 2.0]], "camera", "corrector"),
                               [[0.0, 1.0], [-2.0, 0.0]], atol=1e-12)


def test_the_twin_carries_axes() -> None:
    """A column starts with the corrector alone; a slice and a copy keep what it states."""
    scope = Microscope(name="twin", sections=[MicroscopeSection(name="O", elements=[Lens(name="OL", strength=1.0)])])
    assert scope.axes.names == ("corrector",)
    scope.axes.add(Axis("camera", ("x", "y"), rotation=0.7))
    assert scope.copy().axes.names == ("corrector", "camera")
    assert scope["O":].axes["x"].rotation == pytest.approx(0.7)


@pytest.mark.skipif(not sea_available, reason="the .sea round trip needs sea_eco")
def test_the_twin_axes_survive_a_sea_round_trip(tmp_path: Path) -> None:
    """Matrices, components, descriptions and the primary come back as the classes that wrote them."""
    scope = Microscope(name="twin", sections=[MicroscopeSection(name="O", elements=[Lens(name="OL", strength=1.0)])])
    scope.axes.add(Axis("camera", ("x", "y"), rotation=0.7, description="at the saved operating point"))
    scope.axes.add(Axis("scan", ("u", "v"), matrix=[[-1.0, 0.0], [0.0, 1.0]]))
    scope.axes.set_primary("camera")
    path = tmp_path / "twin.sea"
    scope.to_sea(str(path))
    loaded = Microscope()
    loaded.from_sea(str(path))
    # By class name: pytest's default import mode also imports this package as
    # plain ``rayTEM`` (no ``pySEA.``), so a second Axis class exists in tests only.
    assert type(loaded.axes).__name__ == "Axes" and all(type(axis).__name__ == "Axis" for axis in loaded.axes)
    assert loaded.axes.primary == "camera" and loaded.axes.names == ("corrector", "camera", "scan")
    assert loaded.axes["camera"].description == "at the saved operating point"
    assert loaded.axes["u"].components == ("u", "v")
    np.testing.assert_allclose(loaded.axes.matrix("scan", "corrector"), scope.axes.matrix("scan", "corrector"), atol=1e-12)


def test_the_twin_axes_survive_the_flat_json_layout(tmp_path: Path) -> None:
    """``Microscope.save`` writes the axes as plain data and ``load_microscope`` reads them."""
    from pySEA.rayTEM.assemblies import load_microscope

    scope = Microscope(name="twin", sections=[MicroscopeSection(name="O", elements=[Lens(name="OL", strength=1.0)])])
    scope.axes.add(Axis("camera", ("x", "y"), rotation=0.7, description="stated"))
    scope.save(str(tmp_path / "twin"))
    loaded = load_microscope(str(tmp_path / "twin"))
    assert loaded.axes.names == ("corrector", "camera")
    assert loaded.axes["camera"].rotation == pytest.approx(0.7) and loaded.axes["camera"].description == "stated"
