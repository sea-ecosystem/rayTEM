"""A scope's axes: named 2x2 frames, and transforms between any two of them.

Extended summary
----------------
An instrument keeps several two-dimensional coordinate frames turned (and
sometimes mirrored) relative to one another: an aberration corrector's, a
camera's, a stage's, a projection system's, a scan's. An :class:`Axis` is one
of them: a name, the names of its two components, and the 2x2 matrix that
takes a vector given in its components to the **primary** axis's,
``v_primary = M @ v_axis``. An :class:`Axes` is the set of them a scope has
stated, and converts a vector from any one to any other.

Nothing here is specific to electron microscopes or to any vendor: an axis is a
frame and a matrix. Which axes a scope has, and their names, are the scope's
own statement (sea-pearl's scope-map schema, SM-6 and SM-7).

This file has two homes, the same but for the line marked ``# axes-base``:
``pySEA.rayTEM.axes`` (the twin's) and ``pySEA.sea_pearl._axes_frames`` (the
operational microscope's, used when rayTEM is not installed). A sea-pearl test
fails when they differ anywhere else.

Examples
--------
>>> axes = Axes([Axis("corrector", ("ax", "ay")), Axis("camera", ("x", "y"), rotation=0.5)])
>>> [round(float(value), 6) for value in axes.transform([1.0, 0.0], "camera", "corrector")]
[0.877583, 0.479426]
>>> axes["x"].name
'camera'
"""

from __future__ import annotations

import math
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import ArrayLike

from pySEA.rayTEM.seashells import SEASerializable  # axes-base

#: The primary axis an :class:`Axes` starts with, and its components.
DEFAULT_PRIMARY: str = "corrector"
#: The components of :data:`DEFAULT_PRIMARY`.
DEFAULT_PRIMARY_COMPONENTS: tuple[str, str] = ("ax", "ay")
#: How close to the identity the primary axis's matrix must be, and to each
#: other two statements of one axis must be to agree.
MATRIX_TOLERANCE: float = 1e-9


def _rotation_matrix(rotation: float, flipped: bool) -> np.ndarray:
    """Return ``R(rotation) @ diag(1, -1 if flipped else 1)``.

    Parameters
    ----------
    rotation : float
        Radians, counter-clockwise.
    flipped : bool
        Whether the second component is negated (the other handedness).

    Returns
    -------
    numpy.ndarray
        The 2x2 matrix.

    Raises
    ------
    None
    """

    c, s = math.cos(float(rotation)), math.sin(float(rotation))
    flip = -1.0 if flipped else 1.0
    return np.array([[c, -s * flip], [s, c * flip]], dtype=np.float64)


def _as_matrix(matrix: ArrayLike) -> np.ndarray:
    """Return a 2x2 float matrix, refusing anything else and anything singular.

    Parameters
    ----------
    matrix : ArrayLike
        The candidate.

    Returns
    -------
    numpy.ndarray
        A float64 copy, shape ``(2, 2)``.

    Raises
    ------
    ValueError
        If it is not 2x2, not finite, or (numerically) singular.
    """

    value = np.array(matrix, dtype=np.float64)
    if value.shape != (2, 2) or not np.all(np.isfinite(value)):
        raise ValueError(f"An axis's matrix is a finite 2x2; got {value.tolist()!r}.")
    scale = float(np.sum(value * value))
    if scale == 0.0 or abs(float(np.linalg.det(value))) <= 1e-12 * scale:
        raise ValueError(f"An axis's matrix must be invertible; {value.tolist()!r} is singular.")
    return value


class Axis(SEASerializable):
    """One named two-dimensional frame, stated relative to the primary axis.

    Extended summary
    ----------------
    The matrix is the statement: it takes a vector given in this axis's two
    components to the primary axis's, ``v_primary = matrix @ v_axis``. Its
    :attr:`rotation` and :attr:`flipped` are read from it, so two axes compare
    at a glance. Build one from a rotation and a flip (a rotated, possibly
    mirrored, orthonormal frame) or from any invertible 2x2 (a frame with a
    ``-1`` scale, or a skewed one).

    Parameters
    ----------
    name : str, optional
        The axis's name in its scope (``"corrector"``, ``"camera"`` ...).
    components : Sequence[str], optional
        Its two components' names, the halves a two-axis control in it uses
        (``("ax", "ay")``); default ``("x", "y")``.
    matrix : ArrayLike or None, optional
        The 2x2 matrix to the primary axis. When given, ``rotation`` and
        ``flipped`` must be left at their defaults.
    rotation : float, optional
        Radians from the primary axis's first component to this axis's,
        counter-clockwise (default 0). Used when ``matrix`` is ``None``.
    flipped : bool, optional
        Whether this axis has the other handedness: its second component is
        negated after the rotation (default ``False``).
    description : str, optional
        What the statement is and where it came from (an operating point, a
        measurement).

    Attributes
    ----------
    name : str
        As given.
    description : str
        As given.
    components : tuple[str, str]
        The two component names.
    matrix : numpy.ndarray
        A copy of the 2x2 matrix to the primary axis.
    rotation : float
        Radians in (-pi, pi]: where this axis's first component points in the
        primary axis, ``atan2(matrix[1, 0], matrix[0, 0])``.
    flipped : bool
        ``det(matrix) < 0``.

    Methods
    -------
    to_primary(vector)
        A vector (or a stack) in this axis, in the primary axis.
    from_primary(vector)
        A vector (or a stack) in the primary axis, in this one.

    Raises
    ------
    ValueError
        If the components are not two distinct non-empty names, the matrix is
        not an invertible 2x2, or both a matrix and a rotation or flip are given.

    See Also
    --------
    Axes : The set a scope states, and transforms between them.

    Notes
    -----
    sea-eco stores the matrix and the components (as ``matrix`` and
    ``components``); the rotation and the flip are derived and not stored.

    Examples
    --------
    >>> axis = Axis("stage", ("sx", "sy"), rotation=-1.0, flipped=True)
    >>> round(axis.rotation, 6), axis.flipped
    (-1.0, True)
    >>> Axis("scan", ("u", "v"), matrix=[[-1.0, 0.0], [0.0, 1.0]]).flipped
    True
    """

    seaid_optional = True

    def __init__(
        self,
        name: str = "",
        components: Sequence[str] = ("x", "y"),
        matrix: ArrayLike | None = None,
        rotation: float = 0.0,
        flipped: bool = False,
        description: str = "",
    ) -> None:
        """Hold a name, two component names and a matrix.

        Parameters
        ----------
        name, components, matrix, rotation, flipped, description
            See the class.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            See the class.
        """

        super().__init__()
        parts = [str(part) for part in components]
        if len(parts) != 2 or len(set(parts)) != 2 or not all(parts):
            raise ValueError(f"Axis {name!r}: components are two distinct names; got {list(components)!r}.")
        if matrix is not None and (float(rotation) != 0.0 or bool(flipped)):
            raise ValueError(f"Axis {name!r}: give a matrix, or a rotation and flip, not both.")
        self.name = str(name)
        self.description = str(description)
        self._components = parts
        self._matrix = _as_matrix(matrix) if matrix is not None else _rotation_matrix(rotation, bool(flipped))

    @property
    def components(self) -> tuple[str, str]:
        """Return the two component names.

        Returns
        -------
        tuple[str, str]
            ``("ax", "ay")`` for the corrector.

        Raises
        ------
        None
        """

        first, second = (str(part) for part in self._components)
        return first, second

    @property
    def matrix(self) -> np.ndarray:
        """Return a copy of the 2x2 matrix to the primary axis.

        Returns
        -------
        numpy.ndarray
            ``M`` with ``v_primary = M @ v_axis``.

        Raises
        ------
        None
        """

        return np.array(self._matrix, dtype=np.float64)

    @property
    def rotation(self) -> float:
        """Return where this axis's first component points in the primary axis.

        Returns
        -------
        float
            Radians in (-pi, pi], counter-clockwise from the primary axis's
            first component.

        Raises
        ------
        None
        """

        matrix = self.matrix
        return float(math.atan2(matrix[1, 0], matrix[0, 0]))

    @property
    def flipped(self) -> bool:
        """Return whether this axis has the other handedness from the primary.

        Returns
        -------
        bool
            ``det(matrix) < 0``.

        Raises
        ------
        None
        """

        return bool(np.linalg.det(self.matrix) < 0.0)

    def to_primary(self, vector: ArrayLike) -> np.ndarray:
        """Return a vector given in this axis, in the primary axis.

        Parameters
        ----------
        vector : ArrayLike
            Two components, or a stack of shape ``(n, 2)``.

        Returns
        -------
        numpy.ndarray
            The same shape, in the primary axis.

        Raises
        ------
        ValueError
            If the last dimension is not 2.
        """

        return _apply(self.matrix, vector)

    def from_primary(self, vector: ArrayLike) -> np.ndarray:
        """Return a vector given in the primary axis, in this axis.

        Parameters
        ----------
        vector : ArrayLike
            Two components, or a stack of shape ``(n, 2)``.

        Returns
        -------
        numpy.ndarray
            The same shape, in this axis.

        Raises
        ------
        ValueError
            If the last dimension is not 2.
        """

        return _apply(np.linalg.inv(self.matrix), vector)

    def __repr__(self) -> str:
        """Return a short description.

        Returns
        -------
        str
            ``Axis('camera', x/y, rotation=0.5000, flipped=False)``.

        Raises
        ------
        None
        """

        return (f"Axis({self.name!r}, {'/'.join(self.components)}, "
                f"rotation={self.rotation:.4f}, flipped={self.flipped})")


def _apply(matrix: np.ndarray, vector: ArrayLike) -> np.ndarray:
    """Apply a 2x2 matrix to one vector or to each row of a stack.

    Parameters
    ----------
    matrix : numpy.ndarray
        The 2x2 matrix.
    vector : ArrayLike
        Shape ``(2,)`` or ``(n, 2)``.

    Returns
    -------
    numpy.ndarray
        ``matrix @ vector``, row by row for a stack.

    Raises
    ------
    ValueError
        If the last dimension is not 2.
    """

    value = np.asarray(vector, dtype=np.float64)
    if value.ndim not in (1, 2) or value.shape[-1] != 2:
        raise ValueError(f"A vector here has two components (or is a stack of shape (n, 2)); got shape {value.shape}.")
    return value @ matrix.T if value.ndim == 2 else matrix @ value


class Axes(SEASerializable):
    """The axes a scope has stated, and transforms between any two of them.

    Extended summary
    ----------------
    Every axis is stated relative to the **primary** axis, whose own matrix is
    the identity. An axis the scope has not stated is not assumed to be aligned
    with anything: a transform through it raises. A fresh ``Axes`` holds only
    the primary axis, ``corrector`` (components ``ax``/``ay``).

    Parameters
    ----------
    axes : Sequence[Axis] or None, optional
        The axes. ``None`` (default) holds the primary axis alone, which is
        then ``corrector`` unless ``primary`` names another (then ``axes``
        must state it).
    primary : str, optional
        The primary axis's name, default ``"corrector"``.

    Attributes
    ----------
    primary : str
        The primary axis's name.
    names : tuple[str, ...]
        The axes' names, in the order they were stated.
    primary_axis : Axis
        The primary axis.

    Methods
    -------
    axis_of(key)
        The axis a name, a component, a pair or a control path names, or ``None``.
    add(axis)
        State another axis.
    remove(name)
        Withdraw one (never the primary).
    matrix(from_axis, to_axis)
        The 2x2 matrix taking a vector in one axis to another.
    transform(vector, from_axis, to_axis)
        A vector (or a stack) in one axis, in another.
    rotation_between(from_axis, to_axis)
        Where ``to_axis``'s first component points, in ``from_axis``.
    set_primary(name)
        Make another axis the primary, re-expressing every matrix.
    to_mapping()
        The axes as plain data (JSON-safe).
    from_mapping(mapping)
        Build axes from that data (or from rotations and flips).

    Raises
    ------
    ValueError
        If two axes share a name or a component pair, or the primary axis's
        matrix is not the identity.
    KeyError
        If the primary axis is not among ``axes``.

    See Also
    --------
    Axis : One of them.

    Notes
    -----
    Which axis may be the primary is a scope's rule, not this class's: the
    scope-map schema (SM-7) refuses a scan or stage primary, because those
    move with a setting or with every holder. This class only needs the
    primary to be stated.

    Examples
    --------
    >>> axes = Axes()
    >>> axes.add(Axis("stage", ("sx", "sy"), rotation=1.0))
    >>> round(axes.rotation_between("corrector", "stage"), 6)
    1.0
    >>> axes.set_primary("stage")
    >>> round(axes["corrector"].rotation, 6)
    -1.0
    """

    seaid_optional = True

    def __init__(self, axes: Sequence[Axis] | None = None, primary: str = DEFAULT_PRIMARY) -> None:
        """Hold the stated axes and the primary axis's name.

        Parameters
        ----------
        axes, primary
            See the class.

        Returns
        -------
        None

        Raises
        ------
        ValueError, KeyError
            See the class.
        """

        super().__init__()
        if axes is None:
            axes = [Axis(DEFAULT_PRIMARY, DEFAULT_PRIMARY_COMPONENTS)] if primary == DEFAULT_PRIMARY else []
        self.primary = str(primary)
        self._axes: list[Axis] = []
        for axis in axes:
            self._check_new(axis)
            self._axes.append(axis)
        primary_axis = self.primary_axis
        if not np.allclose(primary_axis.matrix, np.eye(2), atol=MATRIX_TOLERANCE, rtol=0.0):
            raise ValueError(f"The primary axis {self.primary!r} is the reference: its matrix is the identity, "
                             f"not {primary_axis.matrix.tolist()!r}.")

    def _check_new(self, axis: Axis) -> None:
        """Refuse an axis whose name or component pair is already stated.

        Parameters
        ----------
        axis : Axis
            The candidate.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If it is not an :class:`Axis`.
        ValueError
            If its name or pair is taken.
        """

        if not isinstance(axis, Axis):
            raise TypeError(f"Axes holds Axis objects; got {type(axis).__name__}.")
        for other in self._axes:
            if other.name == axis.name:
                raise ValueError(f"An axis named {axis.name!r} is already stated.")
            if other.components == axis.components:
                raise ValueError(f"Components {'/'.join(axis.components)} are already {other.name!r}'s.")

    @property
    def names(self) -> tuple[str, ...]:
        """Return the axes' names, in the order they were stated.

        Returns
        -------
        tuple[str, ...]
            Names.

        Raises
        ------
        None
        """

        return tuple(axis.name for axis in self._axes)

    @property
    def primary_axis(self) -> Axis:
        """Return the primary axis.

        Returns
        -------
        Axis
            The axis every matrix is given to.

        Raises
        ------
        KeyError
            If the primary axis is not stated.
        """

        for axis in self._axes:
            if axis.name == self.primary:
                return axis
        raise KeyError(f"The primary axis {self.primary!r} is not stated; the axes are {', '.join(self.names) or 'none'}.")

    def axis_of(self, key: str | Sequence[str]) -> Axis | None:
        """Return the axis a name, a component, a pair or a control path names.

        Parameters
        ----------
        key : str or Sequence[str]
            An axis name (``"camera"``), one component (``"x"``), a pair
            (``("x", "y")``), or a control path whose last segment is a
            component (``"P.P1.PL1.shift.y"``). A name is looked for first,
            then a component, then a path's last segment.

        Returns
        -------
        Axis or None
            The axis, or ``None`` when the key names none.

        Raises
        ------
        None
        """

        if not isinstance(key, str):
            pair = tuple(str(part) for part in key)
            return next((axis for axis in self._axes if axis.components == pair), None)
        for axis in self._axes:
            if axis.name == key:
                return axis
        for candidate in (key, key.rsplit(".", 1)[-1]):
            for axis in self._axes:
                if candidate in axis.components:
                    return axis
        return None

    def add(self, axis: Axis) -> None:
        """State another axis, relative to the primary.

        Parameters
        ----------
        axis : Axis
            The axis.

        Returns
        -------
        None

        Raises
        ------
        TypeError, ValueError
            If it is not an :class:`Axis`, or its name or pair is taken.
        """

        self._check_new(axis)
        self._axes.append(axis)

    def remove(self, name: str) -> None:
        """Withdraw a stated axis.

        Parameters
        ----------
        name : str
            Its name.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If no axis has the name.
        ValueError
            If it is the primary axis (make another the primary first).
        """

        if name == self.primary:
            raise ValueError(f"{name!r} is the primary axis; set another primary before removing it.")
        axis = self[name]
        self._axes.remove(axis)

    def matrix(self, from_axis: str | Sequence[str], to_axis: str | Sequence[str]) -> np.ndarray:
        """Return the 2x2 matrix taking a vector in one axis to another.

        Parameters
        ----------
        from_axis, to_axis : str or Sequence[str]
            Axes, named any way :meth:`axis_of` takes.

        Returns
        -------
        numpy.ndarray
            ``M_to^-1 @ M_from``.

        Raises
        ------
        KeyError
            If either names no stated axis.
        """

        source, target = self[from_axis], self[to_axis]
        return np.linalg.solve(target.matrix, source.matrix)

    def transform(self, vector: ArrayLike, from_axis: str | Sequence[str], to_axis: str | Sequence[str]) -> np.ndarray:
        """Return a vector given in one axis, in another.

        Parameters
        ----------
        vector : ArrayLike
            Two components, or a stack of shape ``(n, 2)``.
        from_axis, to_axis : str or Sequence[str]
            Axes, named any way :meth:`axis_of` takes.

        Returns
        -------
        numpy.ndarray
            The same shape, in ``to_axis``.

        Raises
        ------
        KeyError
            If either names no stated axis.
        ValueError
            If the vector's last dimension is not 2.
        """

        return _apply(self.matrix(from_axis, to_axis), vector)

    def rotation_between(self, from_axis: str | Sequence[str], to_axis: str | Sequence[str]) -> float:
        """Return where one axis's first component points, seen in another.

        Parameters
        ----------
        from_axis : str or Sequence[str]
            The axis the angle is measured in.
        to_axis : str or Sequence[str]
            The axis whose first component is measured.

        Returns
        -------
        float
            Radians in (-pi, pi], counter-clockwise.

        Raises
        ------
        KeyError
            If either names no stated axis.
        """

        matrix = self.matrix(to_axis, from_axis)
        return float(math.atan2(matrix[1, 0], matrix[0, 0]))

    def set_primary(self, name: str) -> None:
        """Make another stated axis the primary, without moving any axis.

        Every matrix is re-expressed relative to the new primary
        (``M_i <- M_new^-1 @ M_i``), so every transform between two axes is
        unchanged.

        Parameters
        ----------
        name : str
            The new primary axis, named any way :meth:`axis_of` takes.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If it names no stated axis.
        """

        new = self[name]
        inverse = np.linalg.inv(new.matrix)
        for axis in self._axes:
            axis._matrix = np.eye(2) if axis is new else inverse @ axis.matrix
        self.primary = new.name

    def to_mapping(self) -> dict:
        """Return the axes as plain, JSON-safe data.

        Returns
        -------
        dict
            ``{"primary": name, "axes": [{"name", "components", "matrix",
            "description"}, ...]}``, the form :meth:`from_mapping` reads.

        Raises
        ------
        None

        See Also
        --------
        from_mapping : The reverse.

        Notes
        -----
        A wire form for formats that are not sea-eco's (a flat JSON column).
        A ``.sea`` file stores the objects themselves.
        """

        return {
            "primary": self.primary,
            "axes": [{"name": axis.name, "components": list(axis.components), "matrix": axis.matrix.tolist(),
                      "description": axis.description} for axis in self._axes],
        }

    @classmethod
    def from_mapping(cls, mapping: dict) -> "Axes":
        """Build axes from plain data.

        Parameters
        ----------
        mapping : dict
            ``{"primary": name, "axes": [...]}``; each axis gives ``name``,
            ``components`` and either ``matrix`` or ``rotation`` (and
            ``flipped``), optionally ``description``.

        Returns
        -------
        Axes
            The axes.

        Raises
        ------
        ValueError, KeyError
            As :class:`Axes` and :class:`Axis` refuse.

        Examples
        --------
        >>> Axes.from_mapping({"primary": "corrector", "axes": [
        ...     {"name": "corrector", "components": ["ax", "ay"]},
        ...     {"name": "camera", "components": ["x", "y"], "rotation": 0.5}]}).names
        ('corrector', 'camera')
        """

        axes = [Axis(item["name"], item["components"], matrix=item.get("matrix"),
                     rotation=item.get("rotation", 0.0), flipped=item.get("flipped", False),
                     description=item.get("description", "")) for item in mapping.get("axes", [])]
        return cls(axes, primary=mapping.get("primary", DEFAULT_PRIMARY))

    def __getitem__(self, key: str | Sequence[str]) -> Axis:
        """Return the axis a key names.

        Parameters
        ----------
        key : str or Sequence[str]
            Any key :meth:`axis_of` takes.

        Returns
        -------
        Axis
            The axis.

        Raises
        ------
        KeyError
            If it names no stated axis; an axis is never assumed.
        """

        axis = self.axis_of(key)
        if axis is None:
            raise KeyError(f"No stated axis is {key!r}; the stated axes are {', '.join(self.names) or 'none'}.")
        return axis

    def __contains__(self, key: object) -> bool:
        """Return whether a key names a stated axis.

        Parameters
        ----------
        key : object
            Any key :meth:`axis_of` takes.

        Returns
        -------
        bool
            Whether :meth:`axis_of` finds one.

        Raises
        ------
        None
        """

        return isinstance(key, (str, tuple, list)) and self.axis_of(key) is not None

    def __iter__(self) -> Iterator[Axis]:
        """Iterate over the stated axes, in the order they were stated.

        Returns
        -------
        Iterator[Axis]
            The axes.

        Raises
        ------
        None
        """

        return iter(list(self._axes))

    def __len__(self) -> int:
        """Return how many axes are stated.

        Returns
        -------
        int
            The count, the primary included.

        Raises
        ------
        None
        """

        return len(self._axes)

    def __repr__(self) -> str:
        """Return a short description.

        Returns
        -------
        str
            ``Axes(primary='corrector', corrector, camera)``.

        Raises
        ------
        None
        """

        return f"Axes(primary={self.primary!r}, {', '.join(self.names)})"


__all__ = ["Axis", "Axes", "DEFAULT_PRIMARY", "DEFAULT_PRIMARY_COMPONENTS", "MATRIX_TOLERANCE"]
