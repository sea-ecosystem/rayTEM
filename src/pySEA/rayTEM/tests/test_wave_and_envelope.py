# TESTS for the wavelength utility, beam-envelope (covariance) propagation, and
# wave-optics propagation. See notes/eric/PLAN_2026-08-08_signal-and-propagation-additions.md

import sys
sys.path.insert(1,"../../../")
import numpy as np
import pytest

from pySEA.rayTEM.utilities import relativistic_wavelength


# --- wavelength ------------------------------------------------------------

def test_relativistic_wavelength_known_values():
	# Standard TEM textbook values (picometres)
	assert abs(relativistic_wavelength(200)*1e12 - 2.508) < 1e-3
	assert abs(relativistic_wavelength(300)*1e12 - 1.969) < 1e-3
	# monotonic decrease with voltage
	assert relativistic_wavelength(80) > relativistic_wavelength(200) > relativistic_wavelength(300)

def test_relativistic_wavelength_rejects_nonpositive():
	with pytest.raises(ValueError):
		relativistic_wavelength(0)
	with pytest.raises(ValueError):
		relativistic_wavelength(-100)
