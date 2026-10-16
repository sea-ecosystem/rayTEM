# TESTS BELOW INCLUDE: (please run from this directory: "pytest test_source_voltage.py")
# the relativistic_wavelength utility (known textbook values, validation)
# Source's optional accelerating voltage: derived wavelength, the per-ray E column,
# and the default (voltage=None) staying purely geometric so existing ray goldens hold

import sys,pytest
sys.path.insert(1,"../../../")
from pySEA.rayTEM import Source,columnByName
from pySEA.rayTEM.utilities import relativistic_wavelength
import numpy as np

# --- wavelength utility ----------------------------------------------------

# test: standard TEM textbook wavelengths (picometres), and monotonic decrease with voltage
def test_relativistic_wavelength_known_values():
	assert abs(relativistic_wavelength(200)*1e12 - 2.508) < 1e-3
	assert abs(relativistic_wavelength(300)*1e12 - 1.969) < 1e-3
	assert relativistic_wavelength(80) > relativistic_wavelength(200) > relativistic_wavelength(300)

# test: a non-positive accelerating voltage is meaningless and should be rejected
def test_relativistic_wavelength_rejects_nonpositive():
	with pytest.raises(ValueError):
		relativistic_wavelength(0)
	with pytest.raises(ValueError):
		relativistic_wavelength(-100)

# --- Source.voltage --------------------------------------------------------

# test: a voltage-carrying Source derives its wavelength (metres) and seeds the E column (keV)
def test_source_voltage_seeds_wavelength_and_energy_column():
	source = Source(name="gun",voltage=200)
	assert source.voltage == 200
	assert abs(source.wavelength - 2.5079e-12) < 1e-15
	rays = source.rays()
	assert np.allclose(rays[:,columnByName("E")],200)

# test: the default Source is unchanged -- no wavelength, E stays 0 (existing ray goldens hold)
def test_source_without_voltage_stays_geometric():
	source = Source(name="gun")
	assert source.voltage is None
	assert source.wavelength is None
	rays = source.rays()
	assert np.allclose(rays[:,columnByName("E")],0)

# test: voltage survives a round trip through the reinstantiation used by save/load
def test_source_voltage_survives_reinstantiation():
	from pySEA.rayTEM.seashells import safeReinstantiate
	rebuilt = safeReinstantiate(Source(name="gun",voltage=300),"Source")
	assert rebuilt.voltage == 300
	assert abs(rebuilt.wavelength - relativistic_wavelength(300)) < 1e-18
