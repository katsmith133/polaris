import numpy as np
import xarray as xr


def flat_bottom(config, mesh):
    """Return positive-down bathymetry, separately from the wet mask."""
    depth = config.getfloat('vertical_grid', 'bottom_depth')
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError('vertical_grid:bottom_depth must be positive')
    return depth * xr.ones_like(mesh.latCell)


def initial_tracers(config, z_mid):
    """Return Conservative Temperature and Absolute Salinity on model layers.

    The initial column is horizontally uniform. No SST target or heat flux
    is implied by this initial temperature distribution.
    """
    section = config['aquaplanet']
    deep = section.getfloat('deep_temperature')
    contrast = section.getfloat('temperature_contrast')
    scale = section.getfloat('temperature_scale')
    salinity = section.getfloat('salinity')
    if not np.all(np.isfinite([deep, contrast, scale, salinity])):
        raise ValueError('Initial tracer parameters must be finite')
    if scale <= 0.0 or contrast < 0.0 or salinity <= 0.0:
        raise ValueError('Require scale > 0, contrast >= 0 and salinity > 0')
    temperature = deep + contrast * np.exp(z_mid / scale)
    salt = salinity * xr.ones_like(temperature)
    temperature.attrs.update(
        units='degC', long_name='initial Conservative Temperature'
    )
    salt.attrs.update(units='g kg-1', long_name='initial Absolute Salinity')
    return temperature, salt


def wind_forcing(config, latitude):
    """Evaluate steady two-harmonic stress on latitude in radians.

    This is the user-selected sinusoidal prescription, not a reproduction
    of every boundary condition in the original Bryan experiment.
    """
    amplitude = config.getfloat('aquaplanet', 'wind_stress_amplitude')
    if not np.isfinite(amplitude):
        raise ValueError('aquaplanet:wind_stress_amplitude must be finite')
    forcing = xr.Dataset()
    forcing['windStressZonal'] = amplitude * (
        np.sin(2.0 * latitude) + 0.5 * np.sin(6.0 * latitude)
    )
    forcing['windStressMeridional'] = xr.zeros_like(latitude)
    for name in ['windStressZonal', 'windStressMeridional']:
        forcing[name].attrs['units'] = 'N m-2'
    # KPP uses ice fraction even when Langmuir enhancement is disabled.
    forcing['iceFraction'] = xr.zeros_like(latitude)
    forcing['windSpeed10m'] = xr.zeros_like(latitude)
    forcing.iceFraction.attrs['units'] = '1'
    forcing.windSpeed10m.attrs['units'] = 'm s-1'
    # Omega's nonlocal pathway requires the surface tracer-flux calculation.
    # Supply its inputs explicitly without adding heat or freshwater forcing.
    for name in [
        'latentHeatFlux',
        'sensibleHeatFlux',
        'shortWaveHeatFlux',
        'longWaveHeatFluxUp',
        'longWaveHeatFluxDown',
        'seaIceHeatFlux',
    ]:
        forcing[name] = xr.zeros_like(latitude)
        forcing[name].attrs['units'] = 'W m-2'
    for name in [
        'snowFlux',
        'rainFlux',
        'evaporationFlux',
        'seaIceFreshWaterFlux',
        'iceRunoffFlux',
        'riverRunoffFlux',
        'seaIceSalinityFlux',
    ]:
        forcing[name] = xr.zeros_like(latitude)
        forcing[name].attrs['units'] = 'kg m-2 s-1'
    return forcing
