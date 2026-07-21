import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh

from polaris.ocean.coriolis import add_coriolis_to_dataset
from polaris.ocean.model import OceanIOStep
from polaris.ocean.vertical import init_vertical_coord


class Init(OceanIOStep):
    """
    Create a planar mesh and a family of independent initial columns.
    """

    def __init__(self, component, indir):
        super().__init__(component=component, name='init', indir=indir)

    def setup(self):
        super().setup()
        self.add_output_files_for_ocean_model_input(
            horiz_mesh_filename='culled_mesh.nc',
            base_mesh_filename='base_mesh.nc',
            graph_filename='culled_graph.info',
        )

    def run(self):
        logger = self.logger
        config = self.config
        section = config['vert_mix']

        resolution = section.getfloat('resolution')
        dc = 1e3 * resolution

        temp_gradients = _get_gradient_values(config, 'temperature')
        sal_gradients = _get_gradient_values(config, 'salinity')
        nx = len(temp_gradients)
        ny = len(sal_gradients)

        ds_mesh = make_planar_hex_mesh(
            nx=nx, ny=ny, dc=dc, nonperiodic_x=False, nonperiodic_y=False
        )
        write_netcdf(ds_mesh, 'base_mesh.nc')
        ds_mesh = cull(ds_mesh, logger=logger)
        ds_mesh = convert(
            ds_mesh, graphInfoFileName='culled_graph.info', logger=logger
        )
        ds_mesh = add_coriolis_to_dataset(config, ds_mesh)
        self.write_horiz_mesh_dataset(ds_mesh, 'culled_mesh.nc', config)

        ds = ds_mesh.copy()
        x_cell = ds_mesh.xCell
        y_cell = ds_mesh.yCell

        bottom_depth = config.getfloat('vertical_grid', 'bottom_depth')
        ds['bottomDepth'] = bottom_depth * xr.ones_like(x_cell)
        ds['ssh'] = xr.zeros_like(x_cell)
        init_vertical_coord(config, ds)

        z_mid = ds.refZMid
        surface_temperature = section.getfloat('surface_temperature')
        surface_salinity = section.getfloat('surface_salinity')
        u = section.getfloat('zonal_velocity')
        v = section.getfloat('meridional_velocity')

        temp_gradient_2d = _make_cell_parameter(
            x_cell=x_cell,
            y_cell=y_cell,
            values=temp_gradients,
            vary_in='x',
        )
        sal_gradient_2d = _make_cell_parameter(
            x_cell=x_cell,
            y_cell=y_cell,
            values=sal_gradients,
            vary_in='y',
        )

        temperature = surface_temperature + temp_gradient_2d * z_mid
        salinity = surface_salinity + sal_gradient_2d * z_mid

        temperature = temperature.transpose('nCells', 'nVertLevels')
        salinity = salinity.transpose('nCells', 'nVertLevels')

        temperature[:, 0] = surface_temperature
        salinity[:, 0] = surface_salinity

        temperature = temperature.expand_dims(dim='Time', axis=0)
        salinity = salinity.expand_dims(dim='Time', axis=0)

        normal_velocity = u * np.cos(ds_mesh.angleEdge) + v * np.sin(
            ds_mesh.angleEdge
        )
        normal_velocity, _ = xr.broadcast(normal_velocity, ds.refBottomDepth)
        normal_velocity = normal_velocity.transpose('nEdges', 'nVertLevels')
        normal_velocity = normal_velocity.expand_dims(dim='Time', axis=0)

        ds['temperature'] = temperature
        ds['salinity'] = salinity
        ds['normalVelocity'] = normal_velocity
        ds['temperatureGradient'] = temp_gradient_2d
        ds['salinityGradient'] = sal_gradient_2d

        ds.attrs['nx'] = nx
        ds.attrs['ny'] = ny
        ds.attrs['dc'] = dc

        self.write_vert_coord_dataset(ds, 'vert_coord.nc', config)
        self.write_initial_state_dataset(ds, 'init.nc', config)


def _get_gradient_values(config, tracer_name):
    section = config[f'vert_mix_{tracer_name}']
    start = section.getfloat('start')
    stop = section.getfloat('stop')
    increment = section.getfloat('increment')

    if increment <= 0.0:
        raise ValueError(f'{tracer_name} increment must be positive.')
    if stop < start:
        raise ValueError(
            f'{tracer_name} stop must be greater than or equal to start.'
        )

    count = int(round((stop - start) / increment)) + 1
    values = np.array([start + index * increment for index in range(count)])
    if abs(values[-1] - stop) > 1.0e-12:
        raise ValueError(
            f'{tracer_name} range must be exactly divisible by increment.'
        )
    return values


def _make_cell_parameter(x_cell, y_cell, values, vary_in):
    """
    Assign one value from ``values`` to each cell by binning the cell's
    coordinate into ``len(values)`` equal-width bands spanning the full
    periodic domain extent.  This works correctly for hex meshes where
    staggered rows produce more than ``len(values)`` unique coordinate
    positions.
    """
    if vary_in == 'x':
        coord = x_cell
    else:
        coord = y_cell

    n = len(values)
    coord_min = float(coord.min())
    coord_max = float(coord.max())
    # Use a small epsilon expansion so the max-coord cell falls in the last bin
    domain_span = (coord_max - coord_min) * (1.0 + 1.0e-10)
    band_width = domain_span / n

    raw = coord.values
    band_index = np.floor((raw - coord_min) / band_width).astype(int)
    band_index = np.clip(band_index, 0, n - 1)

    return xr.DataArray(values[band_index], dims=('nCells',))
