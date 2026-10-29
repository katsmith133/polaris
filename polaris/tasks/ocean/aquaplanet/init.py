import gsw
import numpy as np
import xarray as xr
from mpas_tools.io import open_dataset

from polaris.coriolis import add_coriolis_to_dataset
from polaris.ocean.model import OceanIOStep
from polaris.ocean.vertical import init_vertical_coord
from polaris.tasks.ocean.aquaplanet.initial_state import (
    flat_bottom,
    initial_tracers,
    wind_forcing,
)


class Init(OceanIOStep):
    """Build the final all-wet ocean mesh, state and steady forcing.

    Future wet-mask/culling operations belong here, before writing the mesh
    and constructing vertical coordinates. Culling requires regenerating
    the graph and reconstruction weights for the final cell numbering.
    """

    def __init__(self, component, mesh, subdir):
        super().__init__(component=component, name='init', subdir=subdir)
        for filename in [
            'base_mesh.nc',
            'graph.info',
            'reconstruction_weights.nc',
        ]:
            self.add_input_file(
                filename=filename,
                work_dir_target=f'{mesh.path}/{filename}',
            )

    def setup(self):
        super().setup()
        self.add_output_files_for_ocean_model_input()
        self.add_output_file(self.get_forcing_filename())

    def run(self):
        config = self.config
        with open_dataset('base_mesh.nc') as source:
            mesh = source.load()
        mesh = add_coriolis_to_dataset(config, mesh)
        self.write_horiz_mesh_dataset(
            mesh, self.get_horiz_mesh_filename(), config
        )

        state = mesh.copy()
        state['bottomDepth'] = flat_bottom(config, mesh)
        state['ssh'] = xr.zeros_like(mesh.latCell)
        init_vertical_coord(config, state)
        temperature, salinity = initial_tracers(config, state.zMid)
        sigma = gsw.sigma0(salinity.values, temperature.values)
        vertical_axis = temperature.get_axis_num('nVertLevels')
        if np.any(np.diff(sigma, axis=vertical_axis) < -1.0e-10):
            raise ValueError(
                'Initial potential density must not decrease down'
            )
        if not np.all(np.isfinite(state.layerThickness.values)) or not bool(
            (state.layerThickness > 0.0).all()
        ):
            raise ValueError(
                'Initial active layers must be finite and positive'
            )
        state['temperature'] = temperature
        state['salinity'] = salinity
        velocity, _ = xr.broadcast(mesh.latEdge, state.refZMid)
        state['normalVelocity'] = xr.zeros_like(velocity).expand_dims('Time')

        self.write_vert_coord_dataset(
            state, self.get_vert_coord_filename(), config
        )
        self.write_initial_state_dataset(
            state,
            self.get_init_filename(),
            config,
            tracer_convention='teos-10',
            lon=np.rad2deg(mesh.lonCell),
            lat=np.rad2deg(mesh.latCell),
        )

        forcing = wind_forcing(config, mesh.latCell)
        if config.get('ocean', 'model') == 'mpas-ocean':
            forcing = forcing.expand_dims('Time')
        # Unlike write_forcing_dataset(), this preserves the KPP ice field
        # in addition to the two stresses in the shared forcing-variable list.
        self.write_model_dataset(forcing, self.get_forcing_filename(), config)
