import xarray as xr

from polaris.ocean.model import OceanIOStep


class Forcing(OceanIOStep):
    """
    Create a forcing file for one forcing amplitude.
    """

    def __init__(
        self,
        component,
        forcing_type,
        forcing_value,
        name,
        indir,
    ):
        super().__init__(component=component, name=name, indir=indir)
        self.forcing_type = forcing_type
        self.forcing_value = forcing_value
        self.add_input_file(filename='init.nc', target='../init/init.nc')
        self.add_output_file(filename='forcing.nc')

    def run(self):
        ds_init = self.open_model_dataset('init.nc', config=self.config)
        ds_init = ds_init.isel(Time=0)

        forcing_array = xr.ones_like(ds_init.temperature).expand_dims(
            dim='Time', axis=0
        )
        # Build a surface field from tracer dimensions to avoid depending on
        # model-specific mesh variables in init.nc.
        forcing_array_surface = xr.ones_like(
            ds_init.temperature.isel(nVertLevels=0)
        ).expand_dims(dim='Time', axis=0)

        ds_forcing = xr.Dataset()
        zero_surface = 0.0 * forcing_array_surface
        zero_volume = 0.0 * forcing_array

        ds_forcing['temperaturePistonVelocity'] = zero_surface
        ds_forcing['salinityPistonVelocity'] = zero_surface
        ds_forcing['temperatureSurfaceRestoringValue'] = zero_surface
        ds_forcing['salinitySurfaceRestoringValue'] = zero_surface
        ds_forcing['temperatureInteriorRestoringRate'] = zero_volume
        ds_forcing['salinityInteriorRestoringRate'] = zero_volume
        ds_forcing['temperatureInteriorRestoringValue'] = 0.0 * forcing_array
        ds_forcing['salinityInteriorRestoringValue'] = 0.0 * forcing_array
        ds_forcing['latentHeatFlux'] = zero_surface
        ds_forcing['sensibleHeatFlux'] = zero_surface
        ds_forcing['shortWaveHeatFlux'] = zero_surface
        ds_forcing['evaporationFlux'] = zero_surface
        ds_forcing['rainFlux'] = zero_surface
        ds_forcing['riverRunoffFlux'] = zero_surface
        ds_forcing['iceRunoffFlux'] = zero_surface
        ds_forcing['subglacialRunoffFlux'] = zero_surface
        ds_forcing['icebergFreshWaterFlux'] = zero_surface
        ds_forcing['windStressZonal'] = zero_surface
        ds_forcing['windStressMeridional'] = zero_surface

        if self.forcing_type == 'wind':
            ds_forcing['windStressZonal'] = (
                self.forcing_value * forcing_array_surface
            )
        else:
            raise ValueError(f'Unsupported forcing type: {self.forcing_type}')

        self.write_model_dataset(ds_forcing, 'forcing.nc', self.config)
