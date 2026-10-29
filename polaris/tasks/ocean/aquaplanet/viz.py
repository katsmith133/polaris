import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from mpas_tools.io import write_netcdf

from polaris.constants import get_constant
from polaris.mesh.reconstruct import (
    cartesian_to_local_geographic,
    tangential_reconstruction,
)
from polaris.ocean.model import OceanIOStep
from polaris.viz import plot_global_mpas_field


class Viz(OceanIOStep):
    """Plot the initial condition and transient response without coastlines."""

    def __init__(self, component, init, forward, subdir):
        super().__init__(component=component, name='viz', subdir=subdir)
        self.init_step = init
        self.add_input_file(
            filename='output.nc', work_dir_target=f'{forward.path}/output.nc'
        )
        self.add_output_file('surface.png')
        self.add_output_file('initial_surface.png')
        self.add_output_file('surface_change.png')
        self.add_output_file('ssh.png')
        self.add_output_file('wind.png')
        self.add_output_file('temperature_section.png')
        self.add_output_file('surface_diagnostics.nc')
        self.add_output_file('kinetic_energy.png')
        self.add_output_file('kinetic_energy.nc')
        self.add_output_file('surface_vorticity.png')
        for filename in [
            'assessment.nc',
            'adjustment.png',
            'inventory_drift.png',
            'velocity_section.png',
            'mechanical_power.png',
        ]:
            self.add_output_file(filename)

    def setup(self):
        super().setup()
        init = self.init_step
        for filename in [
            self.get_horiz_mesh_filename(),
            self.get_init_filename(),
            self.get_forcing_filename(),
        ]:
            self.add_input_file(
                filename=filename, work_dir_target=f'{init.path}/{filename}'
            )

    def run(self):
        with self.open_model_dataset(self.get_horiz_mesh_filename()) as ds:
            mesh = ds.load()
        with self.open_model_dataset(self.get_init_filename()) as ds:
            initial = ds.load()
        with self.open_model_dataset(self.get_forcing_filename()) as ds:
            forcing = ds.load()
        with self.open_model_dataset(
            'output.nc', decode_times=False
        ) as output:
            self._plot_output(mesh, initial, forcing, output)

    def _plot_output(self, mesh, initial, forcing, output):
        """Plot lazy history data while the backing file remains open."""
        if output.sizes.get('Time', 0) == 0:
            raise ValueError(
                'The forward run did not write any output records'
            )
        lat = np.rad2deg(mesh.latCell.values)
        last = output.isel(Time=-1)
        first = initial.isel(Time=0)
        t_initial = first.temperature.isel(nVertLevels=0)
        t_final = last.temperature.isel(nVertLevels=0)
        fields = [
            (
                t_initial,
                'Initial surface temperature',
                'deg C',
                'initial_surface.png',
                'aquaplanet_temperature_viz',
            ),
            (
                t_final,
                'Final surface temperature',
                'deg C',
                'surface.png',
                'aquaplanet_temperature_viz',
            ),
            (
                t_final - t_initial,
                'Surface temperature change',
                'K',
                'surface_change.png',
                'aquaplanet_anomaly_viz',
            ),
            (
                last.ssh,
                'Final sea-surface height',
                'm',
                'ssh.png',
                'aquaplanet_anomaly_viz',
            ),
        ]
        descriptor = None
        for field, title, units, filename, section in fields:
            descriptor = plot_global_mpas_field(
                da=field,
                out_filename=self.work_path(filename),
                config=self.config,
                colormap_section=section,
                mesh_ds=mesh,
                descriptor=descriptor,
                title=title,
                colorbar_label=units,
                plot_land=False,
                dpi=150,
            )

        vorticity = last.relativeVorticity.isel(nVertLevels=0)
        limit = float(abs(vorticity).max(skipna=True))
        if not np.isfinite(limit):
            raise ValueError('Surface vorticity has no finite color limits')
        limit = max(limit, 1.0e-12)
        self.config.set(
            'aquaplanet_vorticity_viz',
            'norm_args',
            value=str({'vmin': -limit, 'vmax': limit}),
        )
        plot_global_mpas_field(
            da=vorticity,
            out_filename=self.work_path('surface_vorticity.png'),
            config=self.config,
            colormap_section='aquaplanet_vorticity_viz',
            mesh_ds=mesh,
            descriptor=descriptor,
            title='Final surface relative vorticity',
            colorbar_label='s$^{-1}$',
            plot_land=False,
            dpi=150,
        )
        energy = self._plot_kinetic_energy(mesh, output)
        self._plot_assessment(mesh, initial, forcing, output, energy)

        order = np.argsort(lat)
        stress = forcing.windStressZonal
        if 'Time' in stress.dims:
            stress = stress.isel(Time=0)
        fig = Figure(figsize=(7, 4), constrained_layout=True)
        ax = fig.add_subplot()
        ax.plot(lat[order], stress.values[order])
        ax.set(
            xlabel='Latitude (degrees)',
            ylabel='Eastward wind stress (N m$^{-2}$)',
            title='Prescribed steady wind stress',
        )
        ax.grid(True)
        fig.savefig(self.work_path('wind.png'), dpi=150)

        # Area-weighted latitude bins work on any unstructured resolution.
        bins = np.arange(-90.0, 91.0, 5.0)
        temperature = last.temperature.transpose('nVertLevels', 'nCells')
        area = mesh.areaCell
        numerator = (
            (temperature * area)
            .groupby_bins(mesh.latCell * 180.0 / np.pi, bins)
            .sum('nCells')
        )
        denominator = area.groupby_bins(
            mesh.latCell * 180.0 / np.pi, bins
        ).sum('nCells')
        zonal_mean = numerator / denominator
        depth = last.zMid.weighted(area).mean('nCells')
        fig = Figure(figsize=(8, 5), constrained_layout=True)
        ax = fig.add_subplot()
        section_image = ax.pcolormesh(
            0.5 * (bins[:-1] + bins[1:]),
            depth.values,
            zonal_mean.values,
            shading='auto',
        )
        ax.set(xlabel='Latitude (degrees)', ylabel='Elevation (m)')
        ax.set_title('Final zonal-mean temperature')
        fig.colorbar(section_image, ax=ax, label='deg C')
        fig.savefig(self.work_path('temperature_section.png'), dpi=150)

        diagnostics = output[['temperature', 'salinity']].isel(nVertLevels=0)
        diagnostics = diagnostics.weighted(area).mean('nCells')
        diagnostics.attrs['description'] = (
            'Area-weighted surface means; not conserved column inventories'
        )
        write_netcdf(diagnostics, self.work_path('surface_diagnostics.nc'))
        if 'boundaryLayerDepth' in last:
            filename = 'kpp_depth.png'
            plot_global_mpas_field(
                da=last.boundaryLayerDepth,
                out_filename=self.work_path(filename),
                config=self.config,
                colormap_section='aquaplanet_kpp_viz',
                mesh_ds=mesh,
                descriptor=descriptor,
                title='Final KPP boundary-layer depth',
                colorbar_label='m',
                plot_land=False,
                dpi=150,
            )
            self.add_produced_file(filename)

    def _plot_kinetic_energy(self, mesh, output):
        """Integrate native specific KE with reference-density layer mass."""
        rho_ref = get_constant('seawater_density_reference')
        if 'PseudoThickness' in output:
            thickness_name = 'PseudoThickness'
        else:
            thickness_name = 'layerThickness'
        if 'time' in output:
            time = output['time']
            if not time.attrs.get('units', '').startswith('seconds since'):
                raise ValueError('Expected native time in seconds since epoch')
            days = time.values.astype(float) / 86400.0
        else:
            days = output.daysSinceStartOfSim.values.astype(float)
        if not np.all(np.isfinite(days)) or np.any(np.diff(days) <= 0.0):
            raise ValueError(
                'History timestamps must be finite and increasing'
            )

        energy = np.empty(output.sizes['Time'])
        for index in range(len(energy)):
            record = (
                output[['kineticEnergyCell', thickness_name]]
                .isel(Time=index)
                .load()
            )
            ke = record.kineticEnergyCell.transpose('nCells', 'nVertLevels')
            thickness = record[thickness_name].transpose(
                'nCells', 'nVertLevels'
            )
            if (
                not np.all(np.isfinite(ke.values))
                or not np.all(np.isfinite(thickness.values))
                or np.any(ke.values < 0.0)
                or np.any(thickness.values <= 0.0)
            ):
                raise ValueError(f'Invalid KE or thickness at record {index}')
            # kineticEnergyCell already contains the factor of one half.
            energy[index] = rho_ref * float(
                (ke * thickness * mesh.areaCell).sum(skipna=False)
            )

        diagnostics = (
            output[['kineticEnergyCell']]
            .isel(nCells=0, nVertLevels=0, drop=True)
            .drop_vars('kineticEnergyCell')
        )
        diagnostics['model_day'] = ('Time', days)
        diagnostics['model_day'].attrs['units'] = 'days'
        diagnostics['kinetic_energy'] = ('Time', energy)
        diagnostics['kinetic_energy'].attrs = {
            'units': 'J',
            'long_name': 'Global reference-density horizontal kinetic energy',
            'reference_density_kg_m3': rho_ref,
            'description': (
                'Sum of native specific kinetic energy times cell area, '
                'reference density and pseudo-thickness (Omega) or '
                'layer thickness (MPAS-Ocean); not a closed energy budget'
            ),
        }
        write_netcdf(diagnostics, self.work_path('kinetic_energy.nc'))
        fig = Figure(figsize=(8, 4), constrained_layout=True)
        ax = fig.add_subplot()
        ax.plot(days, energy)
        ax.set(
            xlabel='Model day',
            ylabel='Kinetic energy (J)',
            title='Global reference-density horizontal kinetic energy',
        )
        ax.grid(True, alpha=0.3)
        fig.savefig(self.work_path('kinetic_energy.png'), dpi=150)
        return diagnostics

    def _plot_assessment(self, mesh, initial, forcing, output, energy):
        """Assess adjustment; mechanical powers are continuum estimates."""
        area = mesh.areaCell.values
        rho_ref = get_constant('seawater_density_reference')
        drag = self.config.getfloat('aquaplanet', 'bottom_drag_coefficient')
        h_name = (
            'PseudoThickness'
            if 'PseudoThickness' in output
            else 'layerThickness'
        )
        initial_h = (
            initial[h_name]
            .isel(Time=0)
            .transpose('nCells', 'nVertLevels')
            .values
        )
        references = {'mass_proxy': np.sum(area[:, None] * initial_h)}
        for name in ['temperature', 'salinity']:
            values = (
                initial[name]
                .isel(Time=0)
                .transpose('nCells', 'nVertLevels')
                .values
            )
            references[name] = np.sum(area[:, None] * initial_h * values)
        stress = forcing
        if 'Time' in stress.dims:
            stress = stress.isel(Time=0)
        tau_u = stress.windStressZonal.values
        tau_v = stress.windStressMeridional.values
        names = [
            'ssh_min',
            'ssh_max',
            'ssh_rms',
            'osbl_mean',
            'osbl_max',
            'min_thickness',
            'mass_proxy',
            'temperature',
            'salinity',
            'wind_work',
            'bottom_drag_loss',
        ]
        metrics: dict[str, list[float]] = {name: [] for name in names}
        variables = [
            'ssh',
            'boundaryLayerDepth',
            'temperature',
            'salinity',
            h_name,
            'normalVelocity',
        ]
        for index in range(output.sizes['Time']):
            record = output[variables].isel(Time=index).load()
            for name in variables:
                if not np.all(np.isfinite(record[name].values)):
                    raise ValueError(f'Nonfinite {name} at record {index}')
            h = record[h_name].transpose('nCells', 'nVertLevels').values
            if np.any(h <= 0):
                raise ValueError('Assessment requires all layers active')
            weights = area[:, None] * h
            metrics['min_thickness'].append(h.min())
            metrics['mass_proxy'].append(weights.sum())
            for name in ['temperature', 'salinity']:
                values = record[name].transpose('nCells', 'nVertLevels').values
                metrics[name].append(np.sum(weights * values))
            ssh = record.ssh.values
            metrics['ssh_min'].append(ssh.min())
            metrics['ssh_max'].append(ssh.max())
            metrics['ssh_rms'].append(
                np.sqrt(np.average(ssh**2, weights=area))
            )
            bld = record.boundaryLayerDepth.values
            metrics['osbl_mean'].append(np.average(bld, weights=area))
            metrics['osbl_max'].append(bld.max())
            surface_u, surface_v = _velocity_layer(mesh, record, 0)
            bottom_u, bottom_v = _velocity_layer(mesh, record, -1)
            metrics['wind_work'].append(
                np.sum(area * (tau_u * surface_u + tau_v * surface_v))
            )
            metrics['bottom_drag_loss'].append(
                rho_ref
                * drag
                * np.sum(area * (bottom_u**2 + bottom_v**2) ** 1.5)
            )

        days = energy.model_day.values
        ds = xr.Dataset(
            {name: ('model_day', values) for name, values in metrics.items()},
            coords={'model_day': days},
        )
        for name, reference in references.items():
            if reference == 0:
                raise ValueError(f'Zero reference inventory: {name}')
            ds[f'{name}_relative_drift'] = (ds[name] - reference) / abs(
                reference
            )
            ds[f'{name}_relative_drift'].attrs['units'] = '1'
        for name in names[:6]:
            ds[name].attrs['units'] = 'm'
        ds.mass_proxy.attrs['units'] = 'm3'
        ds.temperature.attrs['units'] = 'degree_Celsius m3'
        ds.salinity.attrs['units'] = 'g kg-1 m3'
        ds.model_day.attrs['units'] = 'days'
        ds['kinetic_energy'] = ('model_day', energy.kinetic_energy.values)
        ds.kinetic_energy.attrs['units'] = 'J'
        if len(days) > 1:
            ds['ke_storage_rate'] = (
                'model_day',
                np.gradient(energy.kinetic_energy.values, days * 86400.0),
            )
        for name in ['wind_work', 'bottom_drag_loss', 'ke_storage_rate']:
            if name in ds:
                ds[name].attrs['units'] = 'W'
        ds.attrs['description'] = (
            'All-wet flat-bottom assessment. Inventories use pseudo-thickness '
            'for Omega. Wind work and positive bottom-drag loss use '
            'reconstructed cell velocities and are continuum estimates, '
            'not discrete implicit drag work or a closed energy budget. '
            'Viscous losses and pressure/PE conversion are not measured.'
        )
        write_netcdf(ds, self.work_path('assessment.nc'))
        self._plot_assessment_series(ds, h_name, references)
        self._plot_velocity_section(mesh, output)

    def _plot_assessment_series(self, ds, h_name, references):
        """Plot the small, reduced assessment time series."""
        days = ds.model_day.values
        fig = Figure(figsize=(12, 4), constrained_layout=True)
        axes = fig.subplots(1, 3)
        for name in ['ssh_min', 'ssh_max', 'ssh_rms']:
            axes[0].plot(days, ds[name], label=name)
        for name in ['osbl_mean', 'osbl_max']:
            axes[1].plot(days, ds[name], label=name)
        axes[2].plot(days, ds.min_thickness, label='Minimum thickness')
        titles = [
            'SSH adjustment',
            'Boundary-layer evolution',
            'Minimum layer pseudo-thickness'
            if h_name == 'PseudoThickness'
            else 'Minimum layer thickness',
        ]
        for ax, title in zip(axes, titles, strict=True):
            ax.set(title=title, xlabel='Model day', ylabel='m')
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.savefig(self.work_path('adjustment.png'), dpi=150)

        fig = Figure(figsize=(8, 4), constrained_layout=True)
        ax = fig.add_subplot()
        for name in references:
            ax.plot(days, ds[f'{name}_relative_drift'], label=name)
        ax.set(
            title='Relative inventory drift from initial state',
            xlabel='Model day',
            ylabel='Fraction of initial inventory',
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(self.work_path('inventory_drift.png'), dpi=150)

        fig = Figure(figsize=(9, 4), constrained_layout=True)
        ax = fig.add_subplot()
        for name, label in [
            ('wind_work', 'Wind work (+ input)'),
            ('bottom_drag_loss', 'Bottom drag (+ removal)'),
            ('ke_storage_rate', 'KE storage rate'),
        ]:
            if name in ds:
                ax.plot(days, ds[name] / 1.0e9, label=label)
        ax.set(
            title='Continuum power estimates: NOT a closed energy budget',
            xlabel='Model day',
            ylabel='GW',
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(self.work_path('mechanical_power.png'), dpi=150)

    def _plot_velocity_section(self, mesh, output):
        """Area-weighted latitude bins of final reconstructed zonal flow."""
        last = output.isel(Time=-1)
        bins = np.arange(-90.0, 91.0, 5.0)
        lat = np.rad2deg(mesh.latCell.values)
        area = mesh.areaCell.values
        nz = output.sizes['nVertLevels']
        section = np.full((nz, len(bins) - 1), np.nan)
        for level in range(nz):
            u, _ = _velocity_layer(mesh, last, level)
            for band in range(len(bins) - 1):
                selected = (lat >= bins[band]) & (lat < bins[band + 1])
                if selected.any():
                    section[level, band] = np.average(
                        u[selected], weights=area[selected]
                    )
        depth = last.zMid.weighted(mesh.areaCell).mean('nCells')
        limit = max(float(np.nanmax(abs(section))), 1.0e-12)
        fig = Figure(figsize=(8, 5), constrained_layout=True)
        ax = fig.add_subplot()
        image = ax.pcolormesh(
            0.5 * (bins[:-1] + bins[1:]),
            depth.values,
            section,
            cmap='RdBu_r',
            vmin=-limit,
            vmax=limit,
            shading='auto',
        )
        ax.set(
            title='Final area-weighted zonal-mean eastward velocity',
            xlabel='Latitude (degrees)',
            ylabel='Mean elevation (m)',
        )
        fig.colorbar(image, ax=ax, label='m s$^{-1}$')
        fig.savefig(self.work_path('velocity_section.png'), dpi=150)


def _velocity_layer(mesh, record, level):
    """Reconstruct one layer without allocating a full stencil/layer array."""
    velocity = record.normalVelocity.isel(nVertLevels=level)
    ux, uy, uz = tangential_reconstruction(
        mesh,
        velocity,
        stencil=mesh.ReconStencilCell,
        weights=mesh.ReconWeightsCell,
    )
    u, v, _ = cartesian_to_local_geographic(mesh, ux, uy, uz)
    return u.transpose('nCells').values, v.transpose('nCells').values
