import math

import matplotlib.pyplot as plt
import numpy as np

from polaris.ocean.model import OceanIOStep, get_days_since_start
from polaris.viz import use_mplstyle


class Viz(OceanIOStep):
    """
    Plot temperature and salinity profile families over the parameter grid.
    """

    def __init__(self, component, indir, forcing_type, forward_steps):
        super().__init__(component=component, name='viz', indir=indir)
        self.forcing_type = forcing_type
        self.forward_steps = list(forward_steps)

        self.add_input_file(filename='init.nc', target='../init/init.nc')
        for step in self.forward_steps:
            self.add_input_file(
                filename=f'{step.name}.nc',
                target=f'../{step.name}/output.nc',
            )

        self.add_output_file(filename='temperature.png')
        self.add_output_file(filename='salinity.png')

    def run(self):
        use_mplstyle()
        section = self.config['vert_mix']
        t_target = section.getfloat('run_duration')

        ds_init = self.open_model_dataset('init.nc', config=self.config).isel(
            Time=0
        )
        nx = int(ds_init.attrs['nx'])
        ny = int(ds_init.attrs['ny'])

        outputs = []
        for step in self.forward_steps:
            ds = self.open_model_dataset(f'{step.name}.nc', config=self.config)
            t_arr = get_days_since_start(ds)
            t_index = np.argmin(np.abs(t_arr - t_target))
            outputs.append(
                (step.name, float(t_arr[t_index]), ds.isel(Time=t_index))
            )

        for field_name, units in [
            ('temperature', 'degC'),
            ('salinity', 'PSU'),
        ]:
            fig, axes = plt.subplots(
                ny,
                nx,
                figsize=(max(4, 2.5 * nx), max(4, 2.8 * ny)),
                squeeze=False,
                sharex=False,
                sharey=True,
            )
            z_init = _get_depth_array(ds_init)

            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(outputs)))
            for cell_index in range(ds_init.sizes['nCells']):
                x_index = cell_index % nx
                y_index = cell_index // nx
                ax = axes[y_index, x_index]

                initial_profile = _get_cell_profile(
                    ds_init, field_name, cell_index
                )
                ax.plot(
                    initial_profile,
                    z_init[cell_index, :],
                    '--k',
                    label='initial',
                )

                for color, (step_name, t_days, ds_out) in zip(
                    colors, outputs, strict=False
                ):
                    z_final = _get_depth_array(ds_out, default=z_init)
                    final_profile = _get_cell_profile(
                        ds_out, field_name, cell_index
                    )
                    ax.plot(
                        final_profile,
                        z_final[cell_index, :],
                        color=color,
                        label=f'{step_name}, {t_days:.2f} d',
                    )

                ax.set_title(f'x={x_index + 1}, y={y_index + 1}')
                if y_index == ny - 1:
                    ax.set_xlabel(f'{field_name} ({units})')
                if x_index == 0:
                    ax.set_ylabel('z (m)')

            handles, labels = axes[0, 0].get_legend_handles_labels()
            ncol = max(1, math.ceil(len(labels) / 8))
            fig.legend(
                handles,
                labels,
                loc='upper center',
                bbox_to_anchor=(0.5, 0.0),
                frameon=False,
                ncol=ncol,
            )
            fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
            fig.savefig(f'{field_name}.png', dpi=200, bbox_inches='tight')
            plt.close(fig)


def _get_depth_array(ds, default=None):
    """
    Return depth as a (nCells, nVertLevels) ndarray.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset that may include zMid or refZMid

    default : ndarray, optional
        Fallback depth array used if neither zMid nor refZMid is present
    """
    z = None
    for name in ['zMid', 'refZMid']:
        if name in ds:
            z = ds[name]
            break

    if z is None:
        if default is None:
            raise ValueError(
                'Neither zMid nor refZMid are present in the dataset.'
            )
        return default

    if 'Time' in z.dims:
        z = z.isel(Time=0)

    values = z.values
    if values.ndim == 1:
        # Broadcast a 1D vertical coordinate across all cells.
        values = np.broadcast_to(values, (ds.sizes['nCells'], values.size))
    elif values.ndim != 2:
        raise ValueError('Unexpected depth variable dimensions.')

    return values


def _get_cell_profile(ds, tracer_name, cell_index):
    """
    Get one vertical tracer profile for a given cell, accounting for
    model-dependent variable and dimension names.
    """
    if tracer_name == 'temperature':
        candidates = ['temperature', 'Temperature']
    elif tracer_name == 'salinity':
        candidates = ['salinity', 'Salinity']
    else:
        candidates = [tracer_name]

    da = None
    for name in candidates:
        if name in ds:
            da = ds[name]
            break

    if da is None:
        # Some outputs store active tracers in a bundled array instead of
        # scalar temperature/salinity variables.
        for bundle_name in ['tracers', 'Tracers']:
            if bundle_name in ds:
                da = ds[bundle_name]
                tracer_index = 0 if tracer_name == 'temperature' else 1
                for tracer_dim in ['nTracers', 'NTracers']:
                    if tracer_dim in da.dims:
                        da = da.isel({tracer_dim: tracer_index})
                        break
                break

    if da is None:
        raise KeyError(
            f'None of {candidates} nor tracer bundles were found in '
            f'dataset variables: {list(ds.data_vars.keys())}'
        )

    if 'Time' in da.dims:
        da = da.isel(Time=0)

    for cell_dim in ['nCells', 'NCells']:
        if cell_dim in da.dims:
            return da.isel({cell_dim: cell_index}).values

    raise ValueError(
        f'Could not find cell dimension in tracer {da.name}; '
        f'found dims {da.dims}'
    )
