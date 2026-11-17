import math
from pathlib import Path

import numpy as np
from mpas_tools.io import open_dataset

from polaris.ocean.model import OceanModelStep, get_time_interval_string


class Forward(OceanModelStep):
    """Run a wind-only, three-dimensional ocean with KPP mixing."""

    def __init__(self, component, init, resolution, subdir):
        super().__init__(
            component=component,
            name='forward',
            subdir=subdir,
            openmp_threads=1,
            graph_target=f'{init.path}/graph.info',
            update_eos=True,
        )
        self.resolution = resolution
        self.init_step = init
        self.add_output_file(
            'output.nc',
            validate_vars=['temperature', 'salinity', 'normalVelocity'],
        )
        self.add_yaml_file('polaris.ocean.config', 'output.yaml')

    def setup(self):
        init = self.init_step
        self.add_horiz_mesh_input_file(
            work_dir_target=f'{init.path}/{self.get_horiz_mesh_filename()}'
        )
        self.add_vert_coord_input_file(
            work_dir_target=f'{init.path}/{self.get_vert_coord_filename()}'
        )
        self.add_init_input_file(
            work_dir_target=f'{init.path}/{self.get_init_filename()}'
        )
        self.add_input_file(
            filename=self.get_forcing_filename(),
            work_dir_target=f'{init.path}/{self.get_forcing_filename()}',
        )
        if self.config.get(
            'ocean', 'model'
        ) == 'omega' and self.config.getboolean(
            'aquaplanet', 'diagnostics_enabled'
        ):
            self.add_output_file('diagnostics.nc')
        super().setup()

    def compute_cell_count(self):
        """Use the existing spherical convergence resource heuristic."""
        return max(1, math.ceil(6.0e8 / self.resolution**2))

    def dynamic_model_config(self, at_setup):
        super().dynamic_model_config(at_setup)
        section = self.config['aquaplanet']
        dt_option = section.get('time_step').strip().lower()
        if dt_option == 'auto':
            dt = self.resolution * section.getfloat('dt_per_km')
        else:
            dt = float(dt_option)
        duration = 86400.0 * section.getfloat('run_duration')
        output_interval = section.getfloat('output_interval')
        restart_interval = section.getfloat('restart_interval')
        substeps = section.getint('barotropic_substeps')
        if (
            not np.all(np.isfinite([dt, duration, output_interval]))
            or min(dt, duration, output_interval, substeps) <= 0
        ):
            raise ValueError(
                'Time intervals and substep count must be positive'
            )
        for name, interval in [
            ('run_duration', duration),
            ('output_interval', output_interval),
            ('restart_interval', restart_interval),
        ]:
            if not np.isfinite(interval) or interval <= 0.0:
                raise ValueError(f'aquaplanet:{name} must be positive')
            if not np.isclose(interval / dt, round(interval / dt)):
                raise ValueError(f'aquaplanet:{name} must be a multiple of dt')
        if output_interval > duration:
            raise ValueError('output_interval must not exceed run duration')
        if restart_interval > duration:
            raise ValueError('restart_interval must not exceed run duration')
        start_type = section.get('start_type').strip()
        if start_type not in {'StartUp', 'Continue'}:
            raise ValueError(
                "aquaplanet:start_type must be 'StartUp' or 'Continue'"
            )
        for name, interval in [
            ('run_duration', duration),
            ('output_interval', output_interval),
            ('restart_interval', restart_interval),
        ]:
            if not interval.is_integer() or interval > np.iinfo(np.int32).max:
                raise ValueError(
                    f'aquaplanet:{name} must give a whole-second interval '
                    'within the signed 32-bit stream-frequency range'
                )

        replacements = {
            'dt': get_time_interval_string(seconds=dt),
            'btr_dt': get_time_interval_string(seconds=dt / substeps),
            'duration': get_time_interval_string(seconds=duration),
            'duration_seconds': str(int(duration)),
            'restart_interval': get_time_interval_string(
                seconds=restart_interval
            ),
            'restart_seconds': str(int(restart_interval)),
            'output_interval': get_time_interval_string(
                seconds=output_interval
            ),
            'output_seconds': str(int(output_interval)),
            'forcing_filename': self.get_forcing_filename(),
            'start_type': start_type,
        }
        if start_type == 'Continue' and not at_setup:
            pointer = Path('ocn.pointer')
            if not pointer.is_file():
                raise FileNotFoundError(
                    'Continue mode requires ocn.pointer in the forward '
                    'work directory'
                )
            restart_name = pointer.read_text().strip()
            if not restart_name or not Path(restart_name).is_file():
                raise FileNotFoundError(
                    f'ocn.pointer references a missing restart: '
                    f'{restart_name!r}'
                )
        replacements.update(self._diagnostic_replacements(dt))
        viscosity_per_km = section.getfloat('horizontal_viscosity_per_km')
        if not np.isfinite(viscosity_per_km) or viscosity_per_km < 0.0:
            raise ValueError(
                'aquaplanet:horizontal_viscosity_per_km must be '
                'finite and nonnegative'
            )
        replacements['horizontal_viscosity'] = (
            f'{viscosity_per_km * self.resolution:g}'
        )
        replacements.update(self._tracer_diffusivity_replacements())
        replacements.update(self._del4_viscosity_replacements())
        for option in [
            'background_viscosity',
            'background_diffusivity',
            'bottom_drag_coefficient',
        ]:
            value = section.getfloat(option)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f'aquaplanet:{option} must be nonnegative')
            replacements[option] = f'{value:g}'
        self.add_yaml_file(
            'polaris.tasks.ocean.aquaplanet',
            'forward.yaml',
            template_replacements=replacements,
        )

    def _del4_viscosity_replacements(self):
        """Return validated biharmonic viscosity options for both models."""
        section = self.config['aquaplanet']
        viscosity_del4_per_km3 = section.getfloat(
            'horizontal_viscosity_del4_per_km3'
        )
        use_viscosity_del4 = section.getboolean(
            'horizontal_viscosity_del4_enabled'
        )
        if (
            not np.isfinite(viscosity_del4_per_km3)
            or viscosity_del4_per_km3 < 0.0
        ):
            raise ValueError(
                'aquaplanet:horizontal_viscosity_del4_per_km3 must be '
                'finite and nonnegative'
            )
        if use_viscosity_del4 and viscosity_del4_per_km3 <= 0.0:
            raise ValueError(
                'A positive horizontal_viscosity_del4_per_km3 is required '
                'when horizontal_viscosity_del4_enabled is true'
            )
        return {
            'use_horizontal_viscosity_del4': str(use_viscosity_del4).lower(),
            'horizontal_viscosity_del4': (
                f'{viscosity_del4_per_km3 * self.resolution**3:g}'
                if use_viscosity_del4
                else '0.0'
            ),
        }

    def _tracer_diffusivity_replacements(self):
        """Return validated resolution-scaled tracer del2 options."""
        section = self.config['aquaplanet']
        diffusivity_per_km = section.getfloat('horizontal_diffusivity_per_km')
        use_diffusivity = section.getboolean('horizontal_diffusivity_enabled')
        if not np.isfinite(diffusivity_per_km) or diffusivity_per_km < 0.0:
            raise ValueError(
                'aquaplanet:horizontal_diffusivity_per_km must be '
                'finite and nonnegative'
            )
        if use_diffusivity and diffusivity_per_km <= 0.0:
            raise ValueError(
                'A positive horizontal_diffusivity_per_km is required '
                'when horizontal_diffusivity_enabled is true'
            )
        return {
            'use_horizontal_diffusivity': str(use_diffusivity).lower(),
            'horizontal_diffusivity': (
                f'{diffusivity_per_km * self.resolution:g}'
                if use_diffusivity
                else '0.0'
            ),
        }

    def _diagnostic_replacements(self, dt):
        """Configure the bounded Omega diagnostic stream independently."""
        section = self.config['aquaplanet']
        enabled = section.getboolean('diagnostics_enabled')
        interval = section.getfloat('diagnostics_interval')
        if enabled and (
            not np.isfinite(interval)
            or interval <= 0.0
            or not interval.is_integer()
            or interval > np.iinfo(np.int32).max
            or not np.isclose(interval / dt, round(interval / dt))
        ):
            raise ValueError(
                'diagnostics_interval must be positive whole seconds, '
                'within signed 32-bit range and a multiple of dt'
            )
        return {
            'diagnostics_seconds': str(int(interval)) if enabled else '1',
            'diagnostics_units': 'Seconds' if enabled else 'Never',
            'diagnostics_start': section.get('diagnostics_start_time'),
            'diagnostics_end': section.get('diagnostics_end_time'),
        }

    def run(self):
        # Omega falls back to zero forcing after a read failure. Check the
        # staged file ourselves instead of silently accepting a missing wind.
        with self.open_model_dataset(self.get_forcing_filename()) as forcing:
            for name in ['windStressZonal', 'windStressMeridional']:
                if not np.all(np.isfinite(forcing[name].values)):
                    raise ValueError(f'Nonfinite forcing field: {name}')
            amplitude = self.config.getfloat(
                'aquaplanet', 'wind_stress_amplitude'
            )
            if amplitude != 0.0 and not bool(
                (abs(forcing.windStressZonal) > 0.0).any()
            ):
                raise ValueError('The staged wind stress is unexpectedly zero')
        with open_dataset(self.get_horiz_mesh_filename()) as mesh:
            self.logger.info('Aquaplanet horizontal mesh: %s', mesh.sizes)
        super().run()
