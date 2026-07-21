from polaris.ocean.model import OceanModelStep, get_time_interval_string


class Forward(OceanModelStep):
    """
    A forward run for one forcing amplitude in the planar vertical-mixing
    test group.
    """

    def __init__(
        self,
        component,
        forcing_type,
        forcing_step,
        forcing_value,
        name,
        indir,
        ntasks=1,
        min_tasks=1,
        openmp_threads=1,
    ):
        super().__init__(
            component=component,
            name=name,
            indir=indir,
            ntasks=ntasks,
            min_tasks=min_tasks,
            openmp_threads=openmp_threads,
        )

        self.forcing_type = forcing_type
        self.forcing_step = forcing_step
        self.forcing_value = forcing_value

        self.add_yaml_file('polaris.ocean.config', 'output.yaml')

        self.add_horiz_mesh_input_file(target='../init/culled_mesh.nc')
        self.add_vert_coord_input_file(target='../init/vert_coord.nc')
        self.add_init_input_file(target='../init/init.nc')
        self.add_input_file(
            filename='forcing.nc',
            target=f'../{forcing_step.name}/forcing.nc',
        )
        self.add_input_file(
            filename='graph.info', target='../init/culled_graph.info'
        )
        self.add_output_file(filename='output.nc')

    def setup(self):
        super().setup()
        model = self.config.get('ocean', 'model')
        if model == 'omega':
            self.add_input_file(filename='OmegaMesh.nc', target='init.nc')
            self.add_input_file(
                target='coeffs.nc',
                filename='coeffs.nc',
                database_component='ocean',
                database='single_column',
            )

    def dynamic_model_config(self, at_setup):
        super().dynamic_model_config(at_setup=at_setup)

        section = self.config['vert_mix']
        dt = section.getfloat('dt_seconds')
        btr_dt = section.getfloat('btr_dt_seconds')
        output_interval = section.getfloat('output_interval_seconds')
        duration = section.getfloat('run_duration')
        time_integrator = section.get('time_integrator')
        time_integrator_map = {'RK4': 'RungeKutta4'}

        model = self.config.get('ocean', 'model')
        duration_str = get_time_interval_string(days=duration)
        dt_str = get_time_interval_string(seconds=dt)
        btr_dt_str = get_time_interval_string(seconds=btr_dt)
        if model == 'omega':
            output_interval_str = str(int(round(output_interval)))
        else:
            output_interval_str = get_time_interval_string(
                seconds=output_interval
            )

        if model == 'omega':
            time_integrator = time_integrator_map.get(
                time_integrator, time_integrator
            )

        shared_options = {
            'config_time_integrator': time_integrator,
            'config_run_duration': duration_str,
            'config_dt': dt_str,
            'config_btr_dt': btr_dt_str,
            'config_use_cvmix_convection': True,
            'config_use_cvmix_shear': True,
            'config_vert_coord_movement': 'impermeable_interfaces',
            'config_disable_thick_hadv': True,
            'config_disable_thick_vadv': True,
            'config_disable_thick_sflux': True,
            'config_disable_vel_hadv': True,
            'config_disable_vel_hmix': True,
            'config_disable_vel_pgrad': True,
            'config_disable_vel_coriolis': True,
            'config_disable_vel_explicit_bottom_drag': True,
            'config_disable_tr_hadv': True,
            'config_disable_tr_hmix': True,
            'config_disable_tr_nonlocalflux': True,
        }
        mpas_options = {
            'config_use_bulk_wind_stress': self.forcing_type == 'wind',
            'config_use_activeTracers_surface_bulk_forcing': False,
            'config_use_activeTracers_surface_restoring': False,
            'config_use_activeTracers_interior_restoring': False,
        }
        omega_options = {
            'VelDiffTendencyEnable': False,
            'VelHyperDiffTendencyEnable': False,
            'TracerHorzAdvTendencyEnable': False,
            'TracerDiffTendencyEnable': False,
            'TracerHyperDiffTendencyEnable': False,
            'PressureGradTendencyEnable': False,
            'VelocityVertAdvTendencyEnable': False,
            'TracerVertAdvTendencyEnable': False,
        }

        self.add_yaml_file(
            'polaris.tasks.ocean.vert_mix',
            'forward.yaml',
            template_replacements={
                'output_interval': output_interval_str,
            },
        )
        self.add_model_config_options(shared_options, config_model='ocean')
        self.add_model_config_options(mpas_options, config_model='mpas-ocean')
        self.add_model_config_options(omega_options, config_model='Omega')
