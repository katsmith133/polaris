import math

from polaris import Step, Task
from polaris.config import PolarisConfigParser
from polaris.mesh.base import add_spherical_base_mesh_step
from polaris.resolution import resolution_to_string
from polaris.tasks.ocean.aquaplanet.forward import Forward
from polaris.tasks.ocean.aquaplanet.init import Init
from polaris.tasks.ocean.aquaplanet.viz import Viz


def add_aquaplanet_tasks(component):
    """Register mesh-specific variants of the wind-forced global ocean."""
    package = 'polaris.tasks.ocean.aquaplanet'
    for mesh_name in ['qu240', 'icos30']:
        subdir = f'spherical/aquaplanet/{mesh_name}'
        config = PolarisConfigParser(
            filepath=f'{component.name}/{subdir}/aquaplanet.cfg'
        )
        config.add_from_package('polaris.ocean.eos', 'teos10.cfg')
        config.add_from_package(package, 'aquaplanet.cfg')
        config.add_from_package(package, f'{mesh_name}.cfg')
        component.add_task(Aquaplanet(component, config, subdir))


class Aquaplanet(Task):
    """A flat-bottom aquaplanet with four sequential steps.

    Resolution is selected during setup. Changing it requires a fresh setup,
    not just editing the configuration of a previously staged forward step.
    """

    def __init__(self, component, config, subdir):
        mesh_name = subdir.rsplit('/', maxsplit=1)[-1]
        name = f'aquaplanet_{mesh_name}'
        super().__init__(component=component, name=name, subdir=subdir)
        self.set_shared_config(config)
        self._setup_steps()

    def configure(self):
        """Rebuild the steps after applying user configuration overrides."""
        super().configure()
        self._setup_steps()

    def _setup_steps(self):
        prefix = self.config.get('aquaplanet', 'mesh_type')
        if prefix not in {'qu', 'icos'}:
            raise ValueError("aquaplanet:mesh_type must be 'qu' or 'icos'")
        resolution = self.config.getfloat('aquaplanet', 'resolution')
        if not math.isfinite(resolution) or resolution <= 0.0:
            raise ValueError(
                'aquaplanet:resolution must be finite and positive'
            )
        res_str = resolution_to_string(resolution)
        for step in list(self.steps.values()):
            self.remove_step(step)

        mesh, _ = add_spherical_base_mesh_step(prefix, min_res=resolution)
        self.add_step(mesh, symlink='mesh')
        init = Init(
            self.component,
            mesh=mesh,
            subdir=f'{self.subdir}/init/{res_str}',
        )
        forward = Forward(
            self.component,
            init=init,
            resolution=resolution,
            subdir=f'{self.subdir}/forward/{res_str}',
        )
        viz = Viz(
            self.component,
            init=init,
            forward=forward,
            subdir=f'{self.subdir}/viz/{res_str}',
        )
        ocean_steps: list[Step] = [init, forward, viz]
        for step in ocean_steps:
            step.set_shared_config(self.config, link=self.config_filename)
            self.add_step(step)
