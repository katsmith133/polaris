import os

from polaris import Task
from polaris.tasks.ocean.vert_mix.forcing import Forcing
from polaris.tasks.ocean.vert_mix.forward import Forward
from polaris.tasks.ocean.vert_mix.init import Init
from polaris.tasks.ocean.vert_mix.viz import Viz


def add_vert_mix_tasks(component):
    """
    Add vertically mixed planar-column tasks

    Parameters
    ----------
    component : polaris.tasks.ocean.Ocean
        the ocean component that the tasks will be added to
    """
    component.add_task(VertMix(component=component, forcing_type='wind'))


class VertMix(Task):
    """
    A planar vertical-mixing task with one independent water column per cell
    and a sweep over forcing amplitudes.
    """

    def __init__(self, component, forcing_type='wind'):
        name = forcing_type
        subdir = os.path.join('planar', 'vert_mix', name)
        super().__init__(component=component, name=name, subdir=subdir)

        self.config.add_from_package(
            'polaris.tasks.ocean.vert_mix', 'vert_mix.cfg'
        )
        self.config.add_from_package(
            f'polaris.tasks.ocean.vert_mix.{forcing_type}',
            f'{forcing_type}.cfg',
        )

        init = Init(component=component, indir=self.subdir)
        self.add_step(init)

        forcing_values = _get_forcing_values(self.config, forcing_type)
        forward_steps = []
        for index, forcing_value in enumerate(forcing_values):
            label = _format_value_label(forcing_value)
            forcing_name = f'forcing_{index:03d}_{label}'
            forcing_step = Forcing(
                component=component,
                forcing_type=forcing_type,
                forcing_value=forcing_value,
                name=forcing_name,
                indir=self.subdir,
            )
            self.add_step(forcing_step)

            forward_name = f'forward_{index:03d}_{label}'
            forward_step = Forward(
                component=component,
                forcing_type=forcing_type,
                forcing_step=forcing_step,
                forcing_value=forcing_value,
                name=forward_name,
                indir=self.subdir,
            )
            self.add_step(forward_step)
            forward_steps.append(forward_step)

        self.add_step(
            Viz(
                component=component,
                indir=self.subdir,
                forcing_type=forcing_type,
                forward_steps=forward_steps,
            )
        )


def _format_value_label(value):
    label = f'{value:.6g}'
    return label.replace('-', 'm').replace('.', 'p')


def _get_forcing_values(config, forcing_type):
    section = config[f'vert_mix_{forcing_type}']
    start = section.getfloat('start')
    stop = section.getfloat('stop')
    increment = section.getfloat('increment')

    if increment <= 0.0:
        raise ValueError('Forcing increment must be positive.')
    if stop < start:
        raise ValueError(
            'Forcing stop must be greater than or equal to start.'
        )

    count = int(round((stop - start) / increment)) + 1
    values = [start + index * increment for index in range(count)]
    if abs(values[-1] - stop) > 1.0e-12:
        raise ValueError(
            'Forcing range must be exactly divisible by increment.'
        )
    return values
