(ocean-aquaplanet)=

# aquaplanet

## description

The `aquaplanet` tasks are global, all-ocean idealized simulations of a transient response to steady zonal wind stress. They use a flat bottom, initially stratified temperature, uniform salinity, K-profile parameterization (KPP) vertical mixing, and quadratic bottom drag. This is a wind-driven ocean experiment, not a fully forced aquaplanet climate simulation or an equilibrium test.

Two mesh-specific tasks are registered:

| Task | Mesh | Default duration | History output |
| --- | --- | --- | --- |
| `ocean/spherical/aquaplanet/icos30` | Icosahedral, 30 km | 30 days | Every 5 days |
| `ocean/spherical/aquaplanet/qu240` | Quasi-uniform, 240 km | 365 days | Every 5 days |

For example, set up the icosahedral task with Omega using:

```bash
polaris setup -t ocean/spherical/aquaplanet/icos30 -w <work_dir> --model omega
```

Replace the task path with `ocean/spherical/aquaplanet/qu240` to set up the quasi-uniform case. Both tasks also support MPAS-Ocean. Configure task options in the generated task config under the work directory before setup or before running the relevant steps. A new task setup is needed when changing the mesh resolution, because the mesh and resolution-specific inputs are selected at setup time.

## supported models

These tasks support both MPAS-Ocean and Omega.

The history includes relative vorticity, kinetic energy, sea-surface height, temperature, salinity, boundary-layer depth, and other KPP diagnostics. Relative vorticity is stored at vertices. The visualization step produces a map of final surface relative vorticity and integrates a global kinetic-energy time series from the specific kinetic-energy field and layer thickness.

## mesh

The mesh is global and spherical. The `icos30` task uses an approximately 30 km icosahedral mesh, and `qu240` uses an approximately 240 km quasi-uniform mesh. The initial-state step creates an all-wet ocean with a flat 4000 m bottom; no land mask or topographic culling is applied.

## vertical grid

The default grid is a 32-layer z-star grid with a tanh-based vertical distribution, 4000 m bottom depth, and nominal layer thickness limits of 10–250 m. The aquaplanet initial-state step requires finite positive thickness in every active layer. The vertical-grid options are supplied by the common `[vertical_grid]` config section.

## initial conditions

The initial sea-surface height and normal velocity are zero. Salinity is horizontally and vertically uniform at 35 g kg$^{-1}$. Conservative temperature varies with depth as

$$
T(z) = 4 + 16\exp\left(\frac{z}{1000\ \mathrm{m}}\right)\ ^\circ\mathrm{C},
$$

where $z$ is the vertical coordinate, negative below the surface. This gives approximately 20 °C at the surface and approaches 4 °C at depth. The initial profile is checked for static stability.

## forcing

The imposed steady zonal stress is

$$
\tau_x(\phi) = \tau_0\left[\sin(2\phi) + 0.5\sin(6\phi)\right],
\qquad \tau_y = 0,
$$

where $\phi$ is geographic latitude in radians and the default $\tau_0$ is 0.1 N m$^{-2}$. There is no imposed heat, freshwater, or salt flux and no surface tracer restoring. KPP is enabled; Langmuir turbulence is disabled. Zero-valued tracer-forcing input fields are provided for model/KPP compatibility and do not represent additional applied surface forcing.

## time step and run duration

With `time_step = auto`, the timestep is `resolution * dt_per_km`, with `dt_per_km = 2.5 s km-1`. The defaults are therefore 75 s for `icos30` and 600 s for `qu240`. This scaling is a starting point, not a stability guarantee; assess timestep suitability for other resolutions.

The Laplacian momentum-viscosity coefficient scales linearly with the nominal resolution, using `horizontal_viscosity_per_km`. For Omega, `MeshScalingEnable` also applies a local correction based on edge cell width relative to the mesh median. Thus the configured coefficient sets the run-wide scale, while the Omega operator adjusts it locally for mesh-width variation. The local Omega option does not apply to MPAS-Ocean.

Biharmonic momentum viscosity is enabled and scales as $\nu_4 = c_4\Delta x^3$. Its coefficient matches realistic_global's icos30km value at nominal 30 km. Omega additionally applies its local cubic mesh-width correction through `MeshScalingEnable`.

Harmonic horizontal tracer diffusivity is disabled by default to match realistic_global. To enable it for both MPAS-Ocean and Omega, set `horizontal_diffusivity_enabled = True`; `horizontal_diffusivity_per_km` then sets a coefficient that scales linearly with resolution. The retained coefficient gives 10 m$^2$ s$^{-1}$ at 240 km and 1.25 m$^2$ s$^{-1}$ at 30 km. This is separate from KPP and background vertical diffusivity.

The default `icos30` run is 30 days, and `qu240` is 365 days. Both write history every five days and checkpoints every five days. The separate high-frequency diagnostic stream is disabled in both mesh-specific configs. Restart cadence is independent of history cadence. For Omega continuation, set `start_type = Continue` in the forward-step config and retain a valid `ocn.pointer` and its checkpoint in the forward work directory; the configured duration is the duration of that continuation segment.

## config options

Common physics and run options are in `aquaplanet.cfg`; mesh-specific defaults are overlaid from `icos30.cfg` or `qu240.cfg`. Important options in `[aquaplanet]` include:

| Option | Meaning | Shared/default value |
| --- | --- | --- |
| `wind_stress_amplitude` | $\tau_0$ in the stress formula | 0.1 N m$^{-2}$ |
| `deep_temperature`, `temperature_contrast`, `temperature_scale` | Parameters in the initial temperature profile | 4 °C, 16 °C, 1000 m |
| `salinity` | Uniform initial salinity | 35 g kg$^{-1}$ |
| `run_duration` | Forward-segment duration in days | 30 (`icos30`); 365 (`qu240`) |
| `output_interval` | History cadence in seconds | 432000 (5 days) |
| `restart_interval` | Checkpoint cadence in seconds | 432000 (5 days) |
| `start_type` | Omega startup mode, `StartUp` or `Continue` | `StartUp` |
| `time_step`, `dt_per_km` | Timestep in seconds or automatic resolution scaling | `auto`, 2.5 s km$^{-1}$ |
| `bottom_drag_coefficient` | Quadratic bottom-drag coefficient | $10^{-3}$ |
| `horizontal_viscosity_per_km` | Laplacian horizontal momentum viscosity coefficient, scaled linearly with resolution | 33.3333 m$^2$ s$^{-1}$ km$^{-1}$; gives 8000 m$^2$ s$^{-1}$ at 240 km and 1000 m$^2$ s$^{-1}$ at 30 km |
| `horizontal_viscosity_del4_enabled` | Enable biharmonic horizontal momentum viscosity | True |
| `horizontal_viscosity_del4_per_km3` | Biharmonic viscosity coefficient, scaled with resolution cubed | 4.4444 × 10$^6$ m$^4$ s$^{-1}$ km$^{-3}$; gives 6.144 × 10$^{13}$ m$^4$/s at 240 km and 1.2 × 10$^{11}$ m$^4$/s at 30 km |
| `horizontal_diffusivity_enabled` | Enable harmonic horizontal tracer diffusivity | False |
| `horizontal_diffusivity_per_km` | Harmonic horizontal tracer diffusivity per km, scaled linearly with resolution | 0.0416667 m$^2$ s$^{-1}$ km$^{-1}$ when enabled; gives 10 m$^2$ s$^{-1}$ at 240 km and 1.25 m$^2$ s$^{-1}$ at 30 km |
| `background_viscosity`, `background_diffusivity` | Background vertical mixing coefficients | $10^{-4}$ m$^2$ s$^{-1}$, $10^{-5}$ m$^2$ s$^{-1}$ |
| `diagnostics_enabled` | Enable the separate bounded high-frequency diagnostic stream (Omega only) | False in both mesh overlays |

The mesh-specific config also sets `mesh_type` (`icos` or `qu`) and `resolution` in km. The `aquaplanet_*_viz` sections configure temperature, anomaly, KPP-depth, and vorticity maps. The vorticity color range is selected symmetrically from the plotted final surface field.

## outputs and interpretation

The `viz` step writes initial and final surface-temperature maps, surface-temperature change, sea-surface height, prescribed wind stress, a final zonal-mean temperature section, KPP boundary-layer depth, final surface relative vorticity, and a global kinetic-energy time series. It also writes `surface_diagnostics.nc` with area-weighted surface temperature and salinity means, and `assessment.nc` with SSH, boundary-layer, minimum-thickness, and inventory time series.

The assessment includes approximate wind-work and bottom-drag-loss estimates and a final zonal-mean velocity section. These use reconstructed cell-centered velocities and continuum formulas; they are not Omega's discrete implicit drag work or a closed mechanical-energy budget. In particular, viscous dissipation and pressure work/conversion to potential energy are not included. Inventory diagnostics use pseudo-thickness for Omega and should not be interpreted as a full thermodynamic or mechanical-energy budget.

## cores

The forward step estimates cell count as `ceil(6.0e8 / resolution**2)` and lets Polaris choose MPI task counts using the machine and ocean-component cells-per-core settings. The icosahedral 30 km mesh therefore needs substantially more resources than the 240 km quasi-uniform mesh. Check the generated job script and adjust the work-directory configuration for the target machine and requested wall time.