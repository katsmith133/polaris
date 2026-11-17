(dev-ocean-aquaplanet)=

# aquaplanet

The `aquaplanet` component task implements a wind-driven, three-dimensional global ocean transient. User-facing setup, default experiments, configuration options, and limitations are documented in {ref}`ocean-aquaplanet`.

## registration and configuration

`polaris.tasks.ocean.aquaplanet.add_aquaplanet_tasks()` registers `qu240` and `icos30` under `spherical/aquaplanet/<mesh_name>`. Each task reads the common `aquaplanet.cfg` followed by a mesh-specific overlay. The overlays set mesh family and resolution; they also select the initial 30-day/five-day-output icosahedral experiment and the 365-day/five-day-output quasi-uniform experiment. Both disable the separate high-frequency stream.

The config is shared by the task's `init`, `forward`, and `viz` steps. `Aquaplanet.configure()` rebuilds the steps after user overrides so the selected resolution is reflected in the shared base-mesh step and in the resolution-specific ocean-step work directories. Adding another supported resolution requires a mesh overlay and a registration entry; task paths use the task-first layout, for example `spherical/aquaplanet/icos60`.

## mesh and initialization

`_setup_steps()` calls `add_spherical_base_mesh_step()` with the configured mesh family and resolution. The `Init` step consumes the generated base mesh, graph, and reconstruction weights, then writes the Coriolis field, flat bottom, vertical coordinates, initial state, and forcing. The mesh remains all wet; any future culling must happen before generating the final graph and reconstruction weights.

`initial_state.py` defines helper functions for the positive-down flat bottom, vertically varying conservative temperature, uniform absolute salinity, and steady two-harmonic zonal wind stress. Initial normal velocity and SSH are zero. Explicit zero-valued heat/freshwater/salt forcing fields are included where required by KPP/nonlocal model pathways; they are not additional imposed surface forcing.

## forward

`Forward` descends from `OceanModelStep`, uses `update_eos=True`, and obtains the timestep from either the numeric `time_step` option or `resolution * dt_per_km`. It checks that duration, history cadence, and restart cadence are positive, whole-second stream intervals compatible with the timestep; it also validates the optional bounded diagnostic interval. `compute_cell_count()` supplies the spherical resource heuristic to Polaris.

`forward.yaml` configures split-explicit integration, KPP without Langmuir enhancement, horizontal and background mixing, and implicit quadratic bottom drag. The history stream includes state/tracers, SSH, geometry, kinetic energy, relative vorticity, transports, and KPP/vertical-mixing diagnostics. The MPAS-Ocean output stream and Omega `History` stream are configured with corresponding vorticity and kinetic-energy fields. The high-frequency Omega stream is independently controlled by `diagnostics_enabled` and its date window.

Restart writes use a configurable cadence and replace the single `restart.nc` checkpoint while updating `ocn.pointer`. For Omega continuation, `Forward.dynamic_model_config()` requires the pointer and the referenced checkpoint in the forward work directory before runtime YAML generation. The original model epoch is retained; Omega restores the current simulation time from the restart. The stop duration is per segment, not a cumulative target time.

## visualization and assessment

`Viz` reads the history lazily while the file remains open. Snapshot products include initial/final temperature, temperature change, SSH, vorticity at the surface, and OSBL depth, along with prescribed stress and a zonal-mean temperature section. `relativeVorticity` is mapped from Omega's native `RelVortVertex`; the map is therefore vertex-located and uses dual-cell polygons.

`_plot_kinetic_energy()` integrates the native specific horizontal kinetic-energy field using reference density, cell area, and pseudo-thickness for Omega (or layer thickness when provided). It produces a PNG timeseries and a NetCDF timeseries. `_plot_assessment()` records SSH extrema/RMS, OSBL-depth mean/maximum, minimum layer thickness, and pseudo-thickness-weighted temperature/salinity/mass proxies relative to the initial state. `_plot_velocity_section()` bins reconstructed eastward velocity by latitude and model layer.

The wind-work and bottom-drag-loss estimates reconstruct geographic cell velocities from edge-normal flow. The drag estimate applies the continuum quadratic formula to the bottom-layer speed; it is not the model's discrete implicit drag work. Neither this estimate nor its comparison with kinetic-energy storage closes the mechanical budget, because viscous losses and pressure work/conversion to potential energy are not diagnosed. These plots are comparative assessment aids and must be interpreted with the limitations recorded in `assessment.nc` and the User's Guide.