# AtmosphericTurbulenceSimulator

| | | |
|:---:|:---:|:---:|
| [![docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://aryavorskiy.github.io/AtmosphericTurbulenceSimulator.jl/stable/) | [![docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://aryavorskiy.github.io/AtmosphericTurbulenceSimulator.jl/dev/) | [![DOI](https://zenodo.org/badge/1108761097.svg)](https://doi.org/10.5281/zenodo.20734890) |
| [![CI](https://github.com/aryavorskiy/AtmosphericTurbulenceSimulator.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/aryavorskiy/AtmosphericTurbulenceSimulator.jl/actions/workflows/ci.yml)  | [![codecov](https://codecov.io/gh/aryavorskiy/AtmosphericTurbulenceSimulator.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/aryavorskiy/AtmosphericTurbulenceSimulator.jl) | [![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl) |  |

A simple (yet) Julia toolchain for simulating atmospheric turbulence effects on imaging systems.

## Installation

This package is not registered yet. You can install it with the following command in Julia's REPL:

```julia
using Pkg
Pkg.add(url="https://github.com/aryavorskiy/AtmosphericTurbulenceSimulator.jl")
```

## Quick example

### Turbulent phase generation

The core functionality is generating turbulent phase screens using the `SingleLayer` atmosphere specification.
You can generate phase screens with or without Harding interpolation:

```julia
using AtmosphericTurbulenceSimulator

# With Harding interpolation: samples at low resolution, then upsamples
# Using :auto to determine optimal number of interpolation passes
# Fried parameter r0 = 0.2 m
atm = SingleLayer(0.2; interpolate=:auto)

# Generate phase screens and save to HDF5 (64-pixel grid with 2 m size)
simulate_phases(atm, (64, 64), 2; n=3000, file="phases.h5")
```

The Harding interpolation (from [Harding et al. 1999](https://doi.org/10.1364/AO.38.002161))
allows efficient generation of high-resolution phase screens by sampling the turbulence at a coarser
resolution and upsampling in a way that preserves Kolmogorov statistics.

### PSF simulation with imaging pipeline

To simulate actual images through turbulence, combine the atmosphere specification with an imaging
specification and a true-sky model:

```julia
# Define circular aperture and imaging parameters
ap = CircularAperture((64, 64))
img_spec = ImagingSpec(ap, 2, PhotonCount(1e6, 100))

# Atmosphere specification
atm = SingleLayer(0.2, interpolate=:auto)

# True sky models:
# Point source
ts_point = PointSource()

# Binary system: secondary offset by (35, 15) pixels with 0.3× intensity
ts_double = DoubleSystem((35, 15), 0.3)

# Custom image from array
# ts_image = TrueSkyImage(my_image_array)

# Simulate images and save to HDF5 (includes phase screens by default)
simulate_images(ts_point, atm, img_spec; n=3000, file="simulation.h5")
```

This will create a HDF5 file `simulation.h5` containing 3000 simulated images of the point source
through the turbulent atmosphere, along with the phase screens used. The result can be visualized as follows:
<details>
<summary>Show code</summary>

```julia
using HDF5, CairoMakie, Statistics

img_dataset = h5read("simulation.h5", "images")
first_image = img_dataset[:, :, 1]
first_phase = h5read("simulation.h5", "phases", (:, :, 1))
mean_image = dropdims(mean(img_dataset, dims=3), dims=3)

fig = Figure(size=(900, 300))
ax1, hm  = heatmap(fig[1, 1], first_phase, colormap=:viridis, axis=(aspect=DataAspect(), title="Phase screen"))
hidedecorations!(ax1)
ax2, hm2 = heatmap(fig[1, 2], first_image, colormap=:jet, axis=(aspect=DataAspect(), title="Simulated image"))
hidedecorations!(ax2)
ax3, hm3 = heatmap(fig[1, 3], mean_image, colormap=:jet, axis=(aspect=DataAspect(), title="Accumulated (3000 frames)"))
hidedecorations!(ax3)
Colorbar(fig[1, 0], hm; label="[rad]", ticks = MultiplesTicks(5, π, "π"), flipaxis=false)
fig
```
</details>

![Demo image](demo.svg)

## Notes

This toolchain utilizes Julia's multi-threading capabilities; add more threads by launching Julia with `julia --threads=N`, where `N` is the number of threads you want. You can set `N=auto` to use all available CPU threads.

To enable CUDA (or other GPU backends), import the respective packages (e.g., `CUDA.jl`) and add `deviceadapter=CuArray` (or other device array type) to the `simulate_phases`/`simulate_images` function call. As of current version, only [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) is tested, feel free to open an issue if you encounter problems with other backends.

Please note that this package is in early development. There are multiple features planned for the future:
- More advanced atmosphere models
    - [x] Harding interpolation
    - [x] Frozen flow (long exposures)
    - [ ] Frozen flow (time series)
    - [ ] Multi-layer atmospheres
- Input/Output
    - [x] HDF5 support
    - [x] Phase loader
    - [ ] PSF loader for true-sky application
- [x] Long exposures
- [x] Multi-wavelength imaging
- [ ] Wavefront sensor simulation
