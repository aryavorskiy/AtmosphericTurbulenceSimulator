# Overview

AtmosphericTurbulenceSimulator.jl simulates atmospheric turbulence effects in telescope imaging.
It provides tools for sampling turbulent phase screens, propagating them through a pupil model, and
generating image sequences for point sources, binary systems, and extended sky images.

Features include:
- High-fidelity phase screen sampling with Kolmogorov statistics.
- Flexible true-sky models: point sources, binary systems, user-defined images.
- Photon-counting readout with optional background.
- Non-monochromatic filters with wavelength-dependent turbulence and diffraction.
- Long-exposure simulations with wind-driven phase screen evolution.
- HDF5 output for large batches, or in-memory arrays for interactive work.
- CPU multi-threading and CUDA support.

## Installation

The package is not registered yet. Install it from the Julia REPL with:

```julia
using Pkg
Pkg.add(url="https://github.com/aryavorskiy/AtmosphericTurbulenceSimulator.jl")
```

## Core Workflow

A simulation combines three pieces:

1. A true-sky model, such as [`PointSource`](@ref), [`DoubleSystem`](@ref), or [`TrueSkyImage`](@ref). 
   If none is provided, a point source is assumed.
2. An atmosphere model, such as [`SingleLayer`](@ref), which samples turbulent phase screens.
3. An imaging device specification, [`ImagingSpec`](@ref), which defines the aperture, detector grid,
   photon budget, exposure, and filter.

The main entry point is [`simulate_images`](@ref), which takes these pieces and produces a sequence of
images.

```@example quick_start
using AtmosphericTurbulenceSimulator, Plots

aperture = CircularAperture((64, 64), 30)
img_spec = ImagingSpec(aperture, PhotonCount(1e7, 200); filter=FilterSpec(550, bandwidth=40))
atm = SingleLayer(0.1 / 2 * 64; interpolate=:auto)

result = simulate_images(atm, img_spec; n=8, verbose=false)

p = plot(layout=(2, 4), size=(1600, 800))
for i in 1:8
    heatmap!(p[i], result.images[:, :, i], cmap=:jet, cbar=false, clim=extrema(result.images), aspect_ratio=:equal)
end
p
```

For larger simulations, specify the `file` keyword argument to write results directly to disk. 
See the [Examples](@ref) section for more info.

## Atmosphere Model

The current atmosphere model is a single turbulent layer. Phase covariance follows Kolmogorov
statistics:

```math
D_\phi(r) = \big\langle (\phi(x) - \phi(x+r))^2 \big\rangle
          = 6.88 \left( \frac{r}{r_0} \right)^{5/3}.
```

The Fried parameter ``r_0`` controls turbulence strength. Larger ``r_0`` means weaker phase
aberrations. In this package, pass ``r_0`` in pixels on the phase-screen grid.

For large grids, [`SingleLayer`](@ref) can use Harding interpolation ([Harding et al. 1999](https://doi.org/10.1364/AO.38.002161)). 
The phase is sampled on a smaller grid and then upsampled in a way that preserves Kolmogorov statistics. `interpolate=:auto`
selects a coarse grid size based on the default heuristic.

You can also replay phase screens from a dataset or an array using the [`SavedPhases`](@ref) atmosphere specification.

## Imaging Model

The parameters of the imaging system are defined by an [`ImagingSpec`](@ref) object, which includes the following fields:
- The aperture function, which can be a predefined shape like [`CircularAperture`](@ref) or a user-defined array.
- The photon budget, defined by a [`PhotonCount`](@ref) object that specifies the total number of photons and background level.
- The filter specification, defined by a [`FilterSpec`](@ref) that sets the central wavelength and bandwidth for non-monochromatic simulations.
- The exposure time, which can be used to simulate long exposures by averaging multiple phase screens together. See [`Exposure`](@ref).

The imaging pipeline converts each phase screen into a PSF, applies the selected
true-sky model, and optionally applies photon shot noise. A non-monochromatic [`FilterSpec`](@ref)
scales both turbulence strength and diffraction with wavelength, while assuming that the aperture
itself is achromatic.

Long exposures are also supported by averaging multiple phase screens together. To simulate a long exposure, 
you will need non-zero exposure time and a defined wind velocity in the atmosphere model. See 
[this example](@ref "Variable exposure times") for details.

!!! note
    The sampled-bandpass model is most appropriate for narrow bands where the telescope pupil does
    not vary significantly with wavelength.

## Performance

### Backends

This toolchain is compatible with CPU multi-threading. By default, it uses all available threads. 
To control the number of threads, set the `JULIA_NUM_THREADS` environment variable before starting 
Julia, or start Julia with the `--threads` flag:

```bash
julia --threads=auto    # use all available cores
```

Use [`MultiThreaded`](@ref) for more fine-grained control over the CPU threading behavior, such as 
specifying the number of threads or the array type used for computations:

```julia
simulate_images(atm, img_spec; n=100_000, file="simulation.h5", deviceadapter=MultiThreaded(16))
```

GPU execution is also supported. To use GPU arrays, pass the appropriate device adapter, for example 
`deviceadapter=CuArray`. Note that this requires CUDA.jl and a compatible NVIDIA GPU.

```julia
using CUDA
simulate_images(atm, img_spec; n=100_000, file="simulation.h5", deviceadapter=CuArray)
```

!!! warning
    CUDA.jl is the only GPU backend tested so far. Other Julia GPU array backends may work if
    they provide the required array operations and FFT support.

### Batch Size

The `batch` keyword controls both compute batch size and HDF5 chunk size along the image sequence dimension:

```julia
simulate_images(PointSource(), atm, img_spec; n=10_000, batch=256, file="simulation.h5")
```

The default batch size is 128. Increase it when enough memory is available, especially for many CPU
threads or GPU execution. Decrease it if memory pressure is high.

## Next Steps

- Work through [Examples](@ref) for phase-screen generation, imaging simulations, and HDF5 output.
- See [API Reference](@ref) for constructors and keyword arguments.
