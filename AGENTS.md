# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Julia package for simulating atmospheric turbulence effects on imaging systems. Generates
Kolmogorov-statistics phase screens (optionally via Harding interpolation) and propagates them
through a configurable optical pipeline to produce simulated images. Supports CPU multi-threading
and GPU backends (tested with CUDA.jl) via `Adapt.jl`.

## Commands

Run the full test suite:
```
julia --project=test test/runtests.jl
```

Run a single test file (e.g. `test_atmosphere.jl`) without waiting for the full suite:
```
julia --project=test -e 'using Test, AtmosphericTurbulenceSimulator, LinearAlgebra, Statistics, HDF5, ProgressMeter; include("test/test_atmosphere.jl")'
```

When running tests, show all output from `Precompiling packages finished.` onward — everything before it is package-version noise.

## Layout

```
src/
  io.jl           # BufferedDataset, read_batch!, write_batch!, HDF5File, open_file,
                  #   prepare_dataset, simulation_run!!, simulation_run
  atmosphere.jl   # Phase-screen generation: KarhunenLoeveBuffers, HardingInterpolator,
                  #   SavedPhases / SavedPhaseBuffers
  imaging.jl      # Optical pipeline: aperture, PSF computation, true-sky convolution, readout
  simulation.jl   # Public API only: simulate_phases, simulate_images
  precompile.jl   # PrecompileTools workload
test/             # Test suite (runtests.jl + per-module files)
docs/             # Documenter.jl source
contrib/          # Notebooks and benchmarks (not part of the package)
```

**Include order is load-order constrained**: `io.jl` must be included before `atmosphere.jl` because
`SavedPhaseBuffers` (in `atmosphere.jl`) holds a `BufferedDataset` field.

## Architecture

### Phase generation pipeline

`AtmosphereSpec` → `prepare_phasebuffers(spec, plate_size, batch, deviceadapter)` → a sampler struct
→ `samplephases!(sampler)` returns a `(nx, ny, batch)` view into a pre-allocated buffer.

Concrete specs: `SingleLayer` (KL decomposition + optional Harding interpolation),
`SavedPhases` (replays from a dataset).

### Imaging pipeline

`ImagingSpec` + `AtmosphereSpec` → `prepare_buffers(T, atm_spec, img_spec, batch, deviceadapter)`
→ `(phase_buffers, image_buffers)` → `compute_images!(image_buffers, phases, true_sky)`.

On CPU with multiple threads, `prepare_buffers` returns `ImgBufParallel` (one `OpticalBuffers` per
thread); on GPU or single-thread it returns `ImgBufSerial`.

### Simulation loop (`simulation_run!!`)

Drives the loop: calls `samplephases!`, optionally `compute_images!`, then `write_batch!` for both
`phs_bd` and `img_bd`. Both are `BufferedDataset`; passing `BufferedDataset(nothing)` is the no-op
when saving is disabled.

### IO abstraction (`BufferedDataset`)

`BufferedDataset{Dt, Bt}` wraps a dataset and a CPU-side buffer:
- **Write**: `write_batch!(bd, j, data)` — for HDF5 uses `do_write_chunk` (zero-copy); for arrays
  handles boundary truncation.
- **Read**: `read_batch!(dest, bd, j[, ix, iy])` — for HDF5 uses `copyto!(dest, dataset, ix, iy, range)`
  (HDF5.jl overload, no allocation for full batches); partial last batches fall back to standard
  indexing and NaN-pad the remainder.

`SavedPhaseBuffers` wraps a `BufferedDataset` and calls `read_batch!` with stored `crop_indices`
to support replaying a spatially larger saved dataset at a smaller `plate_size`.

## Coding conventions

### Batched operations over loops

Prefer batched array operations instead of scalar loops. If a batched form is impossible,
notify the user before falling back to a loop.

Preferred:
```julia
A .= B[:, range(k+1, length=l)]
```
Not:
```julia
for i in 1:l
    A[:, i] = B[:, i + k]
end
```

### In-place operations in loops

When a loop cannot be avoided, use in-place (`!`) variants and `@views` to avoid allocations:

```julia
@views @. buf[ix, iy, :] += factor * src[ix, iy, :]
```

Use `copy!`, `mul!`, `randn!`, `fill!`, `copyto!` etc. instead of their allocating counterparts.

### GPU compatibility

All performance-critical buffers must be allocated via `similar(existing_array, ...)` or
`Adapt.adapt_storage(deviceadapter, ...)`. Never hard-code `Array{...}(undef, ...)` inside a hot path.

### Function naming

Mutating functions end with `!`. Buffer-preparation functions are named `prepare_*`.
Sampling entry points are named `samplephases!` / `compute_images!`.

### Type parameters

Numeric precision flows from the top-level spec structs (e.g. `ImagingSpec{T}`, `SingleLayer{T}`).
Buffers inherit their element type from the spec; do not hard-code `Float64` inside internal functions.

## Tests

**Do not edit test files or add new tests unless explicitly asked.**

The suite includes JET static analysis (`test_jet.jl`); new public functions should be exercisable
without introducing new JET errors. The JET threshold is currently `<= 22` reports (all from
third-party packages); `@test_broken` marks the ideal target of 0.

## New features — plan before implementing

When asked to develop a new feature, **do not change any code**. Instead, write out a
step-by-step implementation plan and stop. Include:

- Which files and functions will be added or modified.
- The proposed public API (types, function signatures, keyword arguments).
- Any non-obvious design decisions or trade-offs.
- Potential performance concerns and how they will be addressed.

Only proceed with implementation after the plan has been explicitly approved.

## HDF5 output

Datasets are written in chunked form (chunk = one batch) via `BufferedDataset`. When adding
new output fields, follow the `prepare_dataset` / `write_batch!` pattern in `simulation_run` to
keep I/O zero-copy. For reading back saved data, use `read_batch!` with appropriate crop indices.