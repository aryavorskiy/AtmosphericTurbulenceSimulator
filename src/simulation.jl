function simulation_run!!(img_bd::BufferedDataset, phs_bd::BufferedDataset, phase_buffers,
        image_buffers, true_sky_adapt, n; verbose=true)
    batch = batch_length(phase_buffers)
    p = Progress(n, desc="Simulating images", enabled=verbose, dt=1)
    for j in 1:cld(n, batch)
        phases = samplephases!(phase_buffers)
        if image_buffers !== nothing
            images = compute_images!(image_buffers, phases, true_sky_adapt)
            write_batch!(img_bd, j, images)
        end
        write_batch!(phs_bd, j, phases)
        next!(p, step=min(batch, n - (j - 1) * batch))
    end
    finish!(p)
end

function simulation_run(file, phsbuffers, imgbuffers, true_sky_adapt, n;
            verbose=true, savephases::Bool=true)
    open_file(file) do fid
        batch = batch_length(phsbuffers)
        if imgbuffers !== nothing
            img_size = image_size(imgbuffers)
            img_bd =
                prepare_dataset(fid, "images", image_type(imgbuffers), img_size, n, batch)
        else
            img_bd = BufferedDataset(nothing)
        end
        if savephases
            phs_size = plate_size(phsbuffers)
            phs_bd =
                prepare_dataset(fid, "phases", phase_type(phsbuffers), phs_size, n, batch)
        else
            phs_bd = BufferedDataset(nothing)
        end
        simulation_run!!(img_bd, phs_bd, phsbuffers, imgbuffers, true_sky_adapt, n;
            verbose=verbose)
        _data(x) = x.dataset isa Array ? x.dataset : nothing
        if imgbuffers === nothing
            return _data(phs_bd)
        else
            out_ntuple = (phases = _data(phs_bd), images = _data(img_bd))
            if all(isnothing, out_ntuple)
                return nothing
            else
                return out_ntuple
            end
        end
    end
end

"""
    simulate_phases(atm_spec::AtmosphereSpec, plate_size; n, [batch, file, verbose, deviceadapter])

Simulate `n` phase screens using the provided atmosphere specification and write
the results to an HDF5 file.

# Arguments
- `atm_spec`: an `AtmosphereSpec` used to produce phase screens.
- `plate_size`: the size of the phase screens to simulate.

# Keyword Arguments
- `n`: number of phase screens to simulate.
- `d`: diameter setting for the phase screen generation (in the same units as ``r_0``). Defaults to the maximum of `plate_size`.
- `batch`: batch size for buffered computations and HDF5 writes (default 128).
- `file`: output options. Can be a string (filename) or an `HDF5File` object. If set to `nothing`
    (default), no file is written and the phases are returned as an array.
- `verbose`: show progress meter (true by default).
- `deviceadapter`: adapter for device-backed arrays (defaults to `Array`). To use GPU arrays,
  pass e.g. `CUDA.CuArray` here (requires CUDA.jl).
"""
function simulate_phases(atm_spec::AtmosphereSpec, plate_size; n::Int, d=maximum(plate_size),
        batch::Int=DEFAULT_BATCH, file=nothing, verbose=true, deviceadapter=Array)
    batch = min(batch, n)
    phase_buffers = prepare_phasebuffers(atm_spec, plate_size, d / maximum(plate_size), batch, deviceadapter)
    simulation_run(file, phase_buffers, nothing, nothing, n; verbose=verbose)
end

"""
    simulate_images([T, true_sky::TrueSky, ]atm_spec::AtmosphereSpec, img_spec::ImagingSpec; \
        n, [batch, file, verbose, savephases, deviceadapter])

Simulate `n` images using the provided imaging and atmosphere specifications and write
the results to an HDF5 file.

# Arguments
- `T`: output image numeric type; if not provided, defaults to `Int` for finite-photon
    simulations (determined by `img_spec.photon_count.nphotons`) and `Float64` for infinite-photon models.
- `true_sky`: a `TrueSky` model (e.g. `PointSource`, `DoubleSystem`, `TrueSkyImage`).
- `atm_spec`: an `AtmosphereSpec` used to produce phase screens.
- `img_spec`: an `ImagingSpec` describing the aperture, image size, photon budget and filter.

# Keyword Arguments
- `n`: number of images to simulate.
- `batch`: batch size for buffered computations and HDF5 writes (default 128).
- `file`: output options. Can be a string (filename) or an `HDF5File` object. If set to `nothing`
    (default), no file is written and the images and phases are returned as a `NamedTuple` of arrays.
- `verbose`: show progress meter (true by default).
- `savephases`: when true (default), the sampled phase screens are saved in the HDF5 in dataset with
  key `"phases"`.
- `deviceadapter`: adapter for device-backed arrays (defaults to `MultiThreaded()`). To use GPU
  arrays, pass e.g. `CUDA.CuArray` here (requires CUDA.jl). To control the number of CPU threads
  used, pass e.g. `MultiThreaded(4)`.
"""
function simulate_images(::Type{T}, true_sky::TrueSky, atm_spec::AtmosphereSpec, img_spec::ImagingSpec;
    n::Int, batch::Int=DEFAULT_BATCH, file=nothing, verbose=true, savephases::Bool=true, deviceadapter=MultiThreaded()) where {T}
    if !isfinite_photons(img_spec.photon_count) && T <: Integer
        throw(ArgumentError("Integer image eltype not compatible with infinite-photon imaging spec."))
    end
    if true_sky isa TrueSkyImage && size(true_sky.true_sky_fft) != image_size(img_spec)
        throw(ArgumentError("TrueSkyImage size $(size(true_sky.true_sky_fft)) does not match " *
            "image size $(image_size(img_spec))."))
    end

    batch = min(batch, n)
    true_sky_adapt = adapt(deviceadapter, true_sky)
    phase_buffers, image_buffers = prepare_buffers(T, atm_spec, img_spec, batch, deviceadapter)
    simulation_run(file, phase_buffers, image_buffers, true_sky_adapt, n;
        verbose=verbose, savephases=savephases)
end
simulate_images(::Type{T}, atm_spec::AtmosphereSpec, img_spec::ImagingSpec; kwargs...) where {T} =
    simulate_images(T, PointSource(), atm_spec, img_spec; kwargs...)
simulate_images(true_sky::TrueSky, atm_spec::AtmosphereSpec, img_spec::ImagingSpec{T}; kwargs...) where {T} =
    simulate_images(isfinite_photons(img_spec.photon_count) ? Int : T, true_sky, atm_spec, img_spec; kwargs...)
simulate_images(atm_spec::AtmosphereSpec, img_spec::ImagingSpec; kwargs...) =
    simulate_images(PointSource(), atm_spec, img_spec; kwargs...)
