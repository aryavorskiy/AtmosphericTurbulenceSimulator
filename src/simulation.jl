const DEFAULT_BATCH = 128

struct BufferedDataset{Dt,Bt}
    dataset::Dt
    buffer::Bt
end
_make_buffer(ds::HDF5.Dataset) =
    zeros(eltype(ds), HDF5.get_create_properties(ds).chunk::NTuple{3,Int})
_make_buffer(::Union{<:AbstractArray, Nothing}) = nothing
BufferedDataset(ds1) = BufferedDataset(ds1, _make_buffer(ds1))
function write_batch!(bd::BufferedDataset{<:HDF5.Dataset}, j, batch)
    copy!(bd.buffer, batch)
    HDF5.do_write_chunk(bd.dataset, (1, 1, (j - 1) * size(batch, 3) + 1), bd.buffer)
end
function write_batch!(bd::BufferedDataset{<:AbstractArray}, j, batch)
    batch_len = size(batch, 3)
    dset_len = size(bd.dataset, 3)::Int
    j1 = (j - 1) * batch_len + 1
    if dset_len > j * batch_len
        bd.dataset[:, :, j1:j1 + batch_len - 1] .= batch
    else
        bd.dataset[:, :, j1:end] .= @view batch[:, :, 1:dset_len - j1 + 1]
    end
end
write_batch!(::BufferedDataset{Nothing}, _, _) = nothing

function simulation_run!!(img_bd, phs_bd, phase_buffers, image_buffers, true_sky_adapt, n; verbose=true)
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

open_file(f::Function, filename::String) = h5open(f, filename, "w")
open_file(f::Function, ::Nothing) = f(nothing)
prepare_dataset(fid::HDF5.File, name::String, type, sz, n, batch) =
    BufferedDataset(create_dataset(fid, name, type, (sz..., n), chunk=(sz..., batch)))
prepare_dataset(::Nothing, ::String, ::Type{T}, sz, n, ::Int) where T =
    BufferedDataset(Array{T}(undef, (sz..., n)))
function simulation_run(filename, phsbuffers, imgbuffers, true_sky_adapt, n;
            verbose=true, savephases::Bool=true)
    open_file(filename) do fid
        batch = batch_length(phsbuffers)
        if imgbuffers !== nothing
            img_size = image_size(imgbuffers)
            fid !== nothing && (fid["aperture"] = imgbuffers.spec.aperture)
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
    simulate_phases(atm_spec::AtmosphereSpec; n, [batch, filename, verbose, deviceadapter])

Simulate `n` phase screens using the provided atmosphere specification and write
the results to an HDF5 file.

# Arguments
- `atm_spec`: an `AtmosphereSpec` used to produce phase screens.

# Keyword Arguments
- `n`: number of phase screens to simulate.
- `batch`: batch size for buffered computations and HDF5 writes (default 128).
- `filename`: output HDF5 filename (default "simulation.h5"). If set to `nothing`, no file is written
  and the phases are returned as an array.
- `verbose`: show progress meter (true by default).
- `deviceadapter`: adapter for device-backed arrays (defaults to `Array`). To use GPU arrays,
  pass e.g. `CUDA.CuArray` here (requires CUDA.jl).
"""
function simulate_phases(atm_spec::AtmosphereSpec; n::Int, batch::Int=DEFAULT_BATCH, filename="simulation.h5",
        verbose=true, deviceadapter=Array)
    batch = min(batch, n)
    phase_buffers = prepare_phasebuffers(atm_spec, batch, deviceadapter)
    simulation_run(filename, phase_buffers, nothing, nothing, n; verbose=verbose)
end

"""
    simulate_images([T, ]img_spec::ImagingSpec, atm_spec::AtmosphereSpec[, true_sky::TrueSky]; \
        n, [batch, filename, verbose, savephases, deviceadapter])

Simulate `n` images using the provided imaging and atmosphere specifications and write
the results to an HDF5 file.

# Arguments
- `T`: output image numeric type; if not provided, defaults to `Int` for finite-photon
    simulations (determined by `img_spec.photon_count.nphotons`) and `Float64` for infinite-photon models.
- `img_spec`: an `ImagingSpec` describing the aperture, image size, photon budget and filter.
- `atm_spec`: an `AtmosphereSpec` used to produce phase screens.
- `true_sky`: a `TrueSky` model (e.g. `PointSource`, `DoubleSystem`, `TrueSkyImage`).

# Keyword Arguments
- `n`: number of images to simulate.
- `batch`: batch size for buffered computations and HDF5 writes (default 128).
- `filename`: output HDF5 filename (default "simulation.h5"). If set to `nothing`, no file is written
  and the images and phases are returned as a `NamedTuple` of arrays.
- `verbose`: show progress meter (true by default).
- `savephases`: when true, the sampled phase screens are saved in the HDF5 in dataset with
  key `"phases"`, and the pupil function is saved under key `"aperture"` (true by default).
- `deviceadapter`: adapter for device-backed arrays (defaults to `Array`). To use GPU arrays,
  pass e.g. `CUDA.CuArray` here (requires CUDA.jl).
"""
function simulate_images(::Type{T}, img_spec::ImagingSpec, atm_spec::AtmosphereSpec, true_sky::TrueSky=PointSource();
    n::Int, batch::Int=DEFAULT_BATCH, filename="simulation.h5", verbose=true, savephases::Bool=true, deviceadapter=Array) where {T}
    if !isfinite_photons(img_spec.photon_count) && T <: Integer
        throw(ArgumentError("Integer image eltype not compatible with infinite-photon imaging spec."))
    end
    if plate_size(img_spec) != plate_size(atm_spec)
        throw(ArgumentError("Telescope plate size $(plate_size(img_spec)) does not match" *
            "AtmosphereSpec plate size $(plate_size(atm_spec))."))
    end
    if true_sky isa TrueSkyImage && size(true_sky.true_sky_fft) != image_size(img_spec)
        throw(ArgumentError("TrueSkyImage size $(size(true_sky.true_sky_fft)) does not match " *
            "image size $(image_size(img_spec))."))
    end

    batch = min(batch, n)
    true_sky_adapt = adapt(deviceadapter, true_sky)
    phase_buffers = prepare_phasebuffers(atm_spec, batch, deviceadapter)
    image_buffers = prepare_imgbuffers(T, img_spec, batch, deviceadapter)
    simulation_run(filename, phase_buffers, image_buffers, true_sky_adapt, n;
        verbose=verbose, savephases=savephases)
end
simulate_images(img_spec::ImagingSpec{T}, phase_sampler::AtmosphereSpec, true_sky::TrueSky=PointSource(); kwargs...) where {T} =
    simulate_images(isfinite_photons(img_spec.photon_count) ? Int : T, img_spec, phase_sampler, true_sky; kwargs...)
