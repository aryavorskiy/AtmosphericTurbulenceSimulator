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

struct HDF5File
    filename::String
    group::String
    overwrite::Bool
end

"""
    HDF5File(filename[; group="", overwrite=false])

A convenience struct for specifying HDF5 output options.

# Arguments
- `filename`: name of the HDF5 file to write to.
- `group`: optional group within the HDF5 file to write datasets to (default: root group).
- `overwrite`: if `true`, overwrite the group if it already exists (or the entire file if `group=""`),
    otherwise throw an error if datasets with the same name already exist (default: `false`).
"""
HDF5File(filename::String; group::String="", overwrite::Bool=false) = HDF5File(filename, group, overwrite)
function open_file(f::Function, h5file::HDF5File)
    h5open(h5file.filename, h5file.overwrite && h5file.group == "" ? "w" : "cw") do fid
        if h5file.group != ""
            h5file.overwrite && haskey(fid, h5file.group) && HDF5.delete_object(fid[h5file.group])
            f(create_group(fid, h5file.group))
        else
            f(fid)
        end
    end
end
open_file(f::Function, filename::String) = if endswith(lowercase(filename), r".h(df)?5")
    open_file(f, HDF5File(filename))
else
    throw(ArgumentError("Unsupported file extension: $filename. HDF5 expected."))
end
open_file(f::Function, ::Nothing) = f(nothing)
prepare_dataset(fid::Union{HDF5.File,HDF5.Group}, name::String, type, sz, n, batch) =
    BufferedDataset(create_dataset(fid, name, type, (sz..., n), chunk=(sz..., batch)))
prepare_dataset(::Nothing, ::String, ::Type{T}, sz, n, ::Int) where T =
    BufferedDataset(Array{T}(undef, (sz..., n)))
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
- `batch`: batch size for buffered computations and HDF5 writes (default 128).
- `file`: output options. Can be a string (filename) or an `HDF5File` object. If set to `nothing`
    (default), no file is written and the phases are returned as an array.
- `verbose`: show progress meter (true by default).
- `deviceadapter`: adapter for device-backed arrays (defaults to `Array`). To use GPU arrays,
  pass e.g. `CUDA.CuArray` here (requires CUDA.jl).
"""
function simulate_phases(atm_spec::AtmosphereSpec, plate_size; n::Int, batch::Int=DEFAULT_BATCH, file=nothing,
        verbose=true, deviceadapter=Array)
    batch = min(batch, n)
    phase_buffers = prepare_phasebuffers(atm_spec, plate_size, batch, deviceadapter)
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
- `deviceadapter`: adapter for device-backed arrays (defaults to `Array`). To use GPU arrays,
  pass e.g. `CUDA.CuArray` here (requires CUDA.jl).
"""
function simulate_images(::Type{T}, true_sky::TrueSky, atm_spec::AtmosphereSpec, img_spec::ImagingSpec;
    n::Int, batch::Int=DEFAULT_BATCH, file=nothing, verbose=true, savephases::Bool=true, deviceadapter=Array) where {T}
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
