using LinearAlgebra, FFTW, Distributions, HDF5, ProgressMeter, SparseArrays, Adapt

const DEFAULT_BATCH = 512

"""
    FilterSpec

Representation of a spectral filter used by the imaging pipeline.

---
    FilterSpec(base_wavelength, wavelengths[, intensities])

# Arguments
- `base_wavelength`: central wavelength for the filter.
- `wavelengths`: vector of sampled wavelengths within the filter bandpass.
- `intensities`: vector of relative intensities at each sampled wavelength. If not provided,
  equal weights are assumed.
"""
struct FilterSpec{T}
    base_wavelength::T
    wavelengths::Vector{T}
    intensities::Vector{T}
end
FilterSpec(base_wavelength::T1, wavelengths::AbstractVector{T2},
    intensities::AbstractVector{T3}=ones(Int, length(wavelengths))) where {T1,T2,T3} =
    FilterSpec{promote_type(T1, T2, T3)}(base_wavelength, wavelengths, intensities)
MonochromaticFilterSpec(::Type{T}=Int) where T = FilterSpec{T}(1, [1], [1])

"""
    FilterSpec([T, ]base_wavelength; bandpass, tcenter=1, tedge=1, npts=7)

# Arguments
- `base_wavelength`: central wavelength for the filter (same units as `wavelengths`).

# Keyword Arguments
- `bandpass`: total width of the filter bandpass in wavelength units.
- `tcenter`: relative intensity at the center wavelength (default 1).
- `tedge`: relative intensity at the edges of the bandpass (default 1).
"""
function FilterSpec(::Type{T}, base_wavelength::Real; bandpass, tcenter=1, tedge=1, npts=7) where T<:Real
    wavelengths = range(base_wavelength - bandpass / 2, base_wavelength + bandpass / 2, length=npts)
    intensities = range(-pi/2, pi/2, length=npts) .|> x -> cos(x) * (tcenter - tedge) + tedge
    return FilterSpec{T}(base_wavelength, wavelengths, intensities)
end
FilterSpec(base_wavelength::Real; kw...) = FilterSpec(Float64, base_wavelength; kw...)
Base.convert(::Type{FilterSpec{T}}, bspec::FilterSpec) where T<:Real =
    FilterSpec{T}(bspec.base_wavelength,
        bspec.wavelengths, bspec.intensities)

function prepare_spmat(::Type{T}, img_size, bspec::FilterSpec) where T<:Real
    ctr1, ctr2 = img_size .÷ 2 .+ 1
    is = Int[]
    js = Int[]
    vs = complex(T)[]
    linds = LinearIndices(img_size)
    nx, ny = img_size
    for k in eachindex(bspec.wavelengths)
        r = bspec.wavelengths[k] / bspec.base_wavelength
        inten = bspec.intensities[k] / r^2
        for j in 1:ny
            dy = j - ctr2
            sy = ctr2 + dy * r
            iy = floor(Int, sy)
            (1 <= iy < ny) || continue
            for i in 1:nx
                dx = i - ctr1
                sx = ctr1 + dx * r
                ix = floor(Int, sx)
                if 1 <= ix < nx
                    tx = sx - ix
                    ty = sy - iy
                    push!(is, linds[i, j], linds[i, j], linds[i, j], linds[i, j])
                    push!(js, linds[ix, iy], linds[ix + 1, iy], linds[ix, iy + 1], linds[ix + 1, iy + 1])
                    push!(vs,   (1 - tx) * (1 - ty) * inten,
                                tx       * (1 - ty) * inten,
                                (1 - tx) * ty       * inten,
                                tx       * ty       * inten)
                end
            end
        end
    end
    return sparse(is, js, vs, nx * ny, nx * ny)
end

"""
    PhotonCount(nphotons[, background])

Specifies the photon budget for imaging simulations. Set `nphotons` to `Inf` for
continuous flux (`background` can be omitted in this case).
"""
struct PhotonCount{T<:Real}
    nphotons::T
    background::T
end
PhotonCount(nphotons::T1, background::T2) where {T1<:Real,T2<:Real} =
    return PhotonCount{promote_type(T1, T2)}(nphotons, background)
PhotonCount(nphotons::Real) = if isinf(nphotons)
    PhotonCount(Inf, zero(Float64))
else
    throw(ArgumentError("Must specify background when `nphotons` is finite"))
end
Base.convert(::Type{PhotonCount{T}}, pc::PhotonCount) where T<:Real =
    PhotonCount{T}(pc.nphotons, pc.background)
isfinite_photons(pc::PhotonCount) = isfinite(pc.nphotons)

abstract type TrueSky end

"""
    PointSource()

Simple true-sky brightness model. The photon budget and background are configured
via the `ImagingSpec.photon_count` field. See [`ImagingSpec`](@ref) for details on
configuring photon budget and background.
"""
struct PointSource <: TrueSky end

"""
    DoubleSystem(rel_position, intensity)

Model for a two-component source (binary): primary plus a secondary offset by `rel_position`.
The photon budget and background are configured via the `ImagingSpec.photon_count` field.
See [`ImagingSpec`](@ref) for details.

# Arguments
- `rel_position`: `(dx, dy)` integer tuple specifying the secondary's pixel offset.
- `intensity`: multiplicative intensity of the secondary relative to the primary.
"""
struct DoubleSystem{T} <: TrueSky
    rel_position::NTuple{2,Int}
    intensity::T
    DoubleSystem(position, intensity::Real) =
        new{typeof(intensity)}(Tuple(position), intensity)
end

"""
    TrueSkyImage(true_sky::AbstractMatrix{T})

Wrap a real-valued true-sky image for use with the imaging pipeline. The photon budget and
background are configured via the `ImagingSpec.photon_count` field. See [`ImagingSpec`](@ref)
for details.

# Arguments
- `true_sky`: real image array representing spatial sky brightness.
"""
struct TrueSkyImage{MT<:AbstractMatrix{<:Complex}} <: TrueSky
    true_sky_fft::MT
end
function TrueSkyImage(true_sky::AbstractMatrix{T}) where {T<:Real}
    true_sky_fft = ifft(ifftshift(true_sky))
    true_sky_fft ./= true_sky_fft[1, 1]  # normalize DC component to 1
    return TrueSkyImage{typeof(true_sky_fft)}(true_sky_fft)
end
Adapt.adapt_structure(to, ts::TrueSkyImage) =
    TrueSkyImage(Adapt.adapt_storage(to, ts.true_sky_fft))

"""
    ImagingSpec

Container for the imaging system configuration. It is defined by the telescope `aperture`, the
source brightness via `photon_count`, an optional spectral `filter_spec`, and the output `img_size`.
If `img_size` does not match the aperture’s Nyquist grid, the aperture is zero-padded accordingly.
"""
struct ImagingSpec{T, AT<:AbstractMatrix{T}}
    aperture::AT
    photon_count::PhotonCount{T}
    filter_spec::FilterSpec{T}
    img_size::NTuple{2,Int}
end

"""
    ImagingSpec([T, ]aperture, photon_count[; filter_spec, nyquist_oversample, img_size])
    ImagingSpec([T, ]aperture; nphotons, [background, filter_spec, nyquist_oversample, img_size])

Create an imaging system specification.

# Arguments
- `T`: desired numeric element type, inferred from `aperture` if not provided.
- `aperture`: 2D aperture (pupil) array describing the telescope pupil.
- `photon_count`: `PhotonCount` instance describing the photon budget and background.

# Keyword Arguments
- `filter_spec`: `FilterSpec` describing sampled wavelengths and their relative intensities.
  Defaults to a monochromatic filter.
- `nyquist_oversample`: multiplicative factor applied to the default Nyquist image size (`2 * size(aperture)`).
  Defaults to 1. Ignored if `img_size` is provided.
- `img_size`: explicit output image size `(nx, ny)`. If not provided, computed from aperture size
  and `nyquist_oversample`.
- `nphotons` and `background`: alternative way to specify photon budget and background
  when `photon_count` is not provided.
"""
function ImagingSpec(aperture::AbstractMatrix{T}, photon_count::PhotonCount;
    filter_spec::FilterSpec=MonochromaticFilterSpec(), nyquist_oversample::Real=1,
    img_size::NTuple{2,Int}=round.(Int, size(aperture) .* 2 .* nyquist_oversample)) where T<:Real
    fs = convert(FilterSpec{T}, filter_spec)
    pc = convert(PhotonCount{T}, photon_count)
    return ImagingSpec{T, typeof(aperture)}(aperture, pc, fs, img_size)
end
ImagingSpec(aperture::AbstractMatrix; nphotons, background=1, kw...) =
    ImagingSpec(aperture, PhotonCount(nphotons, background); kw...)
ImagingSpec(::Type{T}, aperture::AbstractMatrix, args...; kw...) where T<:Real =
    ImagingSpec(convert.(T, aperture), args...; kw...)

Adapt.adapt_structure(to, imgspec::ImagingSpec) =
    ImagingSpec(Adapt.adapt_storage(to, imgspec.aperture), imgspec.photon_count, imgspec.filter_spec, imgspec.img_size)
plate_size(img_spec::ImagingSpec) = size(img_spec.aperture)
image_size(img_spec::ImagingSpec) = img_spec.img_size
psf_norm(img_spec::ImagingSpec) = sum(abs2, img_spec.aperture) * prod(img_spec.img_size) *
        sum(img_spec.filter_spec.intensities)

struct OpticalBuffers{AT, BT, MT, PT, PCT<:PhotonCount}
    aperture::AT
    radial_blur::BT
    aperture_buffer::MT
    focal_buffer::MT
    fftplan::PT
    photon_count::PCT
end
plate_size(bufs::OpticalBuffers) = size(bufs.aperture)
image_size(bufs::OpticalBuffers) = size(bufs.aperture_buffer)[1:2]
batch_length(bufs::OpticalBuffers) = size(bufs.aperture_buffer, 3)

function OpticalBuffers(imgspec::ImagingSpec, blur, batch::Int)
    complex_type = complex(eltype(imgspec.aperture))
    buf1 = similar(imgspec.aperture, complex_type, imgspec.img_size..., batch)
    buf2 = similar(imgspec.aperture, complex_type, imgspec.img_size..., batch)
    return OpticalBuffers(imgspec.aperture, blur, buf1, buf2, plan_fft(buf1, (1, 2)), imgspec.photon_count)
end
function prepare_blur(imgspec::ImagingSpec)
    if length(imgspec.filter_spec.wavelengths) > 1
        return prepare_spmat(eltype(imgspec.aperture), imgspec.img_size, imgspec.filter_spec)
    else
        return nothing
    end
end
function OpticalBuffers(imgspec::ImagingSpec, batch::Int)
    blur = prepare_blur(imgspec)
    return OpticalBuffers(imgspec, blur, batch)
end

function write_phases!(aperture_buffer, phases, aperture)
    M, N = size(phases)
    Cx, Cy = size(aperture_buffer) .÷ 2
    fill!(aperture_buffer, 0)
    aperture_buffer[Cx - M ÷ 2 + 1:Cx - M ÷ 2 + M, Cy - N ÷ 2 + 1:Cy - N ÷ 2 + N, :] .=
        aperture .* cis.(phases)
end

function radial_blur!(out, src, smat::AbstractMatrix)
    mul!(reshape(out, :, size(src, 3)), smat, reshape(src, :, size(src, 3)))
    return out
end
radial_blur!(out, src, ::Nothing) = copyto!(out, src)

function psf!(bufs::OpticalBuffers, phases)
    write_phases!(bufs.focal_buffer, phases, bufs.aperture)
    mul!(bufs.aperture_buffer, bufs.fftplan, bufs.focal_buffer)
    fftshift!(bufs.focal_buffer, bufs.aperture_buffer, (1, 2))
    bufs.aperture_buffer .= abs2.(bufs.focal_buffer)
    radial_blur!(bufs.focal_buffer, bufs.aperture_buffer, bufs.radial_blur)
end

function readout!(dst::AbstractArray, img::AbstractArray, pc::PhotonCount, psf_norm)
    @assert maximum(abs ∘ imag, img) / maximum(abs ∘ real, img) < 1e-5
    @assert all(x -> real(x) ≥ 0, img)
    if isfinite_photons(pc)
        @. dst = rand(Poisson(real(img) / psf_norm * pc.nphotons + pc.background))
    else
        @. dst = img / psf_norm
    end
end
function apply_truesky!(dst, opt_buf::OpticalBuffers, ts::TrueSkyImage, psf_norm)
    mul!(opt_buf.aperture_buffer, opt_buf.fftplan, opt_buf.focal_buffer)
    opt_buf.aperture_buffer .*= ts.true_sky_fft
    ldiv!(opt_buf.focal_buffer, opt_buf.fftplan, opt_buf.aperture_buffer)
    readout!(dst, opt_buf.focal_buffer, opt_buf.photon_count, psf_norm)
end
function apply_truesky!(dst, opt_buf::OpticalBuffers, ds::DoubleSystem, psf_norm)
    img = opt_buf.focal_buffer
    @assert all(abs.(ds.rel_position) .< size(img)[1:2] .÷ 2)
    @assert size(dst)[1:2] == image_size(opt_buf)
    @assert size(dst, 3) == batch_length(opt_buf)
    o1, o2 = ds.rel_position
    s1_dest, s1_src = o1 > 0 ? (o1 + 1:size(img, 1), 1:size(img, 1) - o1) : (1:size(img, 1) + o1, -o1 + 1:size(img, 1))
    s2_dest, s2_src = o2 > 0 ? (o2 + 1:size(img, 2), 1:size(img, 2) - o2) : (1:size(img, 2) + o2, -o2 + 1:size(img, 2))
    @views @. img[s1_dest, s2_dest, :] += img[s1_src, s2_src, :] * ds.intensity
    readout!(dst, img, opt_buf.photon_count, psf_norm * (1 + ds.intensity))
end
function apply_truesky!(dst, opt_buf::OpticalBuffers, ::PointSource, psf_norm)
    readout!(dst, opt_buf.focal_buffer, opt_buf.photon_count, psf_norm)
end


"""
    CircularAperture([T, ]sz, radius[; aa_dist=1])

Create a circular (optionally anti-aliased) aperture array of shape `sz`. Returns a 2D
numeric array suitable for use as an aperture in `ImagingSpec`.

# Arguments
- `T`: desired numeric element type, `Float64` by default.
- `sz`: aperture size `(nx, ny)`.
- `radius`: radius of the circular aperture in pixels. Defaults to the largest that fits.
- `aa_dist`: anti-aliasing transition width in pixels at the aperture edge.
"""
function CircularAperture(::Type{T}, sz::NTuple{2}, radius=minimum((sz .- 1) .÷ 2); aa_dist=1) where T<:Real
    aperture = zeros(T, sz)
    X, Y = sz .÷ 2 .+ 1
    for I in eachindex(IndexCartesian(), aperture)
        x, y = I[1] - X, I[2] - Y
        r = sqrt(x^2 + y^2)
        if r < radius - aa_dist / 2
            aperture[I] = 1
        elseif r > radius + aa_dist / 2
            aperture[I] = 0
        else
            aperture[I] = 0.5 - (r - radius) / aa_dist
        end
    end
    return aperture
end
CircularAperture(sz::NTuple{2}, radius=minimum((sz .- 1) .÷ 2); kw...) =
    CircularAperture(Float64, sz, radius; kw...)

struct ImgBufSerial{BT<:OpticalBuffers, FT<:Real, AT<:AbstractArray}
    opt_buf::BT
    psf_norm::FT
    img_tensor::AT
end
image_size(img_buf::ImgBufSerial) = image_size(img_buf.opt_buf)
image_type(img_buf::ImgBufSerial) = eltype(img_buf.img_tensor)
function prepare_imgbuffers(::Type{T}, img_spec::ImagingSpec, batch::Int, deviceadapter) where T
    img_spec_adapt = adapt(deviceadapter, img_spec)
    rblur_adapt = adapt(deviceadapter, prepare_blur(img_spec_adapt))
    ImgBufSerial(
        OpticalBuffers(img_spec_adapt, rblur_adapt, batch),
        psf_norm(img_spec),
        adapt(deviceadapter, zeros(T, img_spec.img_size..., batch)))
end
@inline function compute_images!(img_array, img_buf::ImgBufSerial, phases, true_sky)
    psf!(img_buf.opt_buf, phases)
    apply_truesky!(img_buf.img_tensor, img_buf.opt_buf, true_sky, img_buf.psf_norm)
    copyto!(img_array, img_buf.img_tensor)
end

struct ImgBufParallel{T<:Real,BT<:OpticalBuffers,FT<:Real}
    opt_bufs::Vector{BT}
    psf_norm::FT
end
ImgBufParallel(bufs::Vector{BT}, psf_norm::FT, ::Type{T}) where {T,BT<:OpticalBuffers,FT} =
    ImgBufParallel{T,BT,FT}(bufs, psf_norm)
image_size(img_buf::ImgBufParallel) = image_size(img_buf.opt_bufs[1])
image_type(::ImgBufParallel{T}) where T = T
function prepare_imgbuffers(::Type{T}, img_spec::ImagingSpec, ::Int, ::Type{<:Array}) where T
    imgbuf1 = OpticalBuffers(img_spec, 1)
    opt_buf_vector = Array{typeof(imgbuf1)}(undef, Threads.nthreads())
    opt_buf_vector[1] = imgbuf1
    Threads.@threads for i in 2:Threads.nthreads()
        opt_buf_vector[i] = OpticalBuffers(img_spec, 1)
    end
    return ImgBufParallel(opt_buf_vector, psf_norm(img_spec), T)
end
@inline function compute_images!(img_array, img_buf::ImgBufParallel, phases, true_sky)
    Threads.@threads for i in eachindex(img_buf.opt_bufs)
        op_buf = img_buf.opt_bufs[i]
        for j in i:length(img_buf.opt_bufs):size(phases, 3)
            psf!(op_buf, view(phases, :, :, j))
            apply_truesky!(view(img_array, :, :, j), op_buf, true_sky, img_buf.psf_norm)
        end
    end
end

function simulation_run!!(img_dataset, phs_dataset, phsbuffers, imgbuffers, truesky_adapt; n, verbose=true)
    phs_size = plate_size(phsbuffers)
    batch = batch_length(phsbuffers)
    phase_buf_h5 = zeros(phase_type(phsbuffers), phs_size..., batch)
    if imgbuffers !== nothing
        img_size = image_size(imgbuffers)
        image_buf_h5 = zeros(image_type(imgbuffers), img_size..., batch)
    end
    p = Progress(n, desc="Simulating images", enabled=verbose, dt=1)
    for j in 1:cld(n, batch)
        phases = samplephases!(phsbuffers)
        if imgbuffers !== nothing
            compute_images!(image_buf_h5, imgbuffers, phases, truesky_adapt)
            if img_dataset !== nothing
                HDF5.write_chunk(img_dataset, j - 1, image_buf_h5)
            end
        end
        if phs_dataset !== nothing
            if phases isa Array
                HDF5.write_chunk(phs_dataset, j - 1, phases)
            else
                copy!(phase_buf_h5, phases)
                HDF5.write_chunk(phs_dataset, j - 1, phase_buf_h5)
            end
        end
        next!(p, step=min(batch, n - (j - 1) * batch))
    end
    finish!(p)
end

"""
    simulate_images([T, ]img_spec::ImagingSpec, atm_spec::AtmosphereSpec[, truesky::TrueSky]; \
        n, [batch, filename, verbose, savephases, deviceadapter])

Simulate `n` images using the provided imaging and atmosphere specifications and write
the results to an HDF5 file.

# Arguments
- `T`: output image numeric type; if not provided, defaults to `Int` for
  finite-photon simulations (determined by `img_spec.photon_count.nphotons`) and `Float64` for infinite-photon models.
- `img_spec`: an `ImagingSpec` describing the aperture, image size, photon budget and filter.
- `atm_spec`: an `AtmosphereSpec` used to produce phase screens.
- `truesky`: a `TrueSky` model (e.g. `PointSource`, `DoubleSystem`, `TrueSkyImage`).

# Keyword Arguments
- `n`: number of images to simulate.
- `batch`: batch size for buffered computations and HDF5 writes (default 512).
- `filename`: output HDF5 filename (default "simulation.h5").
- `verbose`: show progress meter (true by default).
- `savephases`: when true, the sampled phase screens are saved in the HDF5 in dataset with
  key `"phases"`, and the pupil function is saved under key `"aperture"` (true by default).
- `deviceadapter`: adapter for device-backed arrays (defaults to `Array`). To use GPU arrays,
  pass e.g. `CUDA.CuArray` here (requires CUDA.jl).
"""
function simulate_images(::Type{T}, img_spec::ImagingSpec{FT}, atm_spec::AtmosphereSpec,
    truesky::TrueSky=PointSource(); n::Int, batch::Int=DEFAULT_BATCH, filename="simulation.h5", verbose=true,
    savephases::Bool=true, deviceadapter=Array) where {T,FT}
    if !isfinite_photons(img_spec.photon_count) && T <: Integer
        throw(ArgumentError("Integer image eltype not compatible with infinite-photon imaging spec."))
    end
    if plate_size(img_spec) != plate_size(atm_spec)
        throw(ArgumentError("Telescope plate size $(plate_size(img_spec)) does not match" *
            "AtmosphereSpec plate size $(plate_size(atm_spec))."))
    end
    if truesky isa TrueSkyImage && size(truesky.true_sky_fft) != image_size(img_spec)
        throw(ArgumentError("TrueSkyImage size $(size(truesky.true_sky_fft)) does not match " *
            "image size $(image_size(img_spec))."))
    end

    batch = min(batch, n)
    img_size = image_size(img_spec)
    phs_size = plate_size(atm_spec)
    truesky_adapt = adapt(deviceadapter, truesky)
    phsbuffers = prepare_phasebuffers(atm_spec, batch, deviceadapter)
    imgbuffers = prepare_imgbuffers(T, img_spec, batch, deviceadapter)

    h5open(filename, "w") do fid
        fid["aperture"] = img_spec.aperture
        img_dataset = create_dataset(fid, "images", T, (img_size..., n), chunk=(img_size..., batch))
        if savephases
            phs_dataset = create_dataset(fid, "phases", phase_type(phsbuffers), (phs_size..., n), chunk=(phs_size..., batch))
        else
            phs_dataset = nothing
        end
        simulation_run!!(img_dataset, phs_dataset, phsbuffers, imgbuffers, truesky_adapt;
            n=n, verbose=verbose)
    end
end
simulate_images(img_spec::ImagingSpec, phase_sampler::AtmosphereSpec, true_sky::TrueSky=PointSource(); kwargs...) =
    simulate_images(isfinite_photons(img_spec.photon_count) ? Int : Float64, img_spec, phase_sampler, true_sky; kwargs...)
