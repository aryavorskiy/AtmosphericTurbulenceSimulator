using LinearAlgebra, FFTW, Distributions, HDF5, ProgressMeter, Adapt, ChunkSplitters

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
MonoFilterSpec(::Type{T}=Int) where T = FilterSpec{T}(1, [1], [1])
nwavel(fs::FilterSpec) = length(fs.wavelengths)

struct BilinearInterpolator{IT,VT}
    ix::IT
    iy::IT
    ixp1::IT
    iyp1::IT
    tx::VT
    ty::VT
    can_ff::Bool
end
function BilinearScale(to::AbstractArray, scale::Real)
    nx, ny = size(to)
    sx = range((1 - nx ÷ 2) / scale + nx ÷ 2 + 1, step=1/scale, length=nx)
    sy = range((1 - ny ÷ 2) / scale + ny ÷ 2 + 1, step=1/scale, length=ny)
    ix = copy!(similar(to, Int, nx), clamp.(floor.(Int, sx), 1, nx))
    iy = copy!(similar(to, Int, ny), clamp.(floor.(Int, sy), 1, ny))
    ixp1 = copy!(similar(to, Int, nx), clamp.(ceil.(Int, sx), 1, nx))
    iyp1 = copy!(similar(to, Int, ny), clamp.(ceil.(Int, sy), 1, ny))
    tx = copy!(similar(to, nx), (sx .- ix))
    ty = copy!(similar(to, ny), (sy .- iy))
    return BilinearInterpolator(ix, iy, ixp1, iyp1, tx, ty, scale≈1)
end
function BilinearShift(to::AbstractArray, offset::NTuple{2})
    nx, ny = size(to)
    ix = range(floor(Int, 1 + offset[1]), length=nx)
    iy = range(floor(Int, 1 + offset[2]), length=ny)
    ixp1 = range(ceil(Int, 1 + offset[1]), length=nx)
    iyp1 = range(ceil(Int, 1 + offset[2]), length=ny)
    tx = convert(eltype(to), offset[1] - floor(offset[1]))
    ty = convert(eltype(to), offset[2] - floor(offset[2]))
    return BilinearInterpolator(ix, iy, ixp1, iyp1, tx, ty, all(isinteger, offset))
end
function interpolate_mapmuladd!(to::AbstractArray, from::AbstractArray, interp::BilinearInterpolator,
        factor, g=identity, h=identity)
    tx = interp.tx
    ty = transpose(interp.ty)
    if interp.can_ff
        @views @inbounds @. to += factor * g(h(from[interp.ix, interp.iy, :]))
    else
        @views @inbounds @. to += factor * g(
            (1 - tx) * (1 - ty) * h(from[interp.ix, interp.iy, :]) +
            tx * (1 - ty) * h(from[interp.ixp1, interp.iy, :]) +
            (1 - tx) * ty * h(from[interp.ix, interp.iyp1, :]) +
            tx * ty * h(from[interp.ixp1, interp.iyp1, :]))
    end
    return to
end

"""
    FilterSpec(base_wavelength; bandwidth[, tcenter=1, tedge=1, npts=7])

# Arguments
- `base_wavelength`: central wavelength for the filter (same units as `wavelengths`).

# Keyword Arguments
- `bandwidth`: total width of the filter bandpass in wavelength units.
- `tcenter`: relative intensity at the center wavelength (default 1).
- `tedge`: relative intensity at the edges of the bandpass (default 1).
"""
function FilterSpec(base_wavelength::Real; bandwidth, tcenter=1, tedge=1, npts=7)
    wavelengths = range(base_wavelength - bandwidth / 2, base_wavelength + bandwidth / 2, length=npts)
    intensities = range(-pi/2, pi/2, length=npts) .|> x -> cos(x) * (tcenter - tedge) + tedge
    return FilterSpec(base_wavelength, wavelengths, intensities)
end
Base.convert(::Type{FilterSpec{T}}, bspec::FilterSpec) where T<:Real =
    FilterSpec{T}(bspec.base_wavelength, bspec.wavelengths, bspec.intensities)

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
PhotonCount(nphotons::Real) = isinf(nphotons) ? PhotonCount(Inf, zero(Float64)) :
    throw(ArgumentError("Must specify background when `nphotons` is finite"))
Base.convert(::Type{PhotonCount{T}}, pc::PhotonCount) where T<:Real =
    PhotonCount{T}(pc.nphotons, pc.background)
isfinite_photons(pc::PhotonCount) = isfinite(pc.nphotons)

struct Exposure
    exptime::Float64
    nsteps::Int
    round_offsets::Bool
end
"""
    Exposure(exptime[, nsteps][; round_offsets])

A struct that encapsulates the exposure time and related parameters for long-exposure simulations.
The exposure is simulated by averaging `nsteps` short exposures with appropriate offsets of the wavefront.

# Arguments
- `exptime`: total exposure time. Shares the same time units as the wind velocity in the atmosphere specification.
- `nsteps`: number of steps to simulate for long exposures (default 5).
- `round_offsets`: whether to round the exposure offsets to integers (default false). When true,
  the phase screens are sampled at integer pixel offsets, which can reduce interpolation artifacts.
"""
function Exposure(exptime, nsteps=5; round_offsets=false)
    nsteps == 1 && !iszero(exptime) &&
        @warn "Ignoring non-zero exposure time for single-step exposure."
    if iszero(exptime)
        nsteps = 1
    end
    Exposure(exptime, nsteps, round_offsets)
end

abstract type TrueSky end

"""
    PointSource()

Simple true-sky brightness model. Represents a non-resolved point source.
"""
struct PointSource <: TrueSky end

"""
    DoubleSystem(rel_position, intensity)

Model for a two-component source (binary): primary plus a secondary offset by `rel_position`.

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

Wrap a real-valued true-sky image for use with the imaging pipeline. The pixel scale matches
that of the `ImagingSpec`.

# Arguments
- `true_sky`: real image array representing spatial sky brightness.
"""
struct TrueSkyImage{MT<:AbstractMatrix{<:Complex}} <: TrueSky
    true_sky_fft::MT
end
function TrueSkyImage(true_sky::AbstractMatrix{T}) where {T<:Real}
    true_sky_fft = fft(ifftshift(true_sky))
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
    exposure_spec::Exposure
    img_size::NTuple{2,Int}
end

"""
    ImagingSpec([T, ]aperture, photon_count[; filter, nyquist_oversample, img_size])
    ImagingSpec([T, ]aperture; nphotons, [background, filter, nyquist_oversample, img_size])

Create an imaging system specification.

# Arguments
- `T`: desired number type, inferred from `aperture` if not provided.
- `aperture`: 2D aperture (pupil) array describing the telescope pupil.
- `photon_count`: `PhotonCount` instance describing the photon budget and background.

# Keyword Arguments
- `filter`: `FilterSpec` describing sampled wavelengths and their relative intensities.
  Defaults to a monochromatic filter.
- `exposure`: a number or an [`Exposure`](@ref) instance describing the exposure time and number of
  steps for long exposures. Defaults to zero exposure time (i.e. short exposure).
- `nyquist_oversample`: multiplicative factor applied to the default Nyquist image size (`2 * size(aperture)`).
  Defaults to 1. Ignored if `img_size` is provided.
- `img_size`: explicit output image size `(nx, ny)`. If not provided, computed from aperture size
  and `nyquist_oversample`.
- `nphotons` and `background`: alternative way to specify photon budget and background
  when `photon_count` is not provided.
"""
function ImagingSpec(aperture::AbstractMatrix{T}, photon_count::PhotonCount;
    filter::FilterSpec=MonoFilterSpec(), nyquist_oversample::Real=1,
    exposure::Union{Exposure,Number}=0,
    img_size::NTuple{2,Int}=round.(Int, size(aperture) .* 2 .* nyquist_oversample)) where T<:Real
    fs = convert(FilterSpec{T}, filter)
    pc = convert(PhotonCount{T}, photon_count)
    ex = exposure isa Number ? Exposure(exposure) : exposure
    return ImagingSpec{T, typeof(aperture)}(aperture, pc, fs, ex, img_size)
end
function ImagingSpec(aperture::AbstractMatrix; nphotons, background=nothing, kw...)
    Base.depwarn("`ImagingSpec(ap; nphotons=..., background=...)` is deprecated and will be \
        removed in v0.5. Use `ImagingSpec(ap, PhotonCount(nphotons, background))` instead.", :ImagingSpec)
    if background === nothing
        pc = PhotonCount(nphotons)
    else
        pc = PhotonCount(nphotons, background)
    end
    ImagingSpec(aperture, pc; kw...)
end
ImagingSpec(::Type{T}, aperture::AbstractMatrix, args...; kw...) where T<:Real =
    ImagingSpec(convert.(T, aperture), args...; kw...)

Adapt.adapt_structure(to, img_spec::ImagingSpec) =
    ImagingSpec(Adapt.adapt_storage(to, img_spec.aperture), img_spec.photon_count,
    img_spec.filter_spec, img_spec.exposure_spec, img_spec.img_size)
plate_size(img_spec::ImagingSpec) = size(img_spec.aperture)
image_size(img_spec::ImagingSpec) = img_spec.img_size
psf_norm(img_spec::ImagingSpec) = sum(abs2, img_spec.aperture) * prod(img_spec.img_size) *
        sum(img_spec.filter_spec.intensities)

struct OpticalBuffers{MT, MT2, MT3, PT, IT}
    aperture_buffer::MT
    focal_buffer::MT
    psf_buffer::MT2
    read_buffer::MT3
    fftplan::PT
    interpolators::Vector{IT}
end
image_size(bufs::OpticalBuffers) = size(bufs.aperture_buffer)[1:2]
image_type(bufs::OpticalBuffers) = eltype(bufs.read_buffer)
batch_length(bufs::OpticalBuffers) = size(bufs.aperture_buffer, 3)

function OpticalBuffers(::Type{T}, img_spec::ImagingSpec{NT}, batch::Int) where {T, NT}
    buf1 = similar(img_spec.aperture, complex(NT), img_spec.img_size..., batch, nwavel(img_spec.filter_spec))
    buf2 = similar(buf1)
    psf_buffer = similar(buf1, NT, img_spec.img_size..., batch)
    read_buffer = similar(buf1, T, img_spec.img_size..., batch)
    if nwavel(img_spec.filter_spec) == 1
        interpolators = [BilinearShift(psf_buffer, (0, 0))]
    else
        interpolators = [BilinearScale(psf_buffer, img_spec.filter_spec.wavelengths[w] /
        img_spec.filter_spec.base_wavelength) for w in 1:nwavel(img_spec.filter_spec)]
    end
    return OpticalBuffers(buf1, buf2, psf_buffer, read_buffer, plan_fft(buf1, (1, 2)), interpolators)
end
function write_phases!(aperture_buffer, phases, aperture, filter_spec, offset)
    M, N = size(aperture)
    Cx, Cy = size(aperture_buffer) .÷ 2
    fill!(aperture_buffer, 0)
    for w in 1:nwavel(filter_spec)
        scale = filter_spec.wavelengths[w] / filter_spec.base_wavelength
        ap_slice = @view aperture_buffer[Cx - M ÷ 2 + 1:Cx - M ÷ 2 + M, Cy - N ÷ 2 + 1:Cy - N ÷ 2 + N, :, w]
        interpolate_mapmuladd!(ap_slice, phases, offset, aperture, cis, Base.Fix2(*, scale))
    end
end
write_phases!(bufs::OpticalBuffers, phases, img_spec::ImagingSpec, offset) =
    write_phases!(bufs.focal_buffer, phases, img_spec.aperture, img_spec.filter_spec, offset)

function phases_to_psf!(bufs::OpticalBuffers, img_spec::ImagingSpec)
    mul!(bufs.aperture_buffer, bufs.fftplan, bufs.focal_buffer)
    fftshift!(bufs.focal_buffer, bufs.aperture_buffer, (1, 2))
    for w in 1:nwavel(img_spec.filter_spec)
        mono_psf_field_block = view(bufs.focal_buffer, :, :, :, w)
        scale = img_spec.filter_spec.wavelengths[w] / img_spec.filter_spec.base_wavelength
        factor = img_spec.filter_spec.intensities[w] / scale^2
        interpolate_mapmuladd!(bufs.psf_buffer, mono_psf_field_block, bufs.interpolators[w], factor, identity, abs2)
    end
end

function readout!(dst::AbstractArray, img::AbstractArray, pc::PhotonCount, psf_norm)
    if isfinite_photons(pc)
        @. dst = rand(Poisson(real(img) / psf_norm * pc.nphotons + pc.background))
    else
        @. dst = img / psf_norm
    end
end
readout!(opt_buffer::OpticalBuffers, pc::PhotonCount, psf_norm) =
    readout!(opt_buffer.read_buffer, opt_buffer.psf_buffer, pc, psf_norm)
function apply_truesky!(opt_buffer::OpticalBuffers, ts::TrueSkyImage)
    #TODO Review for possible optimizations
    copyto!(opt_buffer.focal_buffer, opt_buffer.psf_buffer)
    mul!(opt_buffer.aperture_buffer, opt_buffer.fftplan, opt_buffer.focal_buffer)
    opt_buffer.aperture_buffer .*= ts.true_sky_fft
    ldiv!(opt_buffer.focal_buffer, opt_buffer.fftplan, opt_buffer.aperture_buffer)
    opt_buffer.psf_buffer .= real.(@view opt_buffer.focal_buffer[:, :, :, 1])
end
function apply_truesky!(opt_buffer::OpticalBuffers, ds::DoubleSystem)
    img = opt_buffer.psf_buffer
    @assert all(abs.(ds.rel_position) .< image_size(opt_buffer) .÷ 2)
    o1, o2 = ds.rel_position
    s1_dest, s1_src = o1 > 0 ? (o1 + 1:size(img, 1), 1:size(img, 1) - o1) : (1:size(img, 1) + o1, -o1 + 1:size(img, 1))
    s2_dest, s2_src = o2 > 0 ? (o2 + 1:size(img, 2), 1:size(img, 2) - o2) : (1:size(img, 2) + o2, -o2 + 1:size(img, 2))
    @views @. img[s1_dest, s2_dest, :] += img[s1_src, s2_src, :] * ds.intensity
    img ./= (1 + ds.intensity)
end
apply_truesky!(::OpticalBuffers, ::PointSource) = nothing


"""
    CircularAperture([T, ]sz, radius[; aa_dist=1])

Create a circular (optionally anti-aliased) aperture array of shape `sz`. Returns a 2D
numeric array suitable for use as an aperture in `ImagingSpec`.

# Arguments
- `T`: desired number type, `Float64` by default.
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

function padded_plate_size(atm_spec::AtmosphereSpec, img_spec::ImagingSpec)
    max_offset = atm_spec.wind_velocity .* img_spec.exposure_spec.exptime
    return plate_size(img_spec) .+ ceil.(Int, abs.(max_offset))
end
function long_exp_offsets(atm_spec::AtmosphereSpec, img_spec::ImagingSpec)
    n = img_spec.exposure_spec.nsteps
    if n == 1 || iszero(img_spec.exposure_spec.exptime) || all(iszero, atm_spec.wind_velocity)
        offset_list = [atm_spec.wind_velocity .* img_spec.exposure_spec.exptime .* 0]
    else
        offset_list = [atm_spec.wind_velocity .* (img_spec.exposure_spec.exptime * j / (n - 1)) for j in 0:n-1]
    end
    if img_spec.exposure_spec.round_offsets
        offset_list = [round.(offset) for offset in offset_list]
    end
    mins = minimum(first, offset_list), minimum(last, offset_list)
    return [BilinearShift(img_spec.aperture, offset .- mins) for offset in offset_list]
end
function _compute_images!(readout_to, opt_buffer::OpticalBuffers, spec::ImagingSpec, phases, true_sky, offsets, psf_norm)
    fill!(opt_buffer.psf_buffer, 0)
    for offset in offsets
        write_phases!(opt_buffer, phases, spec, offset)
        phases_to_psf!(opt_buffer, spec)
    end
    apply_truesky!(opt_buffer, true_sky)
    readout!(readout_to, opt_buffer.psf_buffer, spec.photon_count, psf_norm * length(offsets))
end

"""
    MultiThreaded([AT=Array, ]nworkers=Threads.nthreads())

Device adapter that enables multi-threaded phase generation and imaging on CPU. `AT` is the
underlying array type; `nworkers` controls how many threads are used (defaults to
`Threads.nthreads()`).
"""
struct MultiThreaded{AT}
    adapter::AT
    nworkers::Int
end
MultiThreaded(::Type{AT}, nworkers::Int) where {AT} = MultiThreaded(Val(AT), nworkers)
MultiThreaded(adapter) = MultiThreaded(adapter, Threads.nthreads())
MultiThreaded(nworkers::Int=Threads.nthreads()) = MultiThreaded(identity, nworkers)
Adapt.adapt_storage(am::MultiThreaded{AT}, x) where {AT} = Adapt.adapt_storage(am.adapter, x)
struct ImgBufParallel{BT<:OpticalBuffers,ST<:ImagingSpec,FT<:Real,OT,AT,CT}
    opt_bufs::Vector{BT}
    chunk_ranges::Vector{CT}
    spec::ST
    psf_norm::FT
    offsets::Vector{OT}
    img_array::AT
end
image_size(img_buf::ImgBufParallel) = image_size(img_buf.opt_bufs[1])
image_type(img_buf::ImgBufParallel) = eltype(img_buf.img_array)
function prepare_buffers(::Type{T}, atm_spec, img_spec::ImagingSpec, batch::Int, adapter::MultiThreaded) where T
    nbufs = min(adapter.nworkers, batch)
    chunk_ranges = collect(chunks(1:batch; n=nbufs))
    img_spec_adapt = adapt(adapter, img_spec)
    opt_buffer1 = OpticalBuffers(T, img_spec_adapt, length(chunk_ranges[1]))
    img_array = similar(opt_buffer1.read_buffer, image_size(img_spec)..., batch)
    opt_bufs = Array{typeof(opt_buffer1)}(undef, nbufs)
    opt_bufs[1] = opt_buffer1
    Threads.@threads for i in 2:nbufs
        opt_bufs[i] = OpticalBuffers(T, img_spec_adapt, length(chunk_ranges[i]))
    end
    return prepare_phasebuffers(atm_spec, padded_plate_size(atm_spec, img_spec), batch, adapter),
        ImgBufParallel(opt_bufs, chunk_ranges, img_spec_adapt, psf_norm(img_spec),
            long_exp_offsets(atm_spec, img_spec), img_array)
end
prepare_buffers(type, atm_spec, img_spec, batch, A) =
    prepare_buffers(type, atm_spec, img_spec, batch, MultiThreaded(A))
prepare_buffers(type, atm_spec, img_spec, batch, ::Type{AT}) where {AT<:Array} =
    prepare_buffers(type, atm_spec, img_spec, batch, MultiThreaded(AT, 1))
function compute_images!(img_buf::ImgBufParallel, phases, true_sky)
    if length(img_buf.chunk_ranges) == 1
        _compute_images!(img_buf.img_array, only(img_buf.opt_bufs), img_buf.spec, phases,
            true_sky, img_buf.offsets, img_buf.psf_norm)
    else
        Threads.@threads for i in eachindex(img_buf.opt_bufs)
            chunk_range = img_buf.chunk_ranges[i]
            _compute_images!(view(img_buf.img_array, :, :, chunk_range), img_buf.opt_bufs[i], img_buf.spec,
                view(phases, :, :, chunk_range), true_sky, img_buf.offsets, img_buf.psf_norm)
        end
    end
    return img_buf.img_array
end
