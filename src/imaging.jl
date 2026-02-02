using LinearAlgebra, FFTW, Distributions, HDF5, ProgressMeter, Adapt

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

struct Interpolator{IT,VT}
    ix::IT
    iy::IT
    ixp1::IT
    iyp1::IT
    tx::VT
    ty::VT
end
function Interpolator(array::AbstractArray, scale::Real)
    nx, ny = size(array)
    sx = range((1 - nx ÷ 2) / scale + nx ÷ 2 + 1, step=1/scale, length=nx)
    sy = range((1 - ny ÷ 2) / scale + ny ÷ 2 + 1, step=1/scale, length=ny)
    ix = copy!(similar(array, Int, nx), clamp.(floor.(Int, sx), 1, nx))
    iy = copy!(similar(array, Int, ny), clamp.(floor.(Int, sy), 1, ny))
    ixp1 = copy!(similar(array, Int, nx), clamp.(ceil.(Int, sx), 1, nx))
    iyp1 = copy!(similar(array, Int, ny), clamp.(ceil.(Int, sy), 1, ny))
    tx = copy!(similar(array, nx), (sx .- ix))
    ty = copy!(similar(array, ny), (sy .- iy))
    return Interpolator(ix, iy, ixp1, iyp1, tx, ty)
end
function interpolate_add!(to::AbstractArray, from::AbstractArray, interp::Interpolator, f)
    size(from)[1:2] == length.((interp.ix, interp.iy)) == length.((interp.ixp1, interp.iyp1)) ||
        throw(DimensionMismatch("incmpatible Interpolator dimensions"))
    size(to) == size(from) ||
        throw(DimensionMismatch("src and dest must have matching sizes"))
    tx = interp.tx
    ty = reshape(interp.ty, (1, :))
    @views @inbounds @. to += f * (
        (1 - tx) * (1 - ty) * from[interp.ix, interp.iy, :] +
        tx * (1 - ty) * from[interp.ixp1, interp.iy, :] +
        (1 - tx) * ty * from[interp.ix, interp.iyp1, :] +
        tx * ty * from[interp.ixp1, interp.iyp1, :])
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
- `T`: desired number type, inferred from `aperture` if not provided.
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
    filter_spec::FilterSpec=MonoFilterSpec(), nyquist_oversample::Real=1,
    img_size::NTuple{2,Int}=round.(Int, size(aperture) .* 2 .* nyquist_oversample)) where T<:Real
    fs = convert(FilterSpec{T}, filter_spec)
    pc = convert(PhotonCount{T}, photon_count)
    return ImagingSpec{T, typeof(aperture)}(aperture, pc, fs, img_size)
end
function ImagingSpec(aperture::AbstractMatrix; nphotons, background=nothing, kw...)
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
    ImagingSpec(Adapt.adapt_storage(to, img_spec.aperture), img_spec.photon_count, img_spec.filter_spec, img_spec.img_size)
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
    psf_buf = similar(buf1, NT, img_spec.img_size..., batch)
    read_buf = similar(buf1, T, img_spec.img_size..., batch)
    interpolators = [Interpolator(psf_buf, img_spec.filter_spec.wavelengths[w] /
        img_spec.filter_spec.base_wavelength) for w in 1:nwavel(img_spec.filter_spec)]
    return OpticalBuffers(buf1, buf2, psf_buf, read_buf, plan_fft(buf1, (1, 2)), interpolators)
end

function write_phases!(aperture_buffer, phases, aperture, filter_spec)
    M, N = size(phases)
    Cx, Cy = size(aperture_buffer) .÷ 2
    fill!(aperture_buffer, 0)
    for w in 1:nwavel(filter_spec)
        scale = filter_spec.wavelengths[w] / filter_spec.base_wavelength
        @. aperture_buffer[Cx - M ÷ 2 + 1:Cx - M ÷ 2 + M, Cy - N ÷ 2 + 1:Cy - N ÷ 2 + N, :, w] =
            aperture * cis(scale * phases)
    end
end

function psf!(bufs::OpticalBuffers, img_spec::ImagingSpec, phases)
    write_phases!(bufs.focal_buffer, phases, img_spec.aperture, img_spec.filter_spec)
    mul!(bufs.aperture_buffer, bufs.fftplan, bufs.focal_buffer)
    fftshift!(bufs.focal_buffer, bufs.aperture_buffer, (1, 2))
    bufs.aperture_buffer .= abs2.(bufs.focal_buffer)
    fill!(bufs.psf_buffer, 0)
    for w in 1:nwavel(img_spec.filter_spec)
        mono_psf_block = view(bufs.aperture_buffer, :, :, :, w)
        scale = img_spec.filter_spec.wavelengths[w] / img_spec.filter_spec.base_wavelength
        factor = img_spec.filter_spec.intensities[w] / scale^2
        if scale ≈ 1
            bufs.psf_buffer .+= mono_psf_block .* factor
        else
            interpolate_add!(bufs.psf_buffer, mono_psf_block, bufs.interpolators[w], factor)
        end
    end
end

function readout!(dst::AbstractArray, img::AbstractArray, pc::PhotonCount, psf_norm)
    if isfinite_photons(pc)
        @. dst = rand(Poisson(real(img) / psf_norm * pc.nphotons + pc.background))
    else
        @. dst = img / psf_norm
    end
end
readout!(opt_buf::OpticalBuffers, pc::PhotonCount, psf_norm) =
    readout!(opt_buf.read_buffer, opt_buf.psf_buffer, pc, psf_norm)
function apply_truesky!(opt_buf::OpticalBuffers, ts::TrueSkyImage)
    #TODO Review for possible optimizations
    copyto!(opt_buf.focal_buffer, opt_buf.psf_buffer)
    mul!(opt_buf.aperture_buffer, opt_buf.fftplan, opt_buf.focal_buffer)
    opt_buf.aperture_buffer .*= ts.true_sky_fft
    ldiv!(opt_buf.focal_buffer, opt_buf.fftplan, opt_buf.aperture_buffer)
    opt_buf.psf_buffer .= real.(@view opt_buf.focal_buffer[:, :, :, 1])
end
function apply_truesky!(opt_buf::OpticalBuffers, ds::DoubleSystem)
    img = opt_buf.psf_buffer
    @assert all(abs.(ds.rel_position) .< image_size(opt_buf) .÷ 2)
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

struct ImgBufSerial{BT<:OpticalBuffers, ST<:ImagingSpec, FT<:Real}
    opt_buf::BT
    spec::ST
    psf_norm::FT
end
image_size(img_buf::ImgBufSerial) = image_size(img_buf.opt_buf)
image_type(img_buf::ImgBufSerial) = image_type(img_buf.opt_buf)
function prepare_imgbuffers(::Type{T}, img_spec::ImagingSpec, batch::Int, deviceadapter) where T
    img_spec_adapt = adapt(deviceadapter, img_spec)
    ImgBufSerial(OpticalBuffers(T, img_spec_adapt, batch), img_spec_adapt, psf_norm(img_spec))
end
function compute_images!(img_buf::ImgBufSerial, phases, true_sky)
    psf!(img_buf.opt_buf, img_buf.spec, phases)
    apply_truesky!(img_buf.opt_buf, true_sky)
    readout!(img_buf.opt_buf, img_buf.spec.photon_count, img_buf.psf_norm)
    return img_buf.opt_buf.read_buffer
end

struct ImgBufParallel{BT<:OpticalBuffers,ST<:ImagingSpec,FT<:Real,AT}
    opt_bufs::Vector{BT}
    spec::ST
    psf_norm::FT
    img_array::AT
end
image_size(img_buf::ImgBufParallel) = image_size(img_buf.opt_bufs[1])
image_type(img_buf::ImgBufParallel) = eltype(img_buf.img_array)
function prepare_imgbuffers(::Type{T}, img_spec::ImagingSpec, batch::Int, ::Type{<:Array}) where T
    imgbuf1 = OpticalBuffers(T, img_spec, 1)
    img_array = similar(imgbuf1.read_buffer, image_size(img_spec)..., batch)
    opt_buf_vector = Array{typeof(imgbuf1)}(undef, Threads.nthreads())
    opt_buf_vector[1] = imgbuf1
    Threads.@threads for i in 2:Threads.nthreads()
        opt_buf_vector[i] = OpticalBuffers(T, img_spec, 1)
    end
    return ImgBufParallel(opt_buf_vector, img_spec, psf_norm(img_spec), img_array)
end
function compute_images!(img_buf::ImgBufParallel, phases, true_sky)
    Threads.@threads for i in eachindex(img_buf.opt_bufs)
        op_buf = img_buf.opt_bufs[i]
        for j in i:length(img_buf.opt_bufs):size(phases, 3)
            psf!(op_buf, img_buf.spec, view(phases, :, :, j))
            apply_truesky!(op_buf, true_sky)
            readout!(view(img_buf.img_array, :, :, j), op_buf.psf_buffer, img_buf.spec.photon_count, img_buf.psf_norm)
        end
    end
    return img_buf.img_array
end
