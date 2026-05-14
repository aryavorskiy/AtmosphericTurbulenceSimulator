using LinearAlgebra, HDF5, Random, Adapt

abstract type AtmosphereSpec{T} end

"""
    kolmogorov_covmat(W)
    kolmogorov_covmat([T, ]size)

Compute the phase covariance matrix of a turbulent layer in the atmosphere, following the Kolmogorov model. The piston
term is excluded in this model. This function assumes unit Fried parameter ``r_0 = 1 px``.

# Arguments
- `T`: number type for the covariance matrix (default matches `W` if provided, otherwise `Float64`).
- `W`: the aperture function as a 2D array of weights. Normalized to `sum(W) == 1` internally.
- `size`: a tuple `(nx, ny)` specifying the size of the aperture function.
"""
function kolmogorov_covmat(W::AbstractMatrix)
    I = eachindex(IndexCartesian(), W)
    C = similar(W, length(I), length(I))
    for i in 1:length(I), j in 1:length(I)
        x = I[i][1] - I[j][1]
        y = I[i][2] - I[j][2]
        C[i, j] = -0.5 * 6.88 * (x^2 + y^2)^(5/6)
    end
    Wp = W ./ sum(W)
    Cp = vec(sum(C .* vec(Wp)', dims=2))
    Cc = sum(Cp .* vec(Wp))
    return C .- (Cp .+ Cp') .+ Cc
end
kolmogorov_covmat(::Type{T}, sz::NTuple{2,Int}) where T = kolmogorov_covmat(ones(T, sz))
kolmogorov_covmat(sz::NTuple{2,Int}) = kolmogorov_covmat(Float64, sz)

const EigenType = Union{Tuple{<:Any,<:Any}, Eigen}
struct KarhunenLoeveBuffers{MT}
    shape::NTuple{2,Int}
    noise_buffer::MT
    noise_transform::MT
    out_buffer::MT
end
function KarhunenLoeveBuffers(sz::NTuple{2,Int}, (E, U)::EigenType, batch::Int)
    @assert length(E) == prod(sz)
    @assert size(U) == (length(E), length(E))
    E .= clamp.(E, 0, Inf)
    noise_transform = U .* sqrt.(E')
    noise_buffer = similar(U, size(U, 2), batch)
    out_buffer = similar(U, prod(sz), batch)
    KarhunenLoeveBuffers(sz, noise_buffer, noise_transform, out_buffer)
end
plate_size(sampler::KarhunenLoeveBuffers) = sampler.shape
batch_length(sampler::KarhunenLoeveBuffers) = size(sampler.noise_buffer, 2)
phase_type(sampler::KarhunenLoeveBuffers) = eltype(sampler.out_buffer)
function samplephases!(sampler::KarhunenLoeveBuffers)
    randn!(sampler.noise_buffer)
    mul!(sampler.out_buffer, sampler.noise_transform, sampler.noise_buffer)
    return reshape(sampler.out_buffer, (sampler.shape..., size(sampler.out_buffer, 2)))
end

struct HardingSpec
    size_to::NTuple{2,Int}
    size_from::NTuple{2,Int}
    nsteps::Int
end
function HardingSpec(final_size::NTuple{2,Int}; interpolate=0, interpolate_from=nothing, size_heuristics=1024)
    if interpolate_from !== nothing
        any(interpolate_from .≤ 11) &&
            throw(ArgumentError("`interpolate_from` dimensions must be greater than 11."))
        interpolated_size = interpolate_from
        n = 0
        while any(final_size .> interpolated_size)
            interpolated_size = 2 .* interpolated_size .- 11
            n += 1
        end
        return HardingSpec(final_size, interpolate_from, n)
    elseif interpolate isa Number
        interpolate_from = cld.(final_size .- 11, 2^interpolate) .+ 11
        return HardingSpec(final_size, interpolate_from=interpolate_from)
    elseif interpolate === :auto
        n = 0
        interpolate_from = final_size
        while prod(interpolate_from) .> size_heuristics
            n += 1
            interpolate_from = cld.(final_size .- 11, 2^n) .+ 11
        end
        return HardingSpec(final_size, interpolate_from, n)
    else
        throw(ArgumentError("`interpolate` must be a Number or :auto"))
    end
end

"""
    SingleLayer([T, ]size, r0[; interpolate, interpolate_from, size_heuristics=1024])

An `AtmosphereSpec` that produces independent (uncorrelated) phase frames for each timestep.

# Arguments
- `T`: the number type for phase screens (default `Float64`).
- `size`: a tuple `(nx, ny)` specifying the phase screen shape in pixels (coarse sampler grid).
- `r0`: Fried parameter (r₀) in pixels.

# Keyword Arguments
- `interpolate`: when specified, the phase screen is sampled at a lower resolution and
    then upsampled using specified number of Harding interpolation passes. If set to `:auto`,
    the number of passes is chosen such that the low-res grid has at most `size_heuristics` total pixels.
- `interpolate_from`: alternatively, specify the low-res grid size directly. This must be
    greater than `(11, 11)` in each dimension.
- `size_heuristics`: when `interpolate=:auto`, the maximum allowed number of pixels
    in the low-res grid. Tweak this based on the capability of your hardware to compute `eigen`
    of a `N×N` matrix, where `N` is the number of pixels in the low-res grid.

# Notes
The Harding interpolation follows "Fast simulation of a Kolmogorov phase screen"
Cressida M. Harding, Rachel A. Johnston, and Richard G. Lane, APPLIED OPTICS Vol. 38, No. 11, April 1999
"""
struct SingleLayer{T<:Real,T2<:Real,KT} <: AtmosphereSpec{T}
    r₀::T
    wind_velocity::NTuple{2,T2}
    harding_kw::KT
end
function SingleLayer(r0::Real; wind_velocity=(0, 0), kw...)
    SingleLayer(float(r0), wind_velocity, kw)
end
SingleLayer(::Type{T}, r0::Real; kw...) where T =
    SingleLayer(convert(T, r0); kw...)
function prepare_phasebuffers(spec::SingleLayer, plate_size::NTuple{2,Int}, batch::Int, deviceadapter)
    harding = HardingSpec(plate_size; spec.harding_kw...)
    low_size = harding.size_from
    low_r₀ = spec.r₀ / 2^harding.nsteps
    covar = Adapt.adapt_storage(deviceadapter, kolmogorov_covmat(typeof(low_r₀), low_size))
    covar .*= low_r₀^(-5/3)
    E, U = eigen(Symmetric(covar))
    kl = KarhunenLoeveBuffers(low_size, (E, U), batch)
    return HardingInterpolator(kl, kl.out_buffer, low_r₀, harding)
end

struct HardingInterpolator{BT,NAT}
    base::BT
    out_bufs::NAT
    noise_std::Float64
    crop_size::NTuple{2,Int}
end
function HardingInterpolator(base, array, r0::Number, hspec::HardingSpec)
    low_size = hspec.size_from
    any(low_size .≤ 11) && throw(ArgumentError("Dimensions must be greater than 11"))
    bufs = map(1:hspec.nsteps) do i
        lsz = low_size
        for _ in 1:i
            lsz = 2 .* lsz .- 11
        end
        similar(array, (lsz .+ 10)..., batch_length(base))
    end
    return HardingInterpolator(base, bufs, sqrt(0.5265 / r0^(5/3)), hspec.size_to)
end
plate_size(sampler::HardingInterpolator) = sampler.crop_size
batch_length(sampler::HardingInterpolator) = batch_length(sampler.base)
phase_type(sampler::HardingInterpolator) = phase_type(sampler.base)

function samplephases!(harding::HardingInterpolator)
    low = samplephases!(harding.base)
    N = length(harding.out_bufs)
    N == 0 && return @view low[1:end, 1:end, :] # Ensure the same view type is returned
    harding_upsample!(harding.out_bufs[1], low, harding.noise_std)
    for i in 2:N
        prev_buffer = harding.out_bufs[i - 1]
        harding_upsample!(harding.out_bufs[i], @view(prev_buffer[6:end-5, 6:end-5, :]),
            harding.noise_std / 2^(5/6 * (i-1)))
    end
    out_buffer = harding.out_bufs[N]
    crop_offset = (size(out_buffer)[1:2] .- harding.crop_size) .÷ 2
    return @view out_buffer[
        crop_offset[1] + 1:crop_offset[1] + harding.crop_size[1],
        crop_offset[2] + 1:crop_offset[2] + harding.crop_size[2],
        :]
end
function harding_upsample!(to, from, noise_std)
    c_d = 0.3198
    c_m = -0.0341
    c_f = -0.0017

    # Padding offset
    n, m = size(from)
    inds_odd_x = range(5, length=n-4, step=2)
    inds_odd_y = range(5, length=m-4, step=2)
    inds_even_x = range(4, length=n-3, step=2)
    inds_even_y = range(4, length=m-3, step=2)

    # Copy low-res
    @views copy!(to[1:2:end, 1:2:end, :], from)

    # Interpolate checker pattern sites
    randn!(@view to[inds_even_x, inds_even_y, :])
    @views @. to[inds_even_x, inds_even_y, :] =
        noise_std * to[inds_even_x, inds_even_y, :] +
        c_d * (to[inds_even_x .+ 1, inds_even_y .+ 1, :] + to[inds_even_x .+ 1, inds_even_y .- 1, :] +
               to[inds_even_x .- 1, inds_even_y .+ 1, :] + to[inds_even_x .- 1, inds_even_y .- 1, :]) +
        c_m * ((to[inds_even_x .+ 3, inds_even_y .+ 1, :] + to[inds_even_x .+ 3, inds_even_y .- 1, :] +
               to[inds_even_x .- 3, inds_even_y .+ 1, :] + to[inds_even_x .- 3, inds_even_y .- 1, :]) +
               (to[inds_even_x .+ 1, inds_even_y .+ 3, :] + to[inds_even_x .+ 1, inds_even_y .- 3, :] +
               to[inds_even_x .- 1, inds_even_y .+ 3, :] + to[inds_even_x .- 1, inds_even_y .- 3, :])) +
        c_f * (to[inds_even_x .+ 3, inds_even_y .+ 3, :] + to[inds_even_x .+ 3, inds_even_y .- 3, :] +
               to[inds_even_x .- 3, inds_even_y .+ 3, :] + to[inds_even_x .- 3, inds_even_y .- 3, :])

    # Fill remaining sites
    randn!(@view to[inds_odd_x, inds_even_y, :])
    @views @. to[inds_odd_x, inds_even_y, :] =
        $(noise_std * 2^(-5/12)) * to[inds_odd_x, inds_even_y, :] +
        c_d * (to[inds_odd_x, inds_even_y .+ 1, :] + to[inds_odd_x, inds_even_y .- 1, :] +
                to[inds_odd_x .+ 1, inds_even_y, :] + to[inds_odd_x .- 1, inds_even_y, :]) +
        c_m * (to[inds_odd_x .+ 1, inds_even_y .+ 2, :] + to[inds_odd_x .+ 1, inds_even_y .- 2, :] +
                to[inds_odd_x .- 1, inds_even_y .+ 2, :] + to[inds_odd_x .- 1, inds_even_y .- 2, :] +
                to[inds_odd_x .+ 2, inds_even_y .+ 1, :] + to[inds_odd_x .+ 2, inds_even_y .- 1, :] +
                to[inds_odd_x .- 2, inds_even_y .+ 1, :] + to[inds_odd_x .- 2, inds_even_y .- 1, :]) +
        c_f * (to[inds_odd_x .+ 3, inds_even_y, :] + to[inds_odd_x .- 3, inds_even_y, :] +
                to[inds_odd_x, inds_even_y .+ 3, :] + to[inds_odd_x, inds_even_y .- 3, :])

    randn!(@view to[inds_even_x, inds_odd_y, :])
    @views @. to[inds_even_x, inds_odd_y, :] =
        $(noise_std * 2^(-5/12)) * to[inds_even_x, inds_odd_y, :] +
        c_d * (to[inds_even_x .+ 1, inds_odd_y, :] + to[inds_even_x .- 1, inds_odd_y, :] +
                to[inds_even_x, inds_odd_y .+ 1, :] + to[inds_even_x, inds_odd_y .- 1, :]) +
        c_m * (to[inds_even_x .+ 1, inds_odd_y .+ 2, :] + to[inds_even_x .+ 1, inds_odd_y .- 2, :] +
                to[inds_even_x .- 1, inds_odd_y .+ 2, :] + to[inds_even_x .- 1, inds_odd_y .- 2, :] +
                to[inds_even_x .+ 2, inds_odd_y .+ 1, :] + to[inds_even_x .+ 2, inds_odd_y .- 1, :] +
                to[inds_even_x .- 2, inds_odd_y .+ 1, :] + to[inds_even_x .- 2, inds_odd_y .- 1, :]) +
        c_f * (to[inds_even_x .+ 3, inds_odd_y, :] + to[inds_even_x .- 3, inds_odd_y, :] +
                to[inds_even_x, inds_odd_y .+ 3, :] + to[inds_even_x, inds_odd_y .- 3, :])
end

"""
    SavedPhases(dataset[; wind_velocity=(0, 0)])

An `AtmosphereSpec` that reuses phase screens saved in a dataset.

The dataset must match the phase-screen layout written by [`simulate_phases`](@ref) and
[`simulate_images`](@ref). Frames are read in order; if more frames are requested than
available, an error is thrown, the tail of the last batch is filled with `NaN`s.

# Arguments
- `dataset`: a 3D array-like containing phase screens with dimensions `(nx, ny, nframes)`.
  This can be an in-memory array or an HDF5 dataset (e.g. `HDF5.Dataset`).

# Keyword Arguments
- `wind_velocity`: two-component velocity used for long-exposure offsets in imaging simulations.
"""
struct SavedPhases{T<:Real,D,WT} <: AtmosphereSpec{T}
    dataset::D
    wind_velocity::NTuple{2,WT}
end
function SavedPhases(dataset; wind_velocity::NTuple{2,<:Real}=(0, 0))
    T = eltype(dataset)
    WT = typeof(wind_velocity[1])
    return SavedPhases{T,typeof(dataset),WT}(dataset, wind_velocity)
end

mutable struct SavedPhaseBuffers{D,B,DB,CT}
    dataset::D
    device_buffer::DB
    buffer::B
    crop_indices::CT
    next_frame::Int
end
plate_size(sampler::SavedPhaseBuffers) = size(sampler.buffer)[1:2]
batch_length(sampler::SavedPhaseBuffers) = size(sampler.buffer, 3)
phase_type(sampler::SavedPhaseBuffers) = eltype(sampler.buffer)
function prepare_phasebuffers(spec::SavedPhases{T}, plate_size::NTuple{2,Int}, batch::Int, deviceadapter) where T
    saved_size = size(spec.dataset)::NTuple{3,Int}
    saved_plate_size = saved_size[1:2]
    all(saved_plate_size .>= plate_size) || throw(ArgumentError(
        "Saved phase dataset has plate size $saved_plate_size, but at least $plate_size was requested."
    ))
    # TODO correct computation of crop indices
    crop_indices = (1:plate_size[1], 1:plate_size[2])
    device_buffer = zeros(T, (plate_size..., batch))
    buffer = Adapt.adapt_storage(deviceadapter, device_buffer)
    return SavedPhaseBuffers(spec.dataset, device_buffer, buffer, crop_indices, 1)
end
function samplephases!(sampler::SavedPhaseBuffers)
    nframes = size(sampler.dataset, 3)::Int
    blen = batch_length(sampler)
    zrange = range(sampler.next_frame, length=blen)
    sampler.next_frame > nframes &&
        throw(BoundsError(sampler.dataset, (sampler.crop_indices..., zrange)))
    if last(zrange) > nframes
        sampler.device_buffer[:, :, 1:(nframes - sampler.next_frame + 1)] =
            sampler.dataset[sampler.crop_indices..., first(zrange):nframes]
        sampler.device_buffer[:, :, (nframes - sampler.next_frame + 2):end] .= NaN
    else
        copyto!(sampler.device_buffer, sampler.dataset, sampler.crop_indices..., zrange)
    end
    sampler.next_frame += blen
    if typeof(sampler.buffer) !== typeof(sampler.device_buffer)
        copyto!(sampler.buffer, sampler.device_buffer)
        return sampler.buffer
    else
        return sampler.device_buffer
    end
end
