using LinearAlgebra, HDF5

"""
    turbulence_covmat(W, r₀, pixel_size)

Compute the phase covariance matrix of a turbulent layer in the atmosphere. The piston
term is excluded in this model.

# Arguments
- `W`: the aperture function. Either a 2D array, or a `(x, y)` tuple representing the size
    of the aperture (in this case the aperture function is assumed to be a square of this
    size).
- `r₀::Real`: the Fried parameter.
- `pixel_size::Real`: the sampling interval in the spatial frequency domain.
"""
function turbulence_covmat(sz::NTuple{2}, r₀, pixel_size, constraint=nothing)
    I = CartesianIndices(sz)
    C = Array{Float64}(undef, length(I), length(I))

    for i in 1:length(I), j in 1:length(I)
        x = I[i][1] - I[j][1]
        y = I[i][2] - I[j][2]
        r = sqrt(x^2 + y^2)
        C[i, j] = -0.5 * 6.88 * (r * pixel_size / r₀)^(5/3)
    end
    if constraint === nothing
        Cp = vec(sum(C, dims=2)) / prod(sz)
        Cc = sum(Cp) / prod(sz)
    elseif constraint isa AbstractMatrix
        @assert size(constraint) == sz
        @assert sum(constraint) ≈ 1
        Cp = vec(sum(C .* vec(constraint)', dims=2))
        Cc = sum(Cp .* vec(constraint))
    else
        error("Invalid constraint of type $(typeof(constraint))")
    end
    return Symmetric(C .- Cp .- Cp' .+ Cc)
end

turbulence_covmat(W::AbstractMatrix, r₀, pixel_size) =
    turbulence_covmat(size(W), r₀, pixel_size, W)

const EigenType = Union{Tuple{<:Any,<:Any}, Eigen}
struct CovariantNoise{MT}
    shape::NTuple{2,Int}
    noise_transform::MT
end
function CovariantNoise(sz::NTuple{2,Int}, (E, U)::EigenType)
    @assert length(E) == prod(sz)
    @assert size(U) == (length(E), length(E))
    E .= clamp.(E, 0, Inf)
    CovariantNoise(sz, U .* sqrt.(E'))
end

function samplephases!(phases, sampler::CovariantNoise, orth_noise=randn(size(sampler.noise_transform, 2)))
    @assert length(phases) == size(sampler.noise_transform, 1)
    @assert size(orth_noise, 1) == size(sampler.noise_transform, 2)
    mul!(reshape(phases, (size(sampler.noise_transform, 1), :)), sampler.noise_transform, orth_noise)
    return phases
end
samplephases(sampler::CovariantNoise) = samplephases!(similar(sampler.noise_transform, sampler.shape), sampler)
covmat(sampler::CovariantNoise) = sampler.noise_transform * sampler.noise_transform'

function project_sampler(sampler::CovariantNoise, basis)
    @assert size(basis, 1) == size(sampler.noise_transform, 1)
    basisq = Matrix(qr(basis).Q)
    return CovariantNoise(sampler.shape, basisq * basisq' * sampler.noise_transform)
end

struct TurbulenceStatistics{VT, MT}
    weights::MT
    covmat_E::VT
    covmat_U::MT
    sampler::CovariantNoise{MT}
    function TurbulenceStatistics(wts::AbstractMatrix, E::AbstractVector, U::AbstractMatrix)
        @assert length(E) == prod(size(wts))
        @assert size(U) == (length(E), length(E))
        new{typeof(E), typeof(U)}(wts / sum(wts), E, U, CovariantNoise(size(wts), (E, U)))
    end
end
function TurbulenceStatistics(wts::AbstractMatrix, cov::AbstractMatrix)
    @assert issymmetric(cov)
    E, U = eigen(Symmetric(cov))
    TurbulenceStatistics(wts, E, U)
end
TurbulenceStatistics(sz::NTuple{2}, args...) = TurbulenceStatistics(ones(sz), args...)

Base.size(turb::TurbulenceStatistics) = size(turb.weights)
Base.length(turb::TurbulenceStatistics) = prod(size(turb))

function h5dump(file, turb::TurbulenceStatistics; overwrite=false)
    h5open(file, overwrite ? "w" : "cw") do fid
        write(fid, "covmat_E", turb.covmat_E)
        write(fid, "covmat_U", turb.covmat_U)
        write(fid, "weights", turb.weights)
    end
end
function h5load(file, ::Type{TurbulenceStatistics})
    h5open(file, "r") do fid
        E = read(fid, "covmat_E")
        U = read(fid, "covmat_U")
        weights = read(fid, "weights")
        return TurbulenceStatistics(weights, E, U)
    end
end

covmat(turb::TurbulenceStatistics) = covmat(turb.sampler)
turbulence_covmat(turb::TurbulenceStatistics) = covmat(turb.sampler)
samplephases!(phases, turb::TurbulenceStatistics, args...) = samplephases!(phases, turb.sampler, args...)
samplephases(turb::TurbulenceStatistics) = samplephases(turb.sampler)

"""
    turbulence_invcovmat(sampler[; zero_piston=true])

Compute the inverse of the phase covariance matrix of a turbulent layer in the atmosphere.

# Arguments
- `sampler`: the `TurbulenceStatistics` object.
- `zero_piston`: if `true` (default), the piston term is set to zero in the inverse matrix. Otherwise,
    it is effectively infinite, but regularized to avoid numerical issues.
"""
function turbulence_invcovmat(turb::TurbulenceStatistics; zero_piston=true)
    if zero_piston
        _inv(x) = abs(x) < 1e-8 ? zero(x) : inv(x)
        iE = _inv.(turb.covmat_E)
    else
        iE = inv.(turb.covmat_E .+ 1e-8)
    end
    return Symmetric(turb.covmat_U * Diagonal(iE) * turb.covmat_U')
end
