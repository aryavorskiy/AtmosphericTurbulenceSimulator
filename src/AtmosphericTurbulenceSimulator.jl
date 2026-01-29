module AtmosphericTurbulenceSimulator

include("atmosphere.jl")
export kolmogorov_covmat, SingleLayer
include("imaging.jl")
export FilterSpec, ImagingSpec, PhotonCount, PointSource, DoubleSystem, TrueSkyImage,
    CircularAperture
include("simulation.jl")
export simulate_images, simulate_phases

include("precompile.jl")

end # module AtmosphericTurbulenceSimulator
