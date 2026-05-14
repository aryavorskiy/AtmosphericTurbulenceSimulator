module AtmosphericTurbulenceSimulator

include("atmosphere.jl")
export kolmogorov_covmat, SingleLayer, SavedPhases
include("imaging.jl")
export FilterSpec, ImagingSpec, PhotonCount, Exposure, PointSource, DoubleSystem, TrueSkyImage,
    CircularAperture
include("simulation.jl")
export HDF5File, simulate_images, simulate_phases

include("precompile.jl")

end # module AtmosphericTurbulenceSimulator
