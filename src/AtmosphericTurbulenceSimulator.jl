module AtmosphericTurbulenceSimulator

include("io.jl")
export HDF5File
include("atmosphere.jl")
export kolmogorov_covmat, SingleLayer, SavedPhases
include("imaging.jl")
export FilterSpec, ImagingSpec, PhotonCount, Exposure, PointSource, DoubleSystem, TrueSkyImage,
    CircularAperture
include("simulation.jl")
export simulate_images, simulate_phases

include("precompile.jl")

end # module AtmosphericTurbulenceSimulator
