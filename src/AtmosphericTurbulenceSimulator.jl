module AtmosphericTurbulenceSimulator

using Unitful
export @u_str
import Unitful: m, cm, mm, μm, nm
export m, cm, mm, μm, nm
import Unitful: s, ms, μs, ns
export s, ms, μs, ns

include("io.jl")
export HDF5File
include("atmosphere.jl")
export kolmogorov_covmat, SingleLayer, SavedPhases
include("imaging.jl")
export FilterSpec, ImagingSpec, PhotonCount, Exposure, PointSource, DoubleSystem, TrueSkyImage,
    CircularAperture, MultiThreaded
include("simulation.jl")
export simulate_images, simulate_phases

include("precompile.jl")

end # module AtmosphericTurbulenceSimulator
