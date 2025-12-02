module AtmosphericTurbulenceSimulator

include("atmosphere.jl")
export turbulence_covmat, turbulence_invcovmat, TurbulenceStatistics, samplephases, samplephases!
include("imaging.jl")
export BandSpec, ImagingPipeline, imgsize, psf, psf!, CircularAperture, simulate_images

end # module AtmosphericTurbulenceSimulator
