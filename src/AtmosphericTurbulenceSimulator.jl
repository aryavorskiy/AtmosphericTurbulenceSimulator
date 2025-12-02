module AtmosphericTurbulenceSimulator

include("atmosphere.jl")
export kolmogorov_covmat, KolmogorovUncorrelated
include("imaging.jl")
export BandSpec, ImagingPipeline, imgsize, psf, psf!, CircularAperture, simulate_images

end # module AtmosphericTurbulenceSimulator
