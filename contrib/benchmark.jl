using AtmosphericTurbulenceSimulator

print("Benchmarking atmospheric turbulence simulation...\n")
turb = KolmogorovUncorrelated((64, 64), 0.2, 2/64)
aperture = CircularAperture((64, 64))
pipeline = ImagingPipeline(aperture)

@assert all(isfinite, turb.sampler.noise_transform)
@time simulate_images(pipeline, turb, n=1000)
@time simulate_images(pipeline, turb, n=1000)
