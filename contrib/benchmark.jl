using AtmosphericTurbulenceSimulator, FFTW

print("Benchmarking atmospheric turbulence simulation...\n")
turb = SingleLayer(Float32, (99, 99), 0.2 / (2/100), interpolate=:auto)
aperture = CircularAperture(Float32, (99, 99))
pipeline = ImagingSpec(aperture, PhotonCount(1e7, 1.0), img_size=(256, 256), filter_spec=FilterSpec(1, bandwidth=0.1))
@time simulate_images(Int32, pipeline, turb, n=30000, savephases=false);
