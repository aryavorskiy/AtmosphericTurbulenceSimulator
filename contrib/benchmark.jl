using AtmosphericTurbulenceSimulator, FFTW

print("Benchmarking atmospheric turbulence simulation...\n")
atm = SingleLayer(Float64, 0.2 / (2/100), interpolate=:auto, wind_velocity=(50, 0) ./ (2/100))
aperture = CircularAperture(Float64, (99, 99))
img_spec = ImagingSpec(aperture, PhotonCount(1e6, 1.0), img_size=(256, 256),
    filter_spec=FilterSpec(1, bandwidth=0.1),
    exposure=Exposure(0.04, 10))
@time simulate_images(img_spec, atm, n=300, savephases=true, filename="simulation_mw.h5");
