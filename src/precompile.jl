using PrecompileTools

@setup_workload begin
    imtypes = (Int, Float64)
    types = (Float64, Float32)
    trueskys = (DoubleSystem((5, 2), 0.5), TrueSkyImage(rand(32, 32)))
    for (IT, T, truesky) in zip(imtypes, types, trueskys)
        @compile_workload begin
            aperture = CircularAperture(T, (16, 16))
            photon_count = PhotonCount(1e6, 100)
            filter_spec = FilterSpec(500; bandwidth=100)
            img_spec = ImagingSpec(aperture, 2.0, photon_count; filter=filter_spec, img_size=(32, 32))
            atm2 = SingleLayer(T, 0.15, interpolate=1)
            img_buf, ph_buf2 = prepare_buffers(IT, atm2, img_spec, 1, Array)
            # simulation_run!!(BufferedDataset(nothing), BufferedDataset(nothing), ph_buf2, img_buf, truesky; n=1, verbose=false)
        end
    end
end
