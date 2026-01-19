using PrecompileTools

@setup_workload begin
    imtypes = (Int, Float64)
    types = (Float64, Float32)
    trueskys = (DoubleSystem((5, 2), 0.5), TrueSkyImage(rand(32, 32)))
    for (IT, T, truesky_noconv) in zip(imtypes, types, trueskys)
        aperture = CircularAperture(T, (16, 16))
        photon_count = PhotonCount(1e6, 100)
        filter_spec = FilterSpec(500; bandwidth=100)
        img_spec = ImagingSpec(aperture, photon_count; filter_spec=filter_spec, img_size=(32, 32))
        img_buf = prepare_imgbuffers(IT, img_spec, 1, Array)
        @compile_workload begin
            atm1 = SingleLayer(T, (16, 16), 50.0)
            ph_buf1 = prepare_phasebuffers(atm1, 1, Array)
            simulation_run!!(nothing, nothing, ph_buf1, img_buf, truesky_noconv; n=1, verbose=false)
            atm2 = SingleLayer(T, (16, 16), 50.0, interpolate=1)
            ph_buf2 = prepare_phasebuffers(atm2, 1, Array)
            simulation_run!!(nothing, nothing, ph_buf2, img_buf, truesky_noconv; n=1, verbose=false)
        end
    end
end
