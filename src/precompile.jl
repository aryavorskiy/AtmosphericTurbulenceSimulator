using PrecompileTools

@setup_workload begin
    tmpfile = tempname() * ".h5"
    imtypes = (Int, Float64)
    types = (Float64, Float32)
    trueskys = (DoubleSystem((5, 2), 0.5, nphotons=1e6), TrueSkyImage(rand(32, 32)))
    for (IT, T, truesky_noconv) in zip(imtypes, types, trueskys)
        img_spec = ImagingSpec(CircularAperture(T, (16, 16)), FilterSpec(500; bandpass=100))
        truesky = convert(TrueSky{T}, truesky_noconv)
        img_buf = prepare_imgbuffers(IT, img_spec, 1, Array)
        @compile_workload begin
            atm1 = SingleLayer(T, (16, 16), 50.0)
            ph_buf1 = prepare_phasebuffers(atm1, 1, Array)
            simulation_run!!(nothing, nothing, ph_buf1, img_buf, truesky; n=1, verbose=false)
            atm2 = SingleLayer(T, (16, 16), 50.0, interpolate=1)
            ph_buf2 = prepare_phasebuffers(atm2, 1, Array)
            simulation_run!!(nothing, nothing, ph_buf2, img_buf, truesky; n=1, verbose=false)
        end
    end
end
