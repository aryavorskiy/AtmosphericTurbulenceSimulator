using PrecompileTools

@setup_workload begin
    tmpfile = tempname() * ".h5"
    types = (Float64, Float32)
    trueskys = (DoubleSystem((5, 2), 0.5, nphotons=1e6), TrueSkyImage(rand(32, 32)))
    for (T, truesky_noconv) in zip(types, trueskys)
        img_spec = ImagingSpec(CircularAperture(T, (16, 16)), FilterSpec(500; bandpass=100))
        truesky = convert(TrueSky{T}, truesky_noconv)
        tmpfile1 = tempname() * ".h5"
        tmpfile2 = tempname() * ".h5"
        @compile_workload begin
            atm1 = SingleLayer(T, (16, 16), 50.0)
            simulate_images(img_spec, atm1, truesky; n=8, verbose=false, filename=tmpfile1)
            atm2 = SingleLayer(T, (16, 16), 50.0, interpolate=1)
            simulate_images(img_spec, atm2, truesky; n=8, verbose=false, filename=tmpfile2)
        end
        rm(tmpfile1, force=true)
        rm(tmpfile2, force=true)
    end
end
