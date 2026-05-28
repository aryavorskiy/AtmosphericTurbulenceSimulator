@testset "Imaging" begin
    ap = CircularAperture((16, 16))
    atm = SingleLayer(5, wind_velocity=(1, 1))
    @testset "True sky" begin
        ts1 = PointSource()
        ts2 = DoubleSystem((3, 2), 0.6)
        ts3 = TrueSkyImage(rand(32, 32))

        pc = PhotonCount(1e6, 100)
        img_spec = ImagingSpec(ap, pc)
        img_spec2 = ImagingSpec(ap, pc, filter=FilterSpec(1, bandwidth=0.1))
        img_spec3 = ImagingSpec(ap, pc, filter=FilterSpec(1, bandwidth=0.1, tedge=0.5), exposure=Exposure(3, 5))

        for (ts, is) in zip((ts1, ts2, ts3), (img_spec, img_spec2, img_spec3))
            res = simulate_images(Int32, ts, atm, is; n=16, file=nothing, verbose=false)
            @test res isa NamedTuple
            @test keys(res) == (:phases, :images)
            images = res.images
            phases = res.phases

            @test size(images) == (32, 32, 16)
            @test size(phases) == (iszero(is.exposure_spec.exptime) ? (16, 16, 16) : (19, 19, 16))

            # Total photon count should be approximately correct
            # (within reasonable variance due to Poisson noise)
            total_photons = sum(images, dims=(1, 2))
            expected = 1e6 + 100.0 * 32 * 32
            @test total_photons ≈ fill(expected, size(total_photons)) rtol = 0.05
            @test eltype(images) == Int32
        end

        @test_throws ArgumentError simulate_images(TrueSkyImage(rand(16, 16)), atm, img_spec; n=16)
    end

    @testset "Continuous vs Poisson" begin
        # Continuous flux
        img_spec_cont = ImagingSpec(ap, PhotonCount(Inf))
        ts_cont = PointSource()
        @test_throws ArgumentError simulate_images(Int32, ts_cont, atm, img_spec_cont; n=16)

        images_cont = simulate_images(ts_cont, atm, img_spec_cont; n=16, file=nothing, verbose=false, deviceadapter=identity).images
        @test eltype(images_cont) == Float64
        @test all(>=(-eps()), images_cont)

        # Poisson sampling
        img_spec_poisson = ImagingSpec(ap, PhotonCount(1e6, 100))
        ts_poisson = PointSource()
        images_poisson = simulate_images(ts_poisson, atm, img_spec_poisson; n=16, file=nothing, verbose=false, deviceadapter=identity).images

        @test eltype(images_poisson) == Int64
        @test all(>=(0), images_poisson)
    end

    @testset "No phases" begin
        ts = PointSource()
        img_spec_float = ImagingSpec(ap, PhotonCount(Inf))

        tmpfile = tempname() * ".h5"
        simulate_images(Float32, ts, atm, img_spec_float; n=16, file=tmpfile, savephases=false,
            verbose=false)

        h5open(tmpfile, "r") do fid
            @test haskey(fid, "images")
            @test !haskey(fid, "phases")  # Should not save phases
        end

        rm(tmpfile, force=true)
    end

    @testset "Saved phases" begin
        tmpfile = tempname() * ".h5"
        img_spec1 = ImagingSpec(ap, PhotonCount(Inf))
        simulate_images(atm, img_spec1, n=10, file=tmpfile)
        img1 = h5read(tmpfile, "images")
        img2 = h5open(tmpfile, "r") do fid
            saved_atm = SavedPhases(fid["phases"]; wind_velocity=(1, 1))
            simulate_images(saved_atm, img_spec1, n=10).images
        end
        rm(tmpfile, force=true)
        @test img1 == img2

        img_spec2 = ImagingSpec(ap, PhotonCount(Inf), exposure=Exposure(3, 3))
        res_e = simulate_images(atm, img_spec2, n=10)
        img_e1 = res_e.images
        img_e2 = simulate_images(SavedPhases(res_e.phases; wind_velocity=(1, 1)), img_spec2,
            n=10, batch=7, verbose=false).images
        @test img_e1 == img_e2

        img_spec3 = ImagingSpec(ap, PhotonCount(1e6, 100), exposure=Exposure(3, 3))
        phs, img_d1 = simulate_images(atm, img_spec3, n=10, batch=7, verbose=false)
        Random.seed!(123)
        img_d2 = simulate_images(SavedPhases(phs; wind_velocity=(1, 1)), img_spec3,
            n=10, batch=7, verbose=false).images
        Random.seed!(123)
        img_d3 = simulate_images(SavedPhases(phs; wind_velocity=(-1, -1)), img_spec3,
            n=11, batch=7, verbose=false).images
        Random.seed!(123)
        @test img_d2 == img_d3[:, :, 1:10] # Should be the same except for the last frame

        @test_throws ArgumentError simulate_images(SavedPhases(res_e.phases; wind_velocity=(2, 2)), img_spec2, n=10)
        @test_throws BoundsError simulate_images(SavedPhases(res_e.phases; wind_velocity=(1, 1)), img_spec2, n=20, batch=7)
    end
end
