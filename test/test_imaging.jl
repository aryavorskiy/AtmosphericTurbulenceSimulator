@testset "Imaging" begin
    ap = CircularAperture((16, 16))
    atm = SingleLayer((16, 16), 5)
    @testset "True sky" begin
        ts1 = PointSource()
        ts2 = DoubleSystem((3, 2), 0.6)
        ts3 = TrueSkyImage(rand(32, 32))

        pc = PhotonCount(1e6, 100)
        img_spec = ImagingSpec(ap, pc)
        img_spec2 = ImagingSpec(ap, pc, filter_spec=FilterSpec(1, bandwidth=0.1))
        img_spec3 = ImagingSpec(ap, pc, filter_spec=FilterSpec(1, bandwidth=0.1, tedge=0.5))

        for (ts, is) in zip((ts1, ts2, ts3), (img_spec, img_spec2, img_spec3))
            tmpfile = tempname() * ".h5"
            simulate_images(Int32, is, atm, ts; n=16, filename=tmpfile, verbose=false)

            h5open(tmpfile, "r") do fid
                @test haskey(fid, "images")
                @test haskey(fid, "phases")
                @test haskey(fid, "aperture")

                images = read(fid["images"])
                phases = read(fid["phases"])
                aperture = read(fid["aperture"])

                @test size(images) == (32, 32, 16)
                @test size(phases) == (16, 16, 16)
                @test size(aperture) == (16, 16)

                # Total photon count should be approximately correct
                # (within reasonable variance due to Poisson noise)
                total_photons = sum(images, dims=(1,2))
                expected = 1e6 + 100.0 * 32 * 32
                @test total_photons ≈ fill(expected, size(total_photons)) rtol=0.05
                @test eltype(images) == Int32
            end

            rm(tmpfile, force=true)
        end

        @test_throws ArgumentError simulate_images(img_spec, SingleLayer((15, 15), 5), ts1; n=16)
        @test_throws ArgumentError simulate_images(img_spec, atm, TrueSkyImage(rand(16, 16)); n=16)
    end

    @testset "Continuous vs Poisson" begin
        # Continuous flux
        img_spec_cont = ImagingSpec(ap; nphotons=Inf)
        ts_cont = PointSource()
        @test_throws ArgumentError simulate_images(Int32, img_spec_cont, atm, ts_cont; n=16)

        tmpfile = tempname() * ".h5"
        simulate_images(img_spec_cont, atm, ts_cont; n=16, filename=tmpfile, verbose=false, deviceadapter=identity)
        h5open(tmpfile, "r") do fid
            images_cont = read(fid["images"])
            @test eltype(images_cont) == Float64
            @test all(>=(-eps()), images_cont)
        end
        rm(tmpfile, force=true)

        # Poisson sampling
        img_spec_poisson = ImagingSpec(ap; nphotons=1e6, background=100)
        ts_poisson = PointSource()
        tmpfile = tempname() * ".h5"
        simulate_images(img_spec_poisson, atm, ts_poisson; n=16, filename=tmpfile, verbose=false, deviceadapter=identity)
        h5open(tmpfile, "r") do fid
            images_poisson = read(fid["images"])
            @test eltype(images_poisson) == Int64
            @test all(>=(0), images_poisson)
        end
        rm(tmpfile, force=true)
    end

    @testset "No phases" begin
        ts = PointSource()
        img_spec_float = ImagingSpec(ap; nphotons=Inf)

        tmpfile = tempname() * ".h5"
        simulate_images(Float32, img_spec_float, atm, ts; n=16, filename=tmpfile, savephases=false,
            verbose=false)

        h5open(tmpfile, "r") do fid
            @test haskey(fid, "images")
            @test !haskey(fid, "phases")  # Should not save phases
            @test haskey(fid, "aperture")
        end

        rm(tmpfile, force=true)
    end
end
