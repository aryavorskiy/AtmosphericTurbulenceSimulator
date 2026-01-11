@testset "Imaging" begin
    ap = CircularAperture((16, 16))
    img_spec = ImagingSpec(ap, nyquist_oversample=1)
    atm = SingleLayer((16, 16), 5)
    @testset "True sky" begin
        ts1 = PointSource(1e6, 100)
        ts2 = DoubleSystem((3, 2), 0.6; nphotons=1e6, background=100)
        ts3 = TrueSkyImage(rand(32, 32); nphotons=1e6, background=100)

        for ts in (ts1, ts2, ts3)
            tmpfile = tempname() * ".h5"
            simulate_images(Int32, img_spec, atm, ts; n=16, filename=tmpfile, verbose=false)

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
        ap = CircularAperture((16, 16), 6.0)
        img_spec = ImagingSpec(ap)
        atm = SingleLayer((16, 16), 100.0)  # Very large r0 for minimal turbulence

        # Continuous flux
        ts_cont = PointSource(Inf, 0.0)
        @test_throws ArgumentError simulate_images(Int32, img_spec, atm, ts_cont; n=16)

        tmpfile = tempname() * ".h5"
        simulate_images(img_spec, atm, ts_cont; n=16, filename=tmpfile, verbose=false)
        h5open(tmpfile, "r") do fid
            images_cont = read(fid["images"])
            @test eltype(images_cont) == Float64
            @test all(>=(-eps()), images_cont)
        end
        rm(tmpfile, force=true)

        # Poisson sampling
        ts_poisson = PointSource(1e7, 0.0)
        tmpfile = tempname() * ".h5"
        simulate_images(img_spec, atm, ts_poisson; n=16, filename=tmpfile, verbose=false)
        h5open(tmpfile, "r") do fid
            images_poisson = read(fid["images"])
            @test eltype(images_poisson) == Int64
            @test all(>=(0), images_poisson)
        end
        rm(tmpfile, force=true)
    end

    @testset "No phases" begin
        ts = PointSource()

        tmpfile = tempname() * ".h5"
        simulate_images(Float32, img_spec, atm, ts; n=16, filename=tmpfile, savephases=false,
            verbose=false)

        h5open(tmpfile, "r") do fid
            @test haskey(fid, "images")
            @test !haskey(fid, "phases")  # Should not save phases
            @test haskey(fid, "aperture")
        end

        rm(tmpfile, force=true)
    end
end
