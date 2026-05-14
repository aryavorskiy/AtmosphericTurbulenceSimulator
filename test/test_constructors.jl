import AtmosphericTurbulenceSimulator: HardingSpec, prepare_buffers

@testset "Constructors" begin
    @testset "SingleLayer" begin
        atm = SingleLayer(5.0)
        @test atm isa SingleLayer
        @test atm.r₀ == 5.0

        # Test Harding interpolation
        hspec = HardingSpec((64, 64); interpolate=:auto)
        @test hspec.nsteps == 2
        @test hspec.size_to == (64, 64)
        @test hspec.size_from == (25, 25)

        hspec_1pass = HardingSpec((64, 64); interpolate=1)
        @test hspec_1pass.nsteps == 1
        @test hspec_1pass.size_to == (64, 64)
        @test hspec_1pass.size_from == (38, 38)

        # Test that small interpolate_from throws error
        @test_throws ArgumentError HardingSpec((64, 64); interpolate_from=(8, 8))
    end

    @testset "SavedPhases" begin
        tmpfile = tempname() * ".h5"
        data = reshape(Float32.(1:4*5*6), 4, 5, 6)
        h5write(tmpfile, "phases", data)

        h5open(tmpfile, "r") do fid
            atm = SavedPhases(fid["phases"]; wind_velocity=(1, 2))
            @test atm isa SavedPhases
            @test atm.wind_velocity == (1, 2)

            atm2 = SavedPhases(fid["phases"])
            @test atm2.wind_velocity == (0, 0)
        end
        rm(tmpfile, force=true)
    end

    @testset "FilterSpec" begin
        filter = FilterSpec(500; bandwidth=100, npts=5)
        @test filter.base_wavelength == 500
        @test filter.wavelengths == range(450, 550, length=5)
        @test filter.intensities == ones(5)
    end

    @testset "CircularAperture" begin
        ap = CircularAperture((32, 32), 15)
        @test size(ap) == (32, 32)
        @test all(0 .<= ap .<= 1)

        # center is filled
        @test ap[16, 16] ≈ 1.0
        # corners are empty
        @test ap[1, 1] ≈ 0.0
        @test ap[1, 32] ≈ 0.0
        @test ap[32, 1] ≈ 0.0
        @test ap[32, 32] ≈ 0.0

        # Test normalization (sum should equal area of circle, approximately)
        expected_pixels = π * 15^2
        @test sum(ap) ≈ expected_pixels rtol=0.05
    end

    @testset "ImagingSpec" begin
        ap = CircularAperture((32, 32), 15)
        img_spec = ImagingSpec(ap; nphotons=Inf)
        @test size(img_spec.aperture) == (32, 32)
        @test img_spec.img_size == (64, 64)

        img_spec_custom = ImagingSpec(ap; nphotons=Inf, nyquist_oversample=1.5)
        @test img_spec_custom.img_size == (96, 96)

        # Test with filter
        filter = FilterSpec(500.0; bandwidth=100.0)
        img_spec_filter = ImagingSpec(ap; nphotons=1e6, background=10, filter=FilterSpec(500.0; bandwidth=100.0))
        @test img_spec_filter.filter_spec.base_wavelength == 500.0
        @test img_spec_filter.photon_count.nphotons == 1e6
        @test img_spec_filter.photon_count.background == 10

        @test_throws ArgumentError ImagingSpec(ap; nphotons=1e6)
    end

    @testset "TrueSky models" begin
        # Test PointSource
        ps = PointSource()
        @test ps isa PointSource

        # Test DoubleSystem
        ds = DoubleSystem((5, 3), 0.5)
        @test ds.rel_position == (5, 3)
        @test ds.intensity == 0.5

        ds_f32 = DoubleSystem((5, 3), Float32(0.5))
        @test ds_f32.intensity isa Float32

        # Test TrueSkyImage
        test_image = rand(32, 32)
        ts_img = TrueSkyImage(test_image)
        @test size(ts_img.true_sky_fft) == (32, 32)

        ts_img_f32 = TrueSkyImage(convert.(Float32, test_image))
        @test eltype(ts_img_f32.true_sky_fft) == ComplexF32
    end

    @testset "ImgBufSerial & ImgBufParallel" begin
        ap = CircularAperture((16, 16))
        img_spec = ImagingSpec(ap; nphotons=1e6, background=100)
        atm = SingleLayer(5.0)

        img_buf_serial = prepare_buffers(Int32, atm, img_spec, 5, identity)[2]
        @test img_buf_serial isa AtmosphericTurbulenceSimulator.ImgBufSerial
        @test length(img_buf_serial.offsets) == 1
        @test img_buf_serial.offsets[1].can_ff

        img_buf_parallel = prepare_buffers(Int32, atm, img_spec, 5, Array)[2]
        @test img_buf_parallel isa AtmosphericTurbulenceSimulator.ImgBufParallel
        @test length(img_buf_parallel.opt_bufs) == Threads.nthreads()
        @test length(img_buf_parallel.offsets) == 1
        @test img_buf_parallel.offsets[1].can_ff

        img_spec2 = ImagingSpec(ap; nphotons=1e6, background=100, exposure=Exposure(0.1, 10))
        img_buf2 = prepare_buffers(Int32, atm, img_spec2, 5, identity)[2]
        @test length(img_buf2.offsets) == 1
        @test img_buf2.offsets[1].can_ff

        atm2 = SingleLayer(5.0; interpolate=:auto, wind_velocity=(10.0, 5.0))
        img_buf3 = prepare_buffers(Int32, atm2, img_spec2, 5, identity)[2]
        @test length(img_buf3.offsets) == 10
        @test img_buf3.offsets[1].can_ff
        @test !img_buf3.offsets[2].can_ff

        img_spec3 = ImagingSpec(ap; nphotons=1e6, background=100, exposure=Exposure(0, 10))
        img_buf4 = prepare_buffers(Int32, atm2, img_spec3, 5, identity)[2]
        @test length(img_buf4.offsets) == 1
        @test img_buf4.offsets[1].can_ff
    end
end
