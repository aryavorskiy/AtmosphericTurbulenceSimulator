@testset "Constructors" begin
    @testset "SingleLayer" begin
        # Test construction without interpolation
        atm = SingleLayer((16, 16), 5.0)
        @test atm isa AtmosphericTurbulenceSimulator.SingleLayer
        @test atm.r₀ == 5.0

        # Test with interpolation
        atm_auto = SingleLayer((64, 64), 5.0; interpolate=:auto)
        @test atm_auto isa SingleLayer{Float64, 2}
        @test atm_auto.harding.interpolate_to == (64, 64)
        @test atm_auto.harding.interpolate_from == (25, 25)

        atm_1pass = SingleLayer((64, 64), 5.0; interpolate=1)
        @test atm_1pass.harding.interpolate_to == (64, 64)
        @test atm_1pass.harding.interpolate_from == (38, 38)

        # Test that small interpolate_from throws error
        @test_throws ArgumentError SingleLayer((64, 64), 5.0; interpolate_from=(8, 8))
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
        img_spec_filter = ImagingSpec(ap; nphotons=1e6, background=10, filter_spec=filter)
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
end
