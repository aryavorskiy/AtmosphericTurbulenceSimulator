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
        # convenience constructor
        filter = FilterSpec(500.0; bandpass=100.0, npts=5)
        @test filter.base_wavelength == 500.0
        @test length(filter.wavelengths) == 5

        # Test type promotion
        filter32 = FilterSpec(Float32, 500.0; bandpass=100.0)
        @test filter32 isa FilterSpec{Float32}
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
        img_spec = ImagingSpec(ap)
        @test size(img_spec.aperture) == (32, 32)
        @test img_spec.img_size == (64, 64)

        img_spec_custom = ImagingSpec(ap, nyquist_oversample=1.5)
        @test img_spec_custom.img_size == (96, 96)

        # Test with filter
        filter = FilterSpec(500.0; bandpass=100.0)
        img_spec_filter = ImagingSpec(ap, filter)
        @test img_spec_filter.filter_spec.base_wavelength == 500.0
    end

    @testset "TrueSky models" begin
        # Test PointSource
        ps_finite = PointSource(1e7, 200)
        @test ps_finite.nphotons == 1e7
        @test ps_finite.background == 200
        @test AtmosphericTurbulenceSimulator.isfinite_photons(ps_finite)
        ps_inf = PointSource()
        @test !AtmosphericTurbulenceSimulator.isfinite_photons(ps_inf)

        ps_f32 = convert(AtmosphericTurbulenceSimulator.TrueSky{Float32}, ps_finite)
        @test ps_f32.nphotons isa Float32

        # Test DoubleSystem
        ds = DoubleSystem((5, 3), 0.5; nphotons=1e7, background=100)
        @test ds.rel_position == (5, 3)
        @test ds.intensity == 0.5
        @test ds.brightness.nphotons == 1e7
        @test ds.brightness.background == 100

        ds_f32 = convert(AtmosphericTurbulenceSimulator.TrueSky{Float32}, ds)
        @test ds_f32.intensity isa Float32

        # Test TrueSkyImage
        test_image = rand(32, 32)
        ts_img = TrueSkyImage(test_image; nphotons=Inf)
        @test size(ts_img.true_sky_fft) == (32, 32)
        @test !AtmosphericTurbulenceSimulator.isfinite_photons(ts_img)

        ts_img_f32 = convert(AtmosphericTurbulenceSimulator.TrueSky{Float32}, ts_img)
        @test eltype(ts_img_f32.true_sky_fft) == ComplexF32
    end
end
