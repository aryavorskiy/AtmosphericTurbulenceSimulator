@testset "Atmosphere" begin
    structure_function(C) = diag(C) .+ diag(C)' .- 2 .* C
    @testset "Kolmogorov covariance matrix" begin
        # Test basic properties
        sz = (4, 5)
        C = kolmogorov_covmat(sz)
        C2 = kolmogorov_covmat(ones(sz))
        C3 = kolmogorov_covmat(rand(sz...))
        @test size(C) == (prod(sz), prod(sz))
        @test size(C3) == (prod(sz), prod(sz))
        @test C == C2
        @test issymmetric(C)
        @test issymmetric(C3)

        # Test that structure function is correct
        D = structure_function(C)
        D3 = structure_function(C3)
        @test all(≥(0), D)
        @test D ≈ D3

        # Test type stability
        @test eltype(C) == Float64
        @test eltype(kolmogorov_covmat(Float32, sz)) == Float32
        @test eltype(kolmogorov_covmat(rand(Float32, sz...))) == Float32
    end

    @testset "Phase screen generation" begin
        atm = SingleLayer(5.0)
        tmpfile = tempname() * ".h5"
        res = simulate_phases(atm, (32, 32); n=16, file=tmpfile, verbose=false)
        @test res === nothing  # When filename is given, should return nothing
        phases = h5read(tmpfile, "phases")
        @test size(phases) == (32, 32, 16)
        @test eltype(phases) == Float64
        rm(tmpfile, force=true)
    end

    @testset "Phase statistics" begin
        # Generate phase screens and check basic statistics
        atm1 = SingleLayer(Float32, 5, interpolate=2)
        atm2 = SingleLayer(Float32, 5, interpolate_from=(15, 15))

        # Use temporary file
        for atm in (atm1, atm2)
            phases = simulate_phases(atm, (64, 64); n=1000, file=nothing, verbose=false)
            @test eltype(phases) == Float32

            for D in [(3, 2), (3, 15), (5, 5), (5, 20), (16, 2), (16, 16)]
                diff = @views phases[D[1]+1:end, D[2]+1:end, :] .- phases[1:end-D[1], 1:end-D[2], :]
                emp_structure = mean(abs2, diff)
                @test emp_structure ≈ 6.88 * (hypot(D...) / 5)^(5/3) rtol=0.1
            end
        end
    end
end
