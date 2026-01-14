using Test
using AtmosphericTurbulenceSimulator
using LinearAlgebra
using Statistics
using HDF5, ProgressMeter
using JET

include("test_constructors.jl")
include("test_atmosphere.jl")
include("test_imaging.jl")

@testset "JET Precompilation Report" begin
    jet_report = JET.report_package(AtmosphericTurbulenceSimulator; toplevel_logger=nothing,
        ignored_modules=[HDF5, ProgressMeter])
    print(jet_report)
    @test length(JET.get_reports(jet_report)) <= 24
    @test_broken length(JET.get_reports(jet_report)) == 0
end
