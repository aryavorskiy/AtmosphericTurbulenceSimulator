using Test
using AtmosphericTurbulenceSimulator
using LinearAlgebra, Random, Statistics
using HDF5, ProgressMeter, ChunkSplitters
using JET

include("test_constructors.jl")
include("test_io.jl")
include("test_atmosphere.jl")
include("test_imaging.jl")
if get(ENV, "RUN_JET_TESTS", "true") == "true"
    # Skip JET tests on Julia LTS in CI
    include("test_jet.jl")
end
