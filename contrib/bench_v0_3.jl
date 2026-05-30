# Benchmark for v0.3.x
#
# Usage:
#   julia -t 8 --project=<worktree>/test bench_v0_3.jl <version_tag> [output.csv]
#
# API differences from v0.4.x:
#   - SingleLayer requires an explicit plate size: SingleLayer((nx,ny), r0; ...)
#   - ImagingSpec uses keyword `filter_spec=` instead of `filter=`
#   - simulate_images argument order: (img_spec, atm; ...) — img_spec before atm
#   - Output file passed as `filename=` (plain String), always written (no file=nothing)
#   - No Exposure / long-exposure support → only mono and broadband cases

using AtmosphericTurbulenceSimulator

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 1 || error("Usage: bench_v0_3.jl <version_tag> [output.csv]")
    const VERSION_TAG = lcase(ARGS[1])
    outfile = length(ARGS) >= 2 ? ARGS[2] : "results_$(VERSION_TAG).csv"
else
    const VERSION_TAG = string(Base.pkgversion(AtmosphericTurbulenceSimulator))
    !@isdefined(outfile) && (outfile = "/dev/null")
end

include(joinpath(@__DIR__, "benchmark_utils.jl"))

atm = SingleLayer((99, 99), r0; interpolate=:auto)

img_spec_mono = ImagingSpec(aperture, PhotonCount(Inf); img_size=(256, 256))
img_spec_bb   = ImagingSpec(aperture, PhotonCount(Inf);
                    filter_spec=FilterSpec(1; bandwidth=0.1), img_size=(256, 256))

cases = [
    "mono"      => img_spec_mono,
    "broadband" => img_spec_bb,
]

run_benchmarks(outfile, cases, Ns) do img_spec, N
    simulate_images(img_spec, atm;
        n=N, batch=BATCH, filename=TMPFILE, verbose=false, savephases=false)
end
