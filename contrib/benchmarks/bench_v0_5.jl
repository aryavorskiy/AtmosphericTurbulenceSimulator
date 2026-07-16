# Benchmark for any v0.5.x release
#
# Usage:
#   julia -t 8 --project=<worktree>/test bench_v0_5.jl <version_tag> [output.csv]
#
# Arguments:
#   version_tag  – label written into the CSV
#   output.csv   – path for results (default: results_<version_tag>.csv)
#
# The three benchmark cases are:
#   mono      – monochromatic, zero exposure time
#   broadband – 7-wavelength filter, zero exposure time
#   long_exp  – monochromatic, 7-step long exposure with wind

using AtmosphericTurbulenceSimulator

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 1 || error("Usage: bench_v0_5.jl <version_tag> [output.csv]")
    const VERSION_TAG = lowercase(ARGS[1])
    outfile = length(ARGS) >= 2 ? ARGS[2] : "results_$(VERSION_TAG).csv"
else
    const VERSION_TAG = string(Base.pkgversion(AtmosphericTurbulenceSimulator))
    !@isdefined(outfile) && (outfile = "/dev/null")
end

include(joinpath(@__DIR__, "benchmark_utils.jl"))

atm_nowind    = SingleLayer(r0; interpolate=:auto)
atm_wind      = SingleLayer(r0; wind_velocity=WIND, interpolate=:auto)

img_spec_mono = ImagingSpec(aperture, DIAMETER, PhotonCount(Inf); img_size=(256, 256))
img_spec_bb   = ImagingSpec(aperture, DIAMETER, PhotonCount(Inf);
                    filter=FilterSpec(550; bandwidth=55), img_size=(256, 256))
img_spec_long = ImagingSpec(aperture, DIAMETER, PhotonCount(Inf);
                    exposure=Exposure(EXPTIME, 7), img_size=(256, 256))

cases = [
    "mono"      => (atm_nowind, img_spec_mono),
    "broadband" => (atm_nowind, img_spec_bb),
    "long_exp"  => (atm_wind,   img_spec_long),
]

run_benchmarks(outfile, cases, Ns) do (atm, img_spec), N
    simulate_images(atm, img_spec;
        n=N, batch=BATCH, file=HDF5File(TMPFILE, overwrite=true), verbose=false, savephases=false)
end
