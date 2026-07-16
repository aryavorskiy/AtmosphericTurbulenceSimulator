# Shared simulation parameters and benchmarking utilities

aperture = CircularAperture(Float64, (99, 99))

# Simulation parameters
const DIAMETER = 2
const r0       = 0.2
const WIND     = (45.0, 22.0)             # wind velocity [px/time]  (v0.4+ only)
const EXPTIME  = 0.04                     # exposure time [time]      (v0.4+ only)
const Ns       = [200, 400, 800, 1500, 3000, 6000, 12000, 20000]
const BATCH    = 128

const TMPFILE = tempname() * ".h5"

# Timing helper
function time_function(f, nruns=1)
    minimum(@elapsed(f()) for _ in 1:nruns)
end

# Benchmark runner
# f(params, N) — version-specific simulate_images call; writes to TMPFILE
# cases        — iterable of name => params pairs
function run_benchmarks(f, outfile, cases, ns)
    println("[$VERSION_TAG] Warming up (N=$(ns[1]))…")
    f(cases[1][2], ns[1])   # warmup run to compile the function
    isfile(TMPFILE) && rm(TMPFILE, force=true)

    open(outfile, "w") do io
        println(io, "case,N,time_s")
        for (name, params) in cases
            println("[$VERSION_TAG] case=$name")
            for N in ns
                t = time_function(N < 1000 ? 3 : 1) do
                    f(params, N)
                end
                isfile(TMPFILE) && rm(TMPFILE, force=true)
                println(io, "$name,$N,$(round(t; digits=4))")
                println("  N=$N  =>  $(round(t; digits=3)) s")
            end
        end
    end
    println("[$VERSION_TAG] Results written to $outfile")
end
