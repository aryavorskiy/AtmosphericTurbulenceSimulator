using Documenter
using AtmosphericTurbulenceSimulator

ENV["GKSwstype"] = "100"

DocMeta.setdocmeta!(AtmosphericTurbulenceSimulator, :DocTestSetup, :(using AtmosphericTurbulenceSimulator); recursive=true)

makedocs(
    sitename = "AtmosphericTurbulenceSimulator.jl",
    modules = [AtmosphericTurbulenceSimulator],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        assets = [asset("assets/favicon-96x96.png", class=:ico, islocal=true)],
    ),
    pages = [
        "Overview" => "index.md",
        "Examples" => "examples.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/aryavorskiy/AtmosphericTurbulenceSimulator.jl.git",
    devbranch = "master",
)
