using Documenter
using SMLMBaGoL

DocMeta.setdocmeta!(SMLMBaGoL, :DocTestSetup, :(using SMLMBaGoL); recursive=true)

makedocs(;
    modules = [SMLMBaGoL],
    authors = "klidke@unm.edu",
    sitename = "SMLMBaGoL.jl",
    format = Documenter.HTML(;
        canonical = "https://juliasmlm.github.io/SMLMBaGoL.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "User Guide" => "guide.md",
        "Mathematics" => [
            "Overview" => "math/index.md",
            "Collapsed Representation" => "math/collapsed.md",
            "Spatial Models & Marginal Likelihood" => "math/marginal.md",
            "Priors" => "math/priors.md",
            "The Sampler: Moves" => "math/moves.md",
            "Hierarchical Learning" => "math/hierarchical.md",
            "MAP-N Estimation" => "math/mapn.md",
            "Large-Dataset Partitioning" => "math/partitioning.md",
            "Uncertainty Correction" => "math/se_adjust.md",
            "Per-Emitter Linear Motion" => "math/motion.md",
        ],
        "Results Gallery" => "results.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(;
    repo = "github.com/JuliaSMLM/SMLMBaGoL.jl",
    devbranch = "main",
)
