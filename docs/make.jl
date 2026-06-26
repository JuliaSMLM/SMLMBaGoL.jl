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
        "Model & Sampler" => "model.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(;
    repo = "github.com/JuliaSMLM/SMLMBaGoL.jl",
    devbranch = "main",
)
