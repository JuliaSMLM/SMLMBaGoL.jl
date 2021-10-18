using SMLMBaGoL
using Documenter

DocMeta.setdocmeta!(SMLMBaGoL, :DocTestSetup, :(using SMLMBaGoL); recursive=true)

makedocs(;
    modules=[SMLMBaGoL],
    authors="klidke@unm.edu",
    repo="https://github.com/kalidke/SMLMBaGoL.jl/blob/{commit}{path}#{line}",
    sitename="SMLMBaGoL.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kalidke.github.io/SMLMBaGoL.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kalidke/SMLMBaGoL.jl",
    devbranch = "main"
)
