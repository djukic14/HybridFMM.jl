using HybridFMM
using Documenter

DocMeta.setdocmeta!(HybridFMM, :DocTestSetup, :(using HybridFMM); recursive=true)

makedocs(;
    modules=[HybridFMM],
    authors="djukic14 <danijel.jukic14@gmail.com> and contributors",
    sitename="HybridFMM.jl",
    format=Documenter.HTML(;
        canonical="https://djukic14.github.io/HybridFMM.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/djukic14/HybridFMM.jl",
    devbranch="main",
)
