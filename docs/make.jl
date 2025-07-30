using shorthand_detector
using Documenter

DocMeta.setdocmeta!(shorthand_detector, :DocTestSetup, :(using shorthand_detector); recursive=true)

makedocs(;
    modules=[shorthand_detector],
    authors="sand",
    sitename="shorthand_detector.jl",
    format=Documenter.HTML(;
        canonical="https://sandsq.github.io/shorthand_detector.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/sandsq/shorthand_detector.jl",
    devbranch="main",
)
