using GreggShorthandTools
using Documenter

DocMeta.setdocmeta!(GreggShorthandTools, :DocTestSetup, :(using GreggShorthandTools); recursive=true)

makedocs(;
    modules=[GreggShorthandTools],
    authors="sand",
    sitename="GreggShorthandTools",
    format=Documenter.HTML(;
        canonical="https://sandsq.github.io/GreggShorthandTools",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/sandsq/GreggShorthandTools",
    devbranch="main",
)
