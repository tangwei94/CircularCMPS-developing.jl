using CircularCMPS
using Documenter

DocMeta.setdocmeta!(CircularCMPS, :DocTestSetup, :(using CircularCMPS); recursive=true)

makedocs(;
    modules=[CircularCMPS],
    authors="Wei Tang <tangwei@smail.nju.edu.cn> and contributors",
    repo="https://github.com/tangwei94/CircularCMPS.jl/blob/{commit}{path}#{line}",
    sitename="CircularCMPS.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://tangwei94.github.io/CircularCMPS.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/tangwei94/CircularCMPS.jl",
)
