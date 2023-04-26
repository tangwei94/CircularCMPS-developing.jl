using LinearAlgebra, TensorKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

hz = 1
hz = parse(Float64, ARGS[1])
maxχ = parse(Int, ARGS[2])
T, Wmat = xxz_af_cmpo(1; hz=hz)

χs = [2, 4, 8, 12, 16]

ψ0 = CMPSData(T.Q, T.Ls)

α = 2^(1/4)
βs = 0.32 * α .^ (0:32)

for β in βs[end:-1:1]
    global ψ0
    ψ = ψ0
    fs, Es, vars = Float64[], Float64[], Float64[]
    ψs = CMPSData[]
    for χ in χs[1:maxχ]
        ψ = expand(ψ, χ, β; perturb=1e-9)
        ψ, f, E, var = power_iteration(T, Wmat, β, ψ; DIIS_D=10, spect_shifting=2)
        push!(fs, f)
        push!(Es, E)
        push!(vars, var)
        push!(ψs, ψ)
        @save "heisenberg/results/heisenberg_hz$(hz)_beta$(β).jld2" fs Es vars ψs
    end
end
