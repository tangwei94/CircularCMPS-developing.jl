using LinearAlgebra, TensorKit
using Revise
using CircularCMPS
using CairoMakie
using JLD2 

J1, J2 = 1, 0.5
T, Wmat = CircularCMPS.heisenberg_j1j2_cmpo_deprecated(J1, J2)

χs = [3, 6, 9, 12]

ψ0 = CMPSData(T.Q, T.Ls)

α = 2^(1/4)
βs = 1.28 * α .^ (0:23)

steps = 1:100

for β in βs[end:-1:end-10]
    global ψ0
    ψ = ψ0
    fs, Es, vars = Float64[], Float64[], Float64[]
    ψs = CMPSData[]
    for χ in χs
        ψ = expand(ψ, χ, β; perturb=1e-9)
        ψ, f, E, var = power_iteration(T, Wmat, β, ψ)
        push!(fs, f)
        push!(Es, E)
        push!(vars, var)
        push!(ψs, ψ)
        @save "J1J2/deprecated-check-again/data/dimer_phase_beta$(β).jld2" fs Es vars ψs
    end
end
