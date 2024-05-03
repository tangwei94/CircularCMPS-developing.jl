using LinearAlgebra, TensorKit
using Revise
using CircularCMPS
using CairoMakie
using JLD2 

J1, J2 = 1, 0.5
T, Wmat = CircularCMPS.heisenberg_j1j2_cmpo_deprecated(J1, J2)
T2 = T*T

χs = [6, 9, 12]

ψ0 = CMPSData(T2.Q, T2.Ls)

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
        ψ, f, E, var = power_iteration(T2, Wmat, β, ψ; do_shifting=false)
        push!(fs, f/2)
        push!(Es, E/2)
        push!(vars, var)
        push!(ψs, ψ)
        @save "J1J2/deprecated-check-again/data/dimer_phase_blk2_beta$(β).jld2" fs Es vars ψs
    end
end