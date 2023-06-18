using LinearAlgebra, TensorKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

J1, J2 = 1, 0.5
T, Wmat = heisenberg_j1j2_cmpo(J1, J2)

ψ = CMPSData(T.Q, T.Ls)

α = 2^(1/4)
βs = 0.64 * α .^ (0:27)

for β in βs
    global ψ
    ψ, f, E, var = power_iteration(T, Wmat, β, ψ, 1e-6; maxiter=100, maxχ=20)
    @save "J1J2/cooling/dimer_phase_beta$(β).jld2" β f E var ψ
    @show β, f, E, var
end


