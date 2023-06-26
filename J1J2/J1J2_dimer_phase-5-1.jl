using LinearAlgebra, TensorKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

J1, J2 = 1, 0.5
T, Wmat = heisenberg_j1j2_cmpo(J1, J2)

α = 2^(1/4)
βs = 0.64 * α .^ (0:27)

for β in βs[23:end]
    @load "J1J2/cooling/dimer_phase_beta$(β).jld2" f E var ψ
    ψ, f, E, var = power_iteration(T, Wmat, β, ψ, PowerMethod(tol_fidel=1e-6, tol_ES=1e-8, maxiter_compress=250))
    @save "J1J2/cooling/dimer_phase_beta-opt2$(β).jld2" β f E var ψ
    @show β, f, E, var
end


