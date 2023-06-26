using LinearAlgebra, TensorKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

J1, J2 = 1, 0.5
T, Wmat = heisenberg_j1j2_cmpo(J1, J2)

α = 2^(1/4)
βs = 0.64 * α .^ (0:27)

ψ = CMPSData(T.Q, T.Ls)

for β in βs[1:3]
    global ψ 
    ψ, f, E, var = power_iteration(T, Wmat, β, ψ, PowerMethod(tol_fidel=1e-8, tol_ES=1e-6, maxiter_compress=250))
    @save "J1J2/cooling1/dimer_phase_beta$(β).jld2" β f E var ψ
    @show β, f, E, var
end


