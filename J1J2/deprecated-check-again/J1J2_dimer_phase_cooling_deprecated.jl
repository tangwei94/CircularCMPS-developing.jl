using LinearAlgebra, TensorKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

J1, J2 = 1, 0.5
T1, Wmat1 = CircularCMPS.heisenberg_j1j2_cmpo_deprecated(J1, J2)

α = 2^(1/4)
βs = 0.64 * α .^ (0:27)

ψ = CMPSData(T1.Q, T1.Ls)
β = βs[end-3]
@show β
@load "J1J2/deprecated-check-again/data/dimer_phase_beta$(β)_cooling.jld2" ψ
for β in βs[end-2:end]
    global ψ 
    ψ, f, E, var = power_iteration(T1, Wmat1, β, ψ, PowerMethod(tol_fidel=1e-8, tol_ES=1e-7, maxiter_compress=250))
    @save "J1J2/deprecated-check-again/data/dimer_phase_beta$(β)_cooling.jld2" β f E var ψ
    @show β, f, E, var
end