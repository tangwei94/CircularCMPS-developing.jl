using LinearAlgebra, TensorKit, KrylovKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

Γ = 1.0
T, Wmat = ising_cmpo(Γ)
ψ = CMPSData(T.Q, T.Ls)

α = 2^(1/4)
βs = 0.32 * α .^ (0:32)

for β in βs
    global ψ
    ψ, f, E, var = power_iteration(T, Wmat, β, ψ, PowerMethod(tol_ES=1e-7))
    @save "ising/results/ising_Gamma$(Γ)_beta$(β)-toles7.jld2" β f E var ψ
    @show β, f, E, var
end

β = βs[30]

@load "ising/results/ising_Gamma0.1_beta$(β)-toles7.jld2" ψ
Λ, U = eigen(ψ.Q)
Q1 = U' * ψ.Q * U
R1 = U' * ψ.Rs[1] * U
norm(R1[2]) - norm(R1[3])

ψ1 = CMPSData(ψ.Q, -ψ.Rs)

ln_ovlp(ψ1, ψ, β) + ln_ovlp(ψ, ψ1, β) - ln_ovlp(ψ, ψ, β) - ln_ovlp(ψ1, ψ1, β)