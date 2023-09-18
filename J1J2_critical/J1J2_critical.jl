using LinearAlgebra, TensorKit
using Revise
using CircularCMPS
using CairoMakie
using JLD2 

J1, J2 = 1, 0.241167
T, Wmat = heisenberg_j1j2_cmpo(J1, J2)
T1, Wmat1 = CircularCMPS.heisenberg_j1j2_cmpo_deprecated(J1, J2)

α = 2^(1/4)
βs = 0.64 * α .^ (0:27)

ψ = CMPSData(T.Q, T.Ls)

for β in βs
    global ψ 
    ψ, f, E, var = power_iteration(T, Wmat, β, ψ, PowerMethod())
    @save "J1J2_critical/data/beta$(β).jld2" β f E var ψ
end

ψ = CMPSData(T1.Q, T1.Ls)
for β in βs
    global ψ 
    ψ, f, E, var = power_iteration(T1, Wmat1, β, ψ, PowerMethod())
    @save "J1J2_critical/data/deprecated_beta$(β).jld2" β f E var ψ
end
