using LinearAlgebra, TensorKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

# following 1908.02068
V = 1
Ω = 0.01
Δ = 4*Ω
#hz = parse(Float64, ARGS[1])
T, Wmat = rydberg_cmpo(Ω, Δ, V)

α = 2^(1/4)
βs = 0.32 * α .^ (0:32)

ψ = CMPSData(T.Q, T.Ls)

for β in βs
    global ψ
    ψ, f, E, var = power_iteration(T, Wmat, β, ψ, PowerMethod(tol_ES=1e-7))
    @save "rydberg/results/rydberg_Omega$(Ω)_Delta$(Δ)_beta$(β)_tolES1e-7.jld2" β f E var ψ
    @show β, f, E, var
end
