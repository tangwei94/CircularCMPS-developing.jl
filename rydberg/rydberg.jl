using LinearAlgebra, TensorKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

# following 1908.02068
V = 1
y = 2.0
Ω = y^(-6)
x = 4
Δ = x*Ω
#hz = parse(Float64, ARGS[1])
T, Wmat = rydberg_cmpo(Ω, Δ, V)
T2 = T * T

@show norm(T.Q)

α = 2^(1/4)
βs = 0.32 * α .^ (0:32)

ψ = CMPSData(T.Q, T.Ls)

@load "rydberg/results/rydberg_y$(y)_x$(x)_beta$(βs[24])_tolES1e-7.jld2" ψ

for β in βs[25:25]
    global ψ
    ψ, f, E, var = power_iteration(T, Wmat, β, ψ, PowerMethod(tol_ES=1e-7, spect_shifting=1, maxiter_power=100))
    @save "rydberg/results/rydberg_y$(y)_x$(x)_beta$(β)_tolES1e-7.jld2" β f E var ψ
    @show β, f, E, var
end
