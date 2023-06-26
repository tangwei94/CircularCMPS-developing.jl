using LinearAlgebra, TensorKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

hz = 1
hz = parse(Float64, ARGS[1])
T, Wmat = xxz_af_cmpo(1; hz=hz)

α = 2^(1/4)
βs = 0.32 * α .^ (0:32)

ψ = CMPSData(T.Q, T.Ls)

for β in βs
    global ψ
    ψ, f, E, var = power_iteration(T, Wmat, β, ψ, PowerMethod(tol_ES=1e-7))
    @save "heisenberg/results/cooling/heisenberg_hz$(hz)_beta$(β)-2.jld2" β f E var ψ
    @show β, f, E, var
end
