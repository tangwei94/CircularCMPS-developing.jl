using LinearAlgebra, TensorKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

hz = 1
hz = parse(Float64, ARGS[1])
T, Wmat = xxz_af_cmpo(1; hz=hz)

ψ = CMPSData(T.Q, T.Ls)

α = 2^(1/4)
βs = 0.32 * α .^ (0:32)

for β in βs
    global ψ
    ψ, f, E, var = power_iteration(T, Wmat, β, ψ, 1e-6; maxiter=200, maxχ=32)
    @save "heisenberg/results/cooling/heisenberg_hz$(hz)_beta$(β).jld2" β f E var ψ
    @show β, f, E, var
end
