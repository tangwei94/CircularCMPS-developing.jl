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

for β in βs
    @load "heisenberg/results/cooling/heisenberg_hz$(hz)_beta$(β).jld2" ψ
    ψ, f, E, var = power_iteration(T, Wmat, β, ψ, PowerMethod(tol_fidel=1e-6, tol_ES=1e-8, maxiter_compress=200))
    @save "heisenberg/results/cooling/heisenberg_hz$(hz)_beta$(β)-2.jld2" β f E var ψ
    @show β, f, E, var
end
