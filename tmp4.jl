using LinearAlgebra, TensorKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

hz = 0.05
T, Wmat = xxz_af_cmpo(1; hz=hz)

α = 2^(1/4)
βs = 0.32 * α .^ (0:32)

β = βs[30]

ψ = CMPSData(T.Q, T.Ls)
result1 = power_iteration(T, Wmat, β, ψ, 1e-6; maxχ=2, maxiter=100);
result2 = power_iteration(T, Wmat, β, result1[1], 1e-6; maxχ=4, maxiter=100);
result3 = power_iteration(T, Wmat, β, result2[1], 1e-6; maxχ=8, maxiter=100);
result4 = power_iteration(T, Wmat, β, result3[1], 1e-6; maxχ=12, maxiter=100);
