using CircularCMPS
using JLD2
using TensorKit, LinearAlgebra

c, μ, L = 1, 4.5, 16

k = parse(Int, ARGS[1])

@load "tmpdata/cmps_c$(c)_mu$(μ)_L$(L).jld2" ψ6

ψ = ψ6
χ = get_χ(ψ)

@show "Computing χ=$(χ)"

_, α = finite_env(K_mat(ψ, ψ), L)
ψ = rescale(ψ, -real(α), L)

p = k*2*pi/L

N1 = effective_N(ψ, p, L)
H1 = effective_H(ψ, p, L; c=c, μ=μ)

@show k
@show norm(H1 - H1') / χ^4 
@show norm(N1 - N1') / χ^4

H̃1 = sqrt(inv(N1)) * H1 * sqrt(inv(N1))
Es, _ = eigen(Hermitian(H̃1))

@save "tmpdata/excitation_c$(c)_mu$(μ)_L$(L)_k$(k)_chi$(χ).jld2" H1 N1 
