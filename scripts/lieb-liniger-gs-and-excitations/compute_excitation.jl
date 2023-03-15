using CircularCMPS
using JLD2
using TensorKit, LinearAlgebra

c, μ, L = 1, 1.426, 16

k = parse(Int, ARGS[1])
@show k

@load "tmpdata/cmps_c$(c)_mu$(μ)_L$(L).jld2" ψ5

ψ = ψ5
χ = get_χ(ψ)

@show "Computing χ=$(χ)"

_, α = finite_env(K_mat(ψ, ψ), L)
ψ = rescale(ψ, -real(α), L)

p = k*2*pi/L

LinearAlgebra.BLAS.set_num_threads(1)
@show LinearAlgebra.BLAS.get_num_threads()

N1 = effective_N(ψ, p, L)
@show norm(N1 - N1') / χ^4
H1 = effective_H(ψ, p, L; c=c, μ=μ)
@show norm(H1 - H1') / χ^4 
M1 = effective_H(ψ, p, L; c=0, μ=-1, k0=0)
@show norm(M1 - M1') / χ^4

@save "tmpdata1/excitation_c$(c)_mu$(μ)_L$(L)_k$(k)_chi$(χ).jld2" H1 N1 M1 
