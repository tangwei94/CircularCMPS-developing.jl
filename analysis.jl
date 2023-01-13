using CircularCMPS
using JLD2 
using TensorKit, LinearAlgebra 

c, μ, L = 1, 4.5, 16

function fE(ψ::CMPSData)
    OH = kinetic(ψ) + c*point_interaction(ψ) - μ * particle_density(ψ)
    expK, _ = finite_env(K_mat(ψ, ψ), L)
    return real(tr(expK * OH))
end

function fN(ψ::CMPSData)
    ON = particle_density(ψ)
    expK, _ = finite_env(K_mat(ψ, ψ), L)
    return real(tr(expK * ON))
end

@load "tmpdata/cmps_c$(c)_mu$(μ)_L$(L).jld2" ψ6

ψ = ψ6
χ = get_χ(ψ)

Egs = fE(ψ) 
Ngs = fN(ψ) 

k = 0 
@load "tmpdata/excitation_c$(c)_mu$(μ)_L$(L)_k$(k).jld2" H1 N1

H̃1 = sqrt(inv(N1)) * H1 * sqrt(inv(N1))
Es, _ = eigen(Hermitian(H̃1))

Es[1:16] ./ L
