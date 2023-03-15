using CircularCMPS
using JLD2
using TensorKit, LinearAlgebra

#c, μ, L = 1, 1.42, 8
c, μ, L = 1, 1.426, 16
#L = parse(Int, ARGS[1]) 

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

ψ1 = CMPSData(rand, 4, 1)
ψ1, E, grad, numfg, history = lieb_liniger_ground_state(c, μ, L, ψ1)
_, α1 = finite_env(K_mat(ψ1, ψ1), L)
ψ1 = rescale(ψ1, -real(α1), L)

@show fE(ψ1), fN(ψ1)
@save "tmpdata/cmps_c$(c)_mu$(μ)_L$(L).jld2" ψ1 

ψ2 = expand(ψ1, 8, L)
ψ2, E2, grad, numfg, history = lieb_liniger_ground_state(c, μ, L, ψ2)
_, α2 = finite_env(K_mat(ψ2, ψ2), L)
ψ2 = rescale(ψ2, -real(α2), L)

@show fE(ψ2), fN(ψ2)
@save "tmpdata/cmps_c$(c)_mu$(μ)_L$(L).jld2" ψ1 ψ2  

ψ3 = expand(ψ2, 12, L)
ψ3, E3, grad, numfg, history = lieb_liniger_ground_state(c, μ, L, ψ3)
_, α3 = finite_env(K_mat(ψ3, ψ3), L)
ψ3 = rescale(ψ3, -real(α3), L)

@show fE(ψ3), fN(ψ3)
@save "tmpdata/cmps_c$(c)_mu$(μ)_L$(L).jld2" ψ1 ψ2 ψ3  

ψ4 = expand(ψ3, 16, L)
ψ4, E4, grad, numfg, history = lieb_liniger_ground_state(c, μ, L, ψ4)
_, α4 = finite_env(K_mat(ψ4, ψ4), L)
ψ4 = rescale(ψ4, -real(α4), L)

@show fE(ψ4), fN(ψ4)
@save "tmpdata/cmps_c$(c)_mu$(μ)_L$(L).jld2" ψ1 ψ2 ψ3 ψ4  

ψ5 = expand(ψ4, 20, L)
ψ5, E5, grad, numfg, history = lieb_liniger_ground_state(c, μ, L, ψ5)
_, α5 = finite_env(K_mat(ψ5, ψ5), L)
ψ5 = rescale(ψ5, -real(α5), L)

@show fE(ψ5), fN(ψ5)
@save "tmpdata/cmps_c$(c)_mu$(μ)_L$(L).jld2" ψ1 ψ2 ψ3 ψ4 ψ5  

#ψ6 = expand(ψ5, 24, L)
#ψ6, E6, grad, numfg, history = lieb_liniger_ground_state(c, μ, L, ψ6)
#_, α6 = finite_env(K_mat(ψ6, ψ6), L)
#ψ6 = rescale(ψ6, -real(α6), L)
#
#@show fE(ψ6), fN(ψ6)
#@save "tmpdata/cmps_c$(c)_mu$(μ)_L$(L).jld2" ψ1 ψ2 ψ3 ψ4 ψ5 ψ6 
#
#ψ7 = expand(ψ6, 28, L)
#ψ7, E7, grad, numfg, history = lieb_liniger_ground_state(c, μ, L, ψ7)
#_, α7 = finite_env(K_mat(ψ7, ψ7), L)
#ψ7 = rescale(ψ7, -real(α7), L)
#
#@show fE(ψ7), fN(ψ7)
#@save "tmpdata/cmps_c$(c)_mu$(μ)_L$(L).jld2" ψ1 ψ2 ψ3 ψ4 ψ5 ψ6 ψ7 
#
#ψ8 = expand(ψ7, 32, L)
#ψ8, E8, grad, numfg, history = lieb_liniger_ground_state(c, μ, L, ψ8)
#_, α8 = finite_env(K_mat(ψ8, ψ8), L)
#ψ8 = rescale(ψ8, -real(α8), L)
#
#@show fE(ψ8), fN(ψ8)
#@save "tmpdata/cmps_c$(c)_mu$(μ)_L$(L).jld2" ψ1 ψ2 ψ3 ψ4 ψ5 ψ6 ψ7 ψ8 
