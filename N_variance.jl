using CircularCMPS
using JLD2 
using TensorKit, LinearAlgebra 
using LiebLinigerBA
using CairoMakie
using Polynomials

#c, μ, L = 1, 1.42, 8
c, μ = 1, 1.426

function fE(ψ::CMPSData, L)
    OH = kinetic(ψ) + c * point_interaction(ψ) - μ * particle_density(ψ)
    expK, _ = finite_env(K_mat(ψ, ψ), L)
    return real(tr(expK * OH))
end

function fN(ψ::CMPSData, L)
    ON = particle_density(ψ)
    expK, _ = finite_env(K_mat(ψ, ψ), L)
    return real(tr(expK * ON))
end

function fN2(ψ::CMPSData, L)
    opN2 = sum(K_otimes.(ψ.Rs .* ψ.Rs, ψ.Rs .* ψ.Rs))
    expK, _ = finite_env(K_mat(ψ, ψ), L)
    return real(tr(expK * opN2))
end

function fC2(ψ::CMPSData, L)
    opN = sum(K_otimes.(ψ.Rs, ψ.Rs))
    Kmat = K_mat(ψ, ψ)
    Λ = diag(eig(Kmat)[1].data)
    @show real(Λ[end] - Λ[end-1])
    C2 = Coeff2(Kmat, 0, L)
    return C2(opN, opN) 
end

function varN(ψ, L)
    opN = sum(K_otimes.(ψ.Rs, ψ.Rs))
    Kmat = K_mat(ψ, ψ)
    C2 = Coeff2(Kmat, 0, L)
    ρgs = fN(ψ, L)
    return C2(opN, opN) + ρgs - L*ρgs^2
end

Ls = [16, 32, 64, 96, 128, 160, 320, 480, 640, 800, 960];#, 1120, 1280, 1440, 1600, 2000, 2400, 2800];
VarNs_12 = Float64[];
VarNs_16 = Float64[];
VarNs_20 = Float64[];
for L in Ls
    @load "tmpdata/cmps_c$(c)_mu$(μ)_L$(L).jld2" ψ3 ψ4 ψ5
    
    push!(VarNs_12, real(varN(ψ3, L)))
    push!(VarNs_16, real(varN(ψ4, L)))
    push!(VarNs_20, real(varN(ψ5, L)))
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (450, 450))

ax1 = Axis(fig[1, 1], 
        xlabel = L"L",
        ylabel = L"\mathrm{var}(N) / L",
        )

scatter!(ax1, Ls, VarNs_12, label=L"\chi=12")
lines!(ax1, Ls, VarNs_12)
scatter!(ax1, Ls, VarNs_16, label=L"\chi=16")
lines!(ax1, Ls, VarNs_16)
scatter!(ax1, Ls, VarNs_20, label=L"\chi=20")
lines!(ax1, Ls, VarNs_20)
@show fig

axislegend(ax1, position=:rb, framevisible=false)
@show fig
Makie.save("fig-varN.pdf", fig)