using LinearAlgebra
using LaTeXStrings
using CairoMakie
using LiebLinigerBA
using Colors
using JLD2
using TensorKit
using Revise
using CircularCMPS

c, μ, L = 1, 1.426, 16
N = Int(L) 
ρ0 = N / L 

@show get_mu(c, L, N)

# ============ the BA solution ============ 
# the ground state
ψ0 = ground_state(c, L, N);
ψ1 = ph_excitation(ψ0, [0], [-1]);
ψ1l = ph_excitation(ψ0, [-1], [0]);
Egs = energy(ψ0);
v, K = v_and_K(c, L, N);

ΔE = energy(ψ1) - Egs
scale_E(x::Float64) = (x - Egs) / ΔE
scale_E(ψ::LLBAState) = (energy(ψ) - Egs) / ΔE

# excited states
ph_actions = [[0], [-1], [-2], [-1, -1], [-3], [-2, -1], [-1, -1, -1]];
states = LLBAState[];
for qL in ph_actions
    for qR in ph_actions
        push!(states, ph_excitation(ψ0, qL, qR))
    end
end

msk_ba = scale_E.(energy.(states)) .< 4.5
states = states[msk_ba];

momenta = momentum.(states)
energies = energy.(states)

function ρ0_form_factor(ψ1, ψ2, target)
    ln_norm_ρ, phase_ρ = ln_ρ0_form_factor(ψ1, ψ2; target=target, p=0.314159)
    return exp(ln_norm_ρ) * phase_ρ
end

#ρ0_ovlps_gs = [ρ0_form_factor(ψi, ψ0, :density) for ψi in states]
#j0_ovlps_gs = [ρ0_form_factor(ψi, ψ0, :current) for ψi in states]

chiral_ovlps_gs = [kacmoody(ψi, ψ0, :holomorphic, v, K) for ψi in states]
antichiral_ovlps_gs = [kacmoody(ψi, ψ0, :antiholomorphic, v, K) for ψi in states]
chiral_ovlps_ψ1 = [kacmoody(ψi, ψ1, :holomorphic, v, K) for ψi in states]
antichiral_ovlps_ψ1 = [kacmoody(ψi, ψ1, :antiholomorphic, v, K) for ψi in states]

# ============ the cMPS results ============ 
function fE(ψ::CMPSData)
    OH = kinetic(ψ) + c * point_interaction(ψ) - μ * particle_density(ψ)
    expK, _ = finite_env(K_mat(ψ, ψ), L)
    return real(tr(expK * OH))
end

function fN(ψ::CMPSData)
    ON = particle_density(ψ)
    expK, _ = finite_env(K_mat(ψ, ψ), L)
    return real(tr(expK * ON))
end

@load "tmpdata/cmps_c$(c)_mu$(μ)_L$(L).jld2" ψ3 ψ4 ψ5

ψ = ψ5
χ = get_χ(ψ)

Egs_cmps = fE(ψ) * L 
Ngs_cmps = fN(ψ) * L 
ΔN = 0.5

momenta_cmps = [0.0]
energies_cmps = [Egs_cmps]
numbers_cmps = [Ngs_cmps]

Es_cmps = []
Vs_cmps = []

for k in -3:3
    global momenta_cmps, energies_cmps, numbers_cmps
    @load "tmpdata1/excitation_c$(c)_mu$(μ)_L$(L)_k$(k)_chi$(χ).jld2" H1 N1 M1

    H̃1 = sqrt(inv(N1)) * H1 * sqrt(inv(N1))
    M̃1 = sqrt(inv(N1)) * M1 * sqrt(inv(N1))
    Es, Vs = eigen(Hermitian(H̃1))
    Ns = real.(diag(Vs' * M̃1 * Vs))

    msk = (abs.(Ns .- Ngs_cmps) .< ΔN) .& (Es .- Egs_cmps .< ΔE * 5) 

    numbers_cmps = vcat(numbers_cmps, Ns[msk])
    energies_cmps = vcat(energies_cmps, Es[msk])
    momenta_cmps = vcat(momenta_cmps, fill(k*2*pi/L, length(Es[msk])))

    push!(Es_cmps, Es)
    push!(Vs_cmps, Vs)
end

# apply Kac-Moody from ground state 
v_cmps = (minimum(Es_cmps[4+1]) - Egs_cmps) / (2*pi/L)
K_cmps = 2*pi*(Ngs_cmps/L)/ v_cmps

numbers_cmps = ComplexF64[Ngs_cmps]
chiral_ovlps_gs_cmps = ComplexF64[0]
antichiral_ovlps_gs_cmps = ComplexF64[0]
chiral_ovlps_ψ1_cmps = ComplexF64[]
antichiral_ovlps_ψ1_cmps = ComplexF64[]

V0 = Vs_cmps[4+1][:, 1]
@load "tmpdata1/excitation_c$(c)_mu$(μ)_L$(L)_k1_chi$(χ).jld2" N1
V0 = sqrt(inv(N1)) * V0

for k in -3:3
    @load "tmpdata1/excitation_c$(c)_mu$(μ)_L$(L)_k$(k)_chi$(χ).jld2" H1 N1 M1
    Vs = Vs_cmps[4+k]
    Es = Es_cmps[4+k]
    
    M̃1 = sqrt(inv(N1)) * M1 * sqrt(inv(N1))
    Ns = real.(diag(Vs' * M̃1 * Vs))
    msk = (abs.(Ns .- Ngs_cmps) .< ΔN) .& (Es .- Egs_cmps .< ΔE * 5)

    @show Ns[msk]
    numbers_cmps = vcat(numbers_cmps, Ns[msk])

    Δphase = 1 
    for ix in Vector(1:χ^2)[msk]
        global chiral_ovlps_ψ1_cmps, antichiral_ovlps_ψ1_cmps, V0
        
        V1 = sqrt(inv(N1)) * Vs[:, ix]

        ovlpρ, ovlpj = Kac_Moody_gen(ψ, V1, k*2*pi/L, L, v_cmps, K_cmps, Ngs_cmps / L)

        # Here the phase adjustment is based on the prior knowledge from the BA solution.
        if ix == Vector(1:χ^2)[msk][1]
            if abs(k) % 2 == 1
                Δphase = norm(ovlpρ) / ovlpρ
            else
                Δphase = -norm(ovlpρ) / ovlpρ
            end

            if k == 1
                pushfirst!(chiral_ovlps_ψ1_cmps, Δphase*(ovlpρ + ovlpj) / 2)
                pushfirst!(antichiral_ovlps_ψ1_cmps, Δphase*(ovlpρ - ovlpj) / 2)
            end
        end
        
        #ovlpρ1, ovlpj1 = Kac_Moody_gen(ψ, V1, V0, k*2*pi/L, 2*pi/L, L, v_cmps, K_cmps, Ngs_cmps / L)

        ## Here the phase adjustment is based on the prior knowledge from the BA solution.
        #if ix == Vector(1:χ^2)[msk][1]
        #    if abs(k) % 2 == 1
        #        Δphase1 = -norm(ovlpρ1) / ovlpρ1
        #    else
        #        Δphase1 = norm(ovlpρ1) / ovlpρ1
        #    end
        #end

        push!(chiral_ovlps_gs_cmps, Δphase*(ovlpρ + ovlpj) / 2)
        push!(antichiral_ovlps_gs_cmps, Δphase*(ovlpρ - ovlpj) / 2)
        @show k, ix, ovlpρ, ovlpj
        #@show k, ix, ovlpρ1, ovlpj1
        #push!(chiral_ovlps_ψ1_cmps, Δphase1*(ovlpρ1 + ovlpj1) / 2)
        #push!(antichiral_ovlps_ψ1_cmps, Δphase1*(ovlpρ1 - ovlpj1) / 2)
    end
end

@save "ovlps_cmps_c$(c)_mu$(μ)_L$(L).jld2" chiral_ovlps_gs_cmps antichiral_ovlps_gs_cmps chiral_ovlps_ψ1_cmps antichiral_ovlps_ψ1_cmps 
@load "ovlps_cmps_c$(c)_mu$(μ)_L$(L).jld2" chiral_ovlps_gs_cmps antichiral_ovlps_gs_cmps chiral_ovlps_ψ1_cmps antichiral_ovlps_ψ1_cmps 
# ============ the plot ============ 

CM = maximum(norm.(vcat(chiral_ovlps_gs, antichiral_ovlps_gs)))
CM_cmps = maximum(norm.(vcat(chiral_ovlps_gs_cmps, antichiral_ovlps_gs_cmps)))

a = range(colorant"red3", colorant"grey60")
b = range(colorant"grey60", colorant"royalblue")
mycmap = vcat(a, b)
function plot_spect(ax, ψi, ovlps)
    msk = isapprox.(momenta .+ 0.1, momentum(ψi) + 0.1) .& isapprox.(energies, energy(ψi))
    msk = .~(msk)
    sc_main = scatter!(ax, momenta[msk] .* L ./ (2*pi), scale_E.(energies[msk]), color=real.(ovlps[msk]), colormap=mycmap, colorrange=(-CM, CM), marker='X', markersize=15, label="BA")
    return sc_main
end

scale_E_cmps(x) = (x - Egs_cmps) / ΔE#(minimum(Es_cmps[4+1]) - Egs_cmps)

function plot_spect_cmps(ax, ψi, ovlps)
    msk = (scale_E_cmps.(energies_cmps) .< 1e-4) .& (abs.(momenta_cmps) .< 1e-4)
    msk = .~(msk)

    msk = msk .& (abs.(numbers_cmps .- Ngs_cmps) .< ΔN)
    @show length(msk), length(momenta_cmps), length(energies_cmps), length(ovlps)

    sc_main = scatter!(ax, momenta_cmps[msk] .* L ./ (2*pi), scale_E_cmps.(energies_cmps[msk]), color=real.(ovlps[msk]), colormap=mycmap, colorrange=(-CM, CM), marker='O', markersize=15, label="cMPS")
    return sc_main
end

font1 = Makie.to_font("/home/wtang/.local/share/fonts/STIXTwoText-Regular.otf")
fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 500), fonts=(; regular=font1))
gf = fig[1, 1] = GridLayout() 
gl = fig[2, 1] = GridLayout()

ax1 = Axis(gf[1, 1], 
        xlabel = L"p L / 2\pi",
        ylabel = L"\text{rescaled } \Delta E", 
        xticks = -4:1:4,
        yticks = 0:1:4,
        )

ylims!(ax1, (-0.15, 4.5))

sc1_orig = scatter!(ax1, [0], [0], color=:darkgreen, marker=:star5, markersize=15)
sc1_main = plot_spect(ax1, ψ0, chiral_ovlps_gs)
sc1_main_cmps = plot_spect_cmps(ax1, ψ0, chiral_ovlps_gs_cmps)
@show fig

ax2 = Axis(gf[1, 2], 
        xlabel = L"p L / 2\pi",
        #ylabel = L"\text{rescaled } \Delta E", 
        xticks = -4:1:4,
        yticks = 0:1:4,
        )

ylims!(ax2, (-0.15, 4.5))

sc2_orig = scatter!(ax2, [0], [0], color=:darkgreen, marker=:star5, markersize=15)
sc2_main = plot_spect(ax2, ψ0, antichiral_ovlps_gs)
sc2_main_cmps = plot_spect_cmps(ax2, ψ0, antichiral_ovlps_gs_cmps)
@show fig

#ax3 = Axis(gf[2, 1], 
#        xlabel = L"p L / 2\pi",
#        #ylabel = L"\text{rescaled } \Delta E", 
#        xticks = -4:1:4,
#        yticks = 0:1:4,
#        )
#
#ylims!(ax3, (-0.15, 4.5))
#
##sc3_orig = scatter!(ax3, [0], [0], color=:darkgreen, marker=:star5, markersize=15)
#sc3_orig = scatter!(ax3, [1], [1], color=:darkgreen, marker=:star5, markersize=15)
#sc3_main = plot_spect(ax3, ψ1, chiral_ovlps_ψ1)
#sc3_main_cmps = plot_spect_cmps(ax3, ψ1, chiral_ovlps_ψ1_cmps)
#@show fig
#
#ax4 = Axis(gf[2, 2], 
#        xlabel = L"p L / 2\pi",
#        #ylabel = L"\text{rescaled } \Delta E", 
#        xticks = -4:1:4,
#        yticks = 0:1:4,
#        )
#
#ylims!(ax4, (-0.15, 4.5))
#sc4_orig = scatter!(ax4, [1], [1], color=:darkgreen, marker=:star5, markersize=15)
#sc4_main = plot_spect(ax4, ψ1, antichiral_ovlps_ψ1)
#sc4_main_cmps = plot_spect_cmps(ax4, ψ1, antichiral_ovlps_ψ1_cmps)
#@show fig

@show fig

for (label, layout) in zip(["(a)", "(b)"], [gf[1, 1], gf[1, 2]])
    Label(layout[1, 1, TopLeft()], label, 
    padding = (0, 5, -5, 0), 
    halign = :right
    )
end

#Label(gf[1,1][1, 1, TopLeft()], L"|\psi_\mathrm{i}\rangle",
#    padding = (0, -310, -470, 0))
#Label(gf[1,2][1, 1, TopLeft()], L"|\psi_\mathrm{i}\rangle",
#    padding = (0, -365, -370, 0))

axislegend(ax1, position=:lb, framevisible=true, labelsize=14)
axislegend(ax2, position=:lb, framevisible=true, labelsize=14)

#leg = Legend(gl[1,1], ax1, orientation=:horizontal, framecolor=:lightgrey, labelsize=16)
#leg.nbanks = 2

Colorbar(gl[1, 1], sc1_main, label=L"\text{form factor}", vertical = false, flipaxis = false) 

@show fig
Makie.save("fig-cmps-spect.pdf", fig)



msk = (scale_E.(energies) .< 2.5) 
msk_cmps = (scale_E_cmps.(energies_cmps) .< 2.5)

momenta[msk]
momenta_cmps[msk_cmps]

momenta_1 = momenta[msk] .- 1e-4 .* scale_E.(energies[msk])
order = sortperm(momenta_1)
momenta_1_cmps = momenta_cmps[msk_cmps] .- 1e-4 .* scale_E_cmps.(energies_cmps[msk_cmps])
order_cmps = sortperm(momenta_1_cmps)
momenta[msk][order]
momenta_cmps[msk_cmps][order_cmps]

scale_E.(energies)[msk][order]
scale_E_cmps.(energies_cmps)[msk_cmps][order_cmps]

chiral_ovlps_gs[msk][order]
chiral_ovlps_gs_cmps[msk_cmps][order_cmps]

open("result.txt","w") do io
   println(io, "momenta ba ", momenta[msk][order])
   println(io, "momenta cmps ", momenta_cmps[msk_cmps][order_cmps])
   println(io, "energies ba ", scale_E.(energies[msk][order]))
   println(io, "energies cmps ", scale_E_cmps.(energies_cmps[msk_cmps][order_cmps]))
   println(io, "N cmps ", real.(numbers_cmps[msk_cmps][order_cmps]))
   println(io, "chiral ovlp ba ", real.(chiral_ovlps_gs[msk][order]))
   println(io, "chiral ovlp cmps norm", norm.(chiral_ovlps_gs_cmps[msk_cmps][order_cmps]))
   println(io, "chiral ovlp cmps angle", angle.(chiral_ovlps_gs_cmps[msk_cmps][order_cmps]))
end