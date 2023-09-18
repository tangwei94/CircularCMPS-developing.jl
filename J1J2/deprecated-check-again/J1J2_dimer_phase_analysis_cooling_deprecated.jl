using LinearAlgebra, TensorKit
using CairoMakie
using JLD2, DelimitedFiles 
using Revise
using CircularCMPS

J1, J2 = 1, 0.5
T, Wmat = heisenberg_j1j2_cmpo(J1, J2)

α = 2^(1/4)
βs = 0.64 * α .^ (0:27)

function read_data(βs)
    free_energies, energies, variances, entropies, EEs, EEsyss = Float64[], Float64[], Float64[], Float64[], Float64[], Float64[]
    chis = Int[]
    for β in βs
        @load "J1J2/deprecated-check-again/data/dimer_phase_beta$(β)_cooling.jld2" f E var ψ
        ψL = Wmat * ψ
        S = (E - f) * β
        χ = dim(space(ψ))
        SE = entanglement_entropy(ψ, β)
        SE_sys = entanglement_entropy(ψL, ψ, β)

        push!(free_energies, f)
        push!(energies, E)
        push!(variances, var)
        push!(entropies, S)
        push!(chis, χ)
        push!(EEs, SE)
        push!(EEsyss, SE_sys)
    end
    return free_energies, energies, variances, entropies, chis, EEs, EEsyss
end

βs = βs[1:end-3]
free_energies, enegies, variances, entropies, chis, EEs, EEsyss = read_data(βs);


dtrg_data = readdlm("J1J2/xtrg_pbc_J1_1.000000_J2_0.500000_L_300_bondD_100.txt", '\t', Float64, '\n'; skipstart=33)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 1500))

ax1 = Axis(fig[1, 1], 
        xlabel = L"T",
        ylabel = L"F", 
        )
lines!(ax1, 1 ./ dtrg_data[:, 1][end-11:end], dtrg_data[:, 2][end-11:end], color=:grey, label=L"\text{XTRG}")
scatter!(ax1, 1 ./ βs[end-8:end], free_energies[end-8:end], label=L"\text{cMPO}")
axislegend(ax1, position=:lb, framevisible=false)
@show fig

ax2 = Axis(fig[2, 1], 
        xlabel = L"T",
        ylabel = L"S", 
        )
lines!(ax2, 1 ./ dtrg_data[:, 1][end-11:end], dtrg_data[:, 4][end-11:end], color=:grey, label=L"\text{XTRG}")
scatter!(ax2, 1 ./ βs[end-8:end], entropies[end-8:end], label=L"\text{cMPO}")
axislegend(ax2, position=:rb, framevisible=false)
@show fig
#
ax3 = Axis(fig[3, 1], 
        xlabel = L"T",
        ylabel = L"\text{variance}", 
        yscale = log10,
        )
scatter!(ax3, 1 ./ βs[end-8:end], variances[end-8:end], label=L"\text{cMPO}")
axislegend(ax3, position=:rt, framevisible=false)
@show fig

ax4 = Axis(fig[4, 1], 
        xlabel = L"T",
        ylabel = L"χ", 
        )
scatter!(ax4, 1 ./ βs[end-8:end], chis[end-8:end], label=L"\text{cMPO}")
axislegend(ax4, position=:rt, framevisible=false)
@show fig

ax5 = Axis(fig[5, 1], 
        xlabel = L"T",
        ylabel = L"\text{single-sided } S_E", 
        )
scatter!(ax5, 1 ./ βs[end-8:end], EEs[end-8:end], label=L"\text{cMPO}")
axislegend(ax5, position=:rb, framevisible=false)
@show fig

ax6 = Axis(fig[6, 1], 
        xlabel = L"T",
        ylabel = L"\text{double-sided } S_E", 
        )
scatter!(ax6, 1 ./ βs[end-8:end], EEsyss[end-8:end], label=L"\text{cMPO}")
axislegend(ax5, position=:rb, framevisible=false)
@show fig

save("J1J2/deprecated-check-again/J1J2_dimer_phase_result_cooling_deprecated.pdf", fig)
