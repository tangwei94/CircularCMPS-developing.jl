using LinearAlgebra, TensorKit
using CairoMakie
using JLD2, DelimitedFiles 
using Revise
using CircularCMPS

J1, J2 = 1, 0.5
T, Wmat = heisenberg_j1j2_cmpo(J1, J2)

α = 2^(1/4)
βs = 0.64 * α .^ (0:27)

function read_data(βs, prefix)
    free_energies, energies, variances, entropies, EEs = Float64[], Float64[], Float64[], Float64[], Float64[]
    chis = Int[]
    for β in βs
        @load "J1J2/cooling1/$(prefix)$(β).jld2" f E var ψ
        S = (E - f) * β
        χ = dim(space(ψ))
        SE = entanglement_entropy(ψ, β)

        push!(free_energies, f)
        push!(energies, E)
        push!(variances, var)
        push!(entropies, S)
        push!(chis, χ)
        push!(EEs, SE)
    end
    return free_energies, energies, variances, entropies, chis, EEs
end

free_energies, enegies, variances, entropies, chis, EEs = read_data(βs, "dimer_phase_beta");
#free_energies2, enegies2, variances2, entropies2, chis2, EEs2 = read_data(βs, "dimer_phase_beta-opt2");

dtrg_data = readdlm("J1J2/xtrg_pbc_J1_1.000000_J2_0.500000_L_300_bondD_100.txt", '\t', Float64, '\n'; skipstart=33)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 1200))

ax1 = Axis(fig[1, 1], 
        xlabel = L"T",
        ylabel = L"F", 
        )
lines!(ax1, 1 ./ dtrg_data[:, 1][end-11:end], dtrg_data[:, 2][end-11:end], color=:grey, label=L"\text{XTRG}")
scatter!(ax1, 1 ./ βs[end-11:end], free_energies[end-11:end], label=L"\text{cMPO, tol=}10^{-6}")
#scatter!(ax1, 1 ./ βs[end-11:end], free_energies2[end-11:end], label=L"\text{cMPO, tol=}10^{-8}")
axislegend(ax1, position=:lb, framevisible=false)
@show fig

ax2 = Axis(fig[2, 1], 
        xlabel = L"T",
        ylabel = L"S", 
        )
lines!(ax2, 1 ./ dtrg_data[:, 1][end-11:end], dtrg_data[:, 4][end-11:end], color=:grey, label=L"\text{XTRG}")
scatter!(ax2, 1 ./ βs[1:end][end-11:end], entropies[end-11:end], label=L"\text{cMPO, tol=}10^{-6}")
#scatter!(ax2, 1 ./ βs[1:end][end-11:end], entropies2[end-11:end], label=L"\text{cMPO, tol=}10^{-8}")
axislegend(ax2, position=:rb, framevisible=false)
@show fig
#
ax3 = Axis(fig[3, 1], 
        xlabel = L"T",
        ylabel = L"\text{variance}", 
        yscale = log10,
        )
#lines!(ax2, 1 ./ dtrg_data[:, 1][end-11:end], dtrg_data[:, 4][end-11:end], color=:grey, label=L"\text{XTRG}")
scatter!(ax3, 1 ./ βs[1:end][end-11:end], variances[end-11:end], label=L"\text{cMPO, tol=}10^{-6}")
#scatter!(ax3, 1 ./ βs[1:end][end-11:end], variances2[end-11:end], label=L"\text{cMPO, tol=}10^{-8}")
axislegend(ax3, position=:rt, framevisible=false)
@show fig

ax4 = Axis(fig[4, 1], 
        xlabel = L"T",
        ylabel = L"χ", 
        )
scatter!(ax4, 1 ./ βs[1:end][end-11:end], chis[end-11:end], label=L"\text{cMPO, tol=}10^{-6}")
#scatter!(ax4, 1 ./ βs[1:end][end-11:end], chis2[end-11:end], label=L"\text{cMPO, tol=}10^{-8}")
axislegend(ax4, position=:rt, framevisible=false)
@show fig

ax5 = Axis(fig[5, 1], 
        xlabel = L"T",
        ylabel = L"S_E", 
        )
scatter!(ax5, 1 ./ βs[1:end][end-11:end], EEs[end-11:end], label=L"\text{cMPO, tol=}10^{-6}")
#scatter!(ax5, 1 ./ βs[1:end][end-11:end], EEs2[end-11:end], label=L"\text{cMPO, tol=}10^{-8}")
axislegend(ax5, position=:rb, framevisible=false)
@show fig

save("J1J2/J1J2_dimer_phase_result_cooling1.pdf", fig)
