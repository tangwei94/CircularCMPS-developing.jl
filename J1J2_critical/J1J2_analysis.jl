using LinearAlgebra, TensorKit
using CairoMakie
using JLD2, DelimitedFiles 
using Revise
using CircularCMPS

J1, J2 = 1, 0.241167
T, Wmat = heisenberg_j1j2_cmpo(J1, J2)

α = 2^(1/4)
βs = 0.64 * α .^ (0:27)

function read_data(βs, prefix)
    free_energies, energies, variances, entropies, EEs, EEsyss = Float64[], Float64[], Float64[], Float64[], Float64[], Float64[]
    chis = Int[]
    for β in βs
        @load "J1J2_critical/data/$(prefix)$(β).jld2" f E var ψ
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

free_energies_7, enegies_7, variances_7, entropies_7, chis_7, EEs_7, EEsyss_7 = read_data(βs, "beta");
free_energies_8, enegies_8, variances_8, entropies_8, chis_8, EEs_8, EEsyss_8 = read_data(βs, "tolES8_beta");

dtrg_data = readdlm("J1J2_critical/xtrg_pbc_J1_1.000000_J2_0.241167_L_300_bondD_100.txt", '\t', Float64, '\n'; skipstart=33)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 1500))

ax1 = Axis(fig[1, 1], 
        xlabel = L"T",
        ylabel = L"F", 
        )
lines!(ax1, 1 ./ dtrg_data[:, 1][end-11:end], dtrg_data[:, 2][end-11:end], color=:grey, label=L"\text{XTRG}")
scatter!(ax1, 1 ./ βs[end-11:end], free_energies_7[end-11:end], marker='x', label=L"\text{cMPO, tolES=1e-7}")
scatter!(ax1, 1 ./ βs[end-11:end], free_energies_8[end-11:end], marker='x', label=L"\text{cMPO, tolES=1e-8}")
axislegend(ax1, position=:lb, framevisible=false)
@show fig

ax2 = Axis(fig[2, 1], 
        xlabel = L"T",
        ylabel = L"S", 
        )
lines!(ax2, 1 ./ dtrg_data[:, 1][end-11:end], dtrg_data[:, 4][end-11:end], color=:grey, label=L"\text{XTRG}")
scatter!(ax2, 1 ./ βs[1:end][end-11:end], entropies_7[end-11:end], marker='x', label=L"\text{cMPO, tolES=1e-7}")
scatter!(ax2, 1 ./ βs[1:end][end-11:end], entropies_8[end-11:end], marker='x', label=L"\text{cMPO, tolES=1e-8}")
axislegend(ax2, position=:rb, framevisible=false)
@show fig
#
ax3 = Axis(fig[3, 1], 
        xlabel = L"T",
        ylabel = L"\text{variance}", 
        yscale = log10,
        )
scatter!(ax3, 1 ./ βs[1:end][end-11:end], variances_7[end-11:end], marker='x', label=L"\text{cMPO, tolES=1e-7}")
scatter!(ax3, 1 ./ βs[1:end][end-11:end], variances_8[end-11:end], marker='x', label=L"\text{cMPO, tolES=1e-8}")
axislegend(ax3, position=:rt, framevisible=false)
@show fig

ax4 = Axis(fig[4, 1], 
        xlabel = L"T",
        ylabel = L"χ", 
        )
scatter!(ax4, 1 ./ βs[1:end][end-11:end], chis_7[end-11:end], marker='x', label=L"\text{cMPO, tolES=1e-7}")
scatter!(ax4, 1 ./ βs[1:end][end-11:end], chis_8[end-11:end], marker='x', label=L"\text{cMPO, tolES=1e-8}")
axislegend(ax4, position=:rt, framevisible=false)
@show fig

ax5 = Axis(fig[5, 1], 
        xlabel = L"T",
        ylabel = L"\text{single-sided } S_E", 
        )
scatter!(ax5, 1 ./ βs[1:end][end-11:end], EEs_7[end-11:end], marker='x', label=L"\text{cMPO, tolES=1e-7}")
scatter!(ax5, 1 ./ βs[1:end][end-11:end], EEs_8[end-11:end], marker='x', label=L"\text{cMPO, tolES=1e-8}")
axislegend(ax5, position=:rb, framevisible=false)
@show fig

ax6 = Axis(fig[6, 1], 
        xlabel = L"T",
        ylabel = L"\text{double-sided } S_E", 
        )
scatter!(ax6, 1 ./ βs[1:end][end-11:end], EEsyss_7[end-11:end], marker='x', label=L"\text{cMPO, tolES=1e-7}")
scatter!(ax6, 1 ./ βs[1:end][end-11:end], EEsyss_8[end-11:end], marker='x', label=L"\text{cMPO, tolES=1e-8}")
axislegend(ax6, position=:rb, framevisible=false)
@show fig

save("J1J2_critical/J1J2_critical_result_tolES8.pdf", fig)
#save("J1J2_critical/J1J2_critical_result_deprecated.pdf", fig)
