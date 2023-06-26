using LinearAlgebra, TensorKit
using CairoMakie
using JLD2, DelimitedFiles 
using Revise
using CircularCMPS

J1, J2 = 1, 0.5
T, Wmat = heisenberg_j1j2_cmpo(J1, J2)

α = 2^(1/4)
βs = 1.28 * α .^ (0:23)
βs = βs[end-10:end]

function read_data(βs, prefix, ix)
    free_energies, energies, variances, entropies = Float64[], Float64[], Float64[], Float64[]
    for β in βs
        @load "J1J2/old_results1/$(prefix)$(β).jld2" fs Es vars
        S = (Es[end] - fs[end]) * β

        push!(free_energies, fs[end-ix])
        push!(energies, Es[end-ix])
        push!(variances, vars[end-ix])
        push!(entropies, S)
    end
    return free_energies, energies, variances, entropies
end

free_energies, enegies, variances, entropies = read_data(βs, "dimer_phase_beta", 2);
free_energies2, enegies2, variances2, entropies2 = read_data(βs, "dimer_phase_beta", 0);

dtrg_data = readdlm("J1J2/xtrg_pbc_J1_1.000000_J2_0.500000_L_300_bondD_100.txt", '\t', Float64, '\n'; skipstart=33)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 1200))

ax1 = Axis(fig[1, 1], 
        xlabel = L"T",
        ylabel = L"F", 
        )
lines!(ax1, 1 ./ dtrg_data[:, 1][end-11:end], dtrg_data[:, 2][end-11:end], color=:grey, label=L"\text{XTRG}")
scatter!(ax1, 1 ./ βs, free_energies, label=L"\text{cMPO, }χ=3")
scatter!(ax1, 1 ./ βs, free_energies2, label=L"\text{cMPO, }χ=9")
#lines!(ax1, 1 ./ βs[1:end], fill(-3/8, length(βs)), color=:red, linestyle=:dash, label=L"\text{gs}")
#lines!(ax1, 1 ./ βs[1:end][end-10:end], free_energies_lowT[end-10:end], linestyle=:dash, label=L"\text{low T tensor}")
#scatter!(ax1, [0], [-3/8], marker=:star5, markersize=15, color=:red, label=L"\text{exact, gs}")
axislegend(ax1, position=:lb, framevisible=false)
@show fig

ax2 = Axis(fig[2, 1], 
        xlabel = L"T",
        ylabel = L"S", 
        )
lines!(ax2, 1 ./ dtrg_data[:, 1][end-11:end], dtrg_data[:, 4][end-11:end], color=:grey, label=L"\text{XTRG}")
scatter!(ax2, 1 ./ βs[1:end], entropies, label=L"\text{cMPO, }χ=3")
scatter!(ax2, 1 ./ βs[1:end], entropies2, label=L"\text{cMPO, }χ=9")
axislegend(ax2, position=:rb, framevisible=false)
@show fig
#
ax3 = Axis(fig[3, 1], 
        xlabel = L"T",
        ylabel = L"\text{variance}", 
        yscale = log10,
        )
scatter!(ax3, 1 ./ βs[1:end], variances, label=L"\text{cMPO, }χ=3")
scatter!(ax3, 1 ./ βs[1:end], variances2, label=L"\text{cMPO, }χ=9")
axislegend(ax3, position=:rt, framevisible=false)
@show fig

save("J1J2/J1J2_dimer_phase_result_gauged_results.pdf", fig)
