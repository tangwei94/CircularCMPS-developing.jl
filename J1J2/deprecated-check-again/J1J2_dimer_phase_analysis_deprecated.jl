using LinearAlgebra, TensorKit
using CairoMakie
using JLD2, DelimitedFiles 
using Revise
using CircularCMPS

J1, J2 = 1, 0.5
T, Wmat = CircularCMPS.heisenberg_j1j2_cmpo_deprecated(J1, J2)

α = 2^(1/4)
βs = 1.28 * α .^ (0:23)
βs = βs[end-10:end]

function read_data(βs, prefix, ix)
    free_energies, energies, variances, entropies = Float64[], Float64[], Float64[], Float64[]
    for β in βs
        @load "J1J2/deprecated-check-again/data/$(prefix)$(β).jld2" fs Es vars ψs
        S = (Es[end] - fs[end]) * β

        (β ≈ βs[1]) && (@show space(ψs[end-ix].Q))
        push!(free_energies, fs[end-ix])
        push!(energies, Es[end-ix])
        push!(variances, vars[end-ix])
        push!(entropies, S)
    end
    return free_energies, energies, variances, entropies
end

free_energies_3, enegies_3, variances_3, entropies_3 = read_data(βs, "dimer_phase_beta", 3);
free_energies_9, enegies_9, variances_9, entropies_9 = read_data(βs, "dimer_phase_beta", 1);
free_energies_12, enegies_12, variances_12, entropies_12 = read_data(βs, "dimer_phase_beta", 0);

dtrg_data = readdlm("J1J2/xtrg_pbc_J1_1.000000_J2_0.500000_L_300_bondD_100.txt", '\t', Float64, '\n'; skipstart=33)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 1200))

ax1 = Axis(fig[1, 1], 
        xlabel = L"T",
        ylabel = L"F", 
        )
lines!(ax1, 1 ./ dtrg_data[:, 1][end-11:end], dtrg_data[:, 2][end-11:end], color=:grey, label=L"\text{XTRG}")
scatter!(ax1, 1 ./ βs, free_energies_3, marker='x', label=L"\text{cMPO, }χ=3")
scatter!(ax1, 1 ./ βs, free_energies_9, marker='x', label=L"\text{cMPO, }χ=9")
scatter!(ax1, 1 ./ βs, free_energies_12, marker='x', label=L"\text{cMPO, }χ=12")
axislegend(ax1, position=:lb, framevisible=false)
@show fig

ax2 = Axis(fig[2, 1], 
        xlabel = L"T",
        ylabel = L"S", 
        )
lines!(ax2, 1 ./ dtrg_data[:, 1][end-11:end], dtrg_data[:, 4][end-11:end], color=:grey, label=L"\text{XTRG}")
scatter!(ax2, 1 ./ βs[1:end], entropies_3, label=L"\text{cMPO, }χ=3")
scatter!(ax2, 1 ./ βs[1:end], entropies_9, label=L"\text{cMPO, }χ=9")
scatter!(ax2, 1 ./ βs[1:end], entropies_12, label=L"\text{cMPO, }χ=12")
axislegend(ax2, position=:rb, framevisible=false)
@show fig

ax3 = Axis(fig[3, 1], 
        xlabel = L"T",
        ylabel = L"\text{variance}", 
        yscale = log10,
        )
scatter!(ax3, 1 ./ βs[1:end], variances_3, label=L"\text{cMPO, }χ=3")
scatter!(ax3, 1 ./ βs[1:end], variances_9, label=L"\text{cMPO, }χ=9")
scatter!(ax3, 1 ./ βs[1:end], variances_12, label=L"\text{cMPO, }χ=12")
axislegend(ax3, position=:rt, framevisible=false)
@show fig

save("J1J2/deprecated-check-again/J1J2_dimer_phase_result_deprecated.pdf", fig)
