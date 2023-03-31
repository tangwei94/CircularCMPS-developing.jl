using LinearAlgebra, TensorKit
using CairoMakie
using JLD2, DelimitedFiles 
using Revise
using CircularCMPS

J1, J2 = 1, 0.5
T, Wmat = heisenberg_j1j2_cmpo(J1, J2)
χs = [3, 6]

α = 2^(1/4)
βs = 1.28 * α .^ (0:23)

free_energies_3 = Float64[]
energies_3 = Float64[] 
variances_3 = Float64[]
entropies_3 = Float64[]
free_energies_6 = Float64[]
energies_6 = Float64[] 
variances_6 = Float64[]
entropies_6 = Float64[]
for β in βs 
    @load "J1J2/dimer_phase_beta$(β).jld2" fs Es vars
    S = (Es[end-1] - fs[end-1]) * β

    push!(free_energies_3, fs[end-1])
    push!(energies_3, Es[end-1])
    push!(variances_3, Es[end-1])
    push!(entropies_3, S)

    S = (Es[end] - fs[end]) * β

    push!(free_energies_6, fs[end])
    push!(energies_6, Es[end])
    push!(variances_6, Es[end])
    push!(entropies_6, S)
end

dtrg_data = readdlm("J1J2/xtrg_pbc_J1_1.000000_J2_0.241167_L_300_bondD_100.txt", '\t', Float64, '\n'; skipstart=33)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))

ax1 = Axis(fig[1, 1], 
        xlabel = L"T",
        ylabel = L"F", 
        )
lines!(ax1, 1 ./ dtrg_data[:, 1], dtrg_data[:, 2], label=L"\text{XTRG}")
scatter!(ax1, 1 ./ βs, free_energies_3, label=L"\text{cMPO, }\chi=3")
scatter!(ax1, 1 ./ βs, free_energies_6, label=L"\text{cMPO, }\chi=6")
axislegend(ax1, position=:rb, framevisible=false)
@show fig

ax2 = Axis(fig[2, 1], 
        xlabel = L"T",
        ylabel = L"S", 
        )
lines!(ax2, 1 ./ dtrg_data[:, 1], dtrg_data[:, 4], label=L"\text{XTRG}")
scatter!(ax2, 1 ./ βs, entropies_3, label=L"\text{cMPO, }\chi=3")
scatter!(ax2, 1 ./ βs, entropies_6, label=L"\text{cMPO, }\chi=6")
axislegend(ax2, position=:rb, framevisible=false)
@show fig





