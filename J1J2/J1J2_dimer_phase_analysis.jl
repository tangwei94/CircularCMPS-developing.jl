using LinearAlgebra, TensorKit
using CairoMakie
using JLD2, DelimitedFiles 
using Revise
using CircularCMPS

J1, J2 = 1, 0.5
T, Wmat = heisenberg_j1j2_cmpo(J1, J2)
χs = [3, 6, 9]

α = 2^(1/4)
βs = 1.28 * α .^ (0:23)

free_energies_6 = Float64[]
energies_6 = Float64[] 
variances_6 = Float64[]
entropies_6 = Float64[]
for β in βs[1:end-1] 
    @load "J1J2/dimer_phase_beta$(β).jld2" fs Es vars
    S = (Es[end] - fs[end]) * β

    push!(free_energies_6, fs[end])
    push!(energies_6, Es[end])
    push!(variances_6, Es[end])
    push!(entropies_6, S)
end

@show energies_6
@show free_energies_6

free_energies_blk = Float64[] 
energies_blk = Float64[] 
variances_blk = Float64[] 
entropies_blk = Float64[]
for β in βs[1:12] 
    @load "J1J2/dimer_phase_blk2_beta$(β).jld2" fs Es vars
    S = (Es[end] - fs[end]) * β

    push!(free_energies_blk, fs[end])
    push!(energies_blk, Es[end])
    push!(variances_blk, Es[end])
    push!(entropies_blk, S)
end

dtrg_data = readdlm("J1J2/xtrg_pbc_J1_1.000000_J2_0.241167_L_300_bondD_100.txt", '\t', Float64, '\n'; skipstart=33)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))

ax1 = Axis(fig[1, 1], 
        xlabel = L"T",
        ylabel = L"F", 
        )
lines!(ax1, 1 ./ dtrg_data[:, 1], dtrg_data[:, 2], color=:grey, label=L"\text{XTRG}")
scatter!(ax1, 1 ./ βs[1:end-1], free_energies_6, label=L"\text{cMPO, shift spect }")
scatter!(ax1, 1 ./ βs[1:12], free_energies_blk, marker='X', color=:red, label=L"\text{cMPO, blk}")
scatter!(ax1, [0], [-3/8], marker=:star5, markersize=15, color=:red, label=L"\text{exact, gs}")
axislegend(ax1, position=:rb, framevisible=false)
@show fig

ax2 = Axis(fig[2, 1], 
        xlabel = L"T",
        ylabel = L"S", 
        )
lines!(ax2, 1 ./ dtrg_data[:, 1], dtrg_data[:, 4], color=:grey, label=L"\text{XTRG}")
scatter!(ax2, 1 ./ βs[1:end-1], entropies_6, label=L"\text{cMPO, shift spect}")
scatter!(ax2, 1 ./ βs[1:12], entropies_blk, marker='X', color=:red, label=L"\text{cMPO, blk}")
axislegend(ax2, position=:rb, framevisible=false)
@show fig


save("J1J2/J1J2_dimer_phase_result.pdf", fig)


