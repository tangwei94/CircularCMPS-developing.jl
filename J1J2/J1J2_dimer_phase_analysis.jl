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

free_energies, energies, variances, entropies = Float64[], Float64[], Float64[], Float64[]
free_energies_chi6, energies_chi6, variances_chi6, entropies_chi6 = Float64[], Float64[], Float64[], Float64[]
free_energies_chi3, energies_chi3, variances_chi3, entropies_chi3 = Float64[], Float64[], Float64[], Float64[]
for β in βs[1:end] 
    @load "J1J2/dimer_phase_beta$(β).jld2" fs Es vars
    S = (Es[end] - fs[end]) * β

    push!(free_energies, fs[end])
    push!(energies, Es[end])
    push!(variances, vars[end])
    push!(entropies, S)

    S = (Es[end-1] - fs[end-1]) * β

    push!(free_energies_chi6, fs[end-1])
    push!(energies_chi6, Es[end-1])
    push!(variances_chi6, vars[end-1])
    push!(entropies_chi6, S)

    S = (Es[end-2] - fs[end-2]) * β

    push!(free_energies_chi3, fs[end-2])
    push!(energies_chi3, Es[end-2])
    push!(variances_chi3, vars[end-2])
    push!(entropies_chi3, S)
end

free_energies_blk, energies_blk, variances_blk, entropies_blk = Float64[], Float64[], Float64[], Float64[]
free_energies_blk_chi6, energies_blk_chi6, variances_blk_chi6, entropies_blk_chi6 = Float64[], Float64[], Float64[], Float64[]
free_energies_blk_chi3, energies_blk_chi3, variances_blk_chi3, entropies_blk_chi3 = Float64[], Float64[], Float64[], Float64[]
for β in βs[1:18] 
    @load "J1J2/dimer_phase_blk2_beta$(β).jld2" fs Es vars
    S = (Es[end] - fs[end]) * β

    push!(free_energies_blk, fs[end])
    push!(energies_blk, Es[end])
    push!(variances_blk, vars[end])
    push!(entropies_blk, S)

    S = (Es[end-1] - fs[end-1]) * β

    push!(free_energies_blk_chi6, fs[end-1])
    push!(energies_blk_chi6, Es[end-1])
    push!(variances_blk_chi6, vars[end-1])
    push!(entropies_blk_chi6, S)

    S = (Es[end-2] - fs[end-2]) * β

    push!(free_energies_blk_chi3, fs[end-2])
    push!(energies_blk_chi3, Es[end-2])
    push!(variances_blk_chi3, vars[end-2])
    push!(entropies_blk_chi3, S)
end

β0 = 1.28 * α .^ 23
@load "J1J2/dimer_phase_beta$(β0).jld2" ψs
ϕ = ψs[end-2]
ϕL = W_mul(Wmat, ϕ)
vars_lowT = Float64[] 
free_energies_lowT = Float64[] 
entropies_lowT = Float64[] 
for β in βs[1:end]
    push!(vars_lowT, variance(T, ϕ, β))
    f = free_energy(T, ϕL, ϕ, β)
    E = energy(T, ϕL, ϕ, β)
    push!(free_energies_lowT, f)
    push!(entropies_lowT, -(f-E)*β)
end

dtrg_data = readdlm("J1J2/xtrg_pbc_J1_1.000000_J2_0.500000_L_300_bondD_100.txt", '\t', Float64, '\n'; skipstart=33)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))

ax1 = Axis(fig[1, 1], 
        xlabel = L"T",
        ylabel = L"F", 
        )
lines!(ax1, 1 ./ dtrg_data[:, 1][end-10:end], dtrg_data[:, 2][end-10:end], color=:grey, label=L"\text{XTRG}")
scatter!(ax1, 1 ./ βs[1:end][end-10:end], free_energies_chi3[end-10:end], label=L"\text{cMPO, shift spect }")
scatter!(ax1, 1 ./ βs[1:18][end-5:end], free_energies_blk[end-5:end], marker='X', color=:red, label=L"\text{cMPO, blk}")
#lines!(ax1, 1 ./ βs[1:end], fill(-3/8, length(βs)), color=:red, linestyle=:dash, label=L"\text{gs}")
#lines!(ax1, 1 ./ βs[1:end][end-10:end], free_energies_lowT[end-10:end], linestyle=:dash, label=L"\text{low T tensor}")
#scatter!(ax1, [0], [-3/8], marker=:star5, markersize=15, color=:red, label=L"\text{exact, gs}")
axislegend(ax1, position=:rb, framevisible=false)
@show fig

ax2 = Axis(fig[2, 1], 
        xlabel = L"T",
        ylabel = L"S", 
        )
lines!(ax2, 1 ./ dtrg_data[:, 1], dtrg_data[:, 4], color=:grey, label=L"\text{XTRG}")
scatter!(ax2, 1 ./ βs[1:end], entropies, label=L"\text{cMPO, shift spect}")
scatter!(ax2, 1 ./ βs[1:18], entropies_blk, marker='X', color=:red, label=L"\text{cMPO, blk}")
lines!(ax2, 1 ./ βs[1:end], entropies_lowT, linestyle=:dash, label=L"\text{low T tensor}")
axislegend(ax2, position=:rb, framevisible=false)
@show fig

save("J1J2/J1J2_dimer_phase_result.pdf", fig)

vars_2to1 = Float64[]
vars_2to1_chi6 = Float64[]
vars_2to1_chi3 = Float64[]
for β in βs[1:18] 
    @load "J1J2/dimer_phase_blk2_beta$(β).jld2" ψs

    push!(vars_2to1, variance(T, ψs[end], β))
    push!(vars_2to1_chi6, variance(T, ψs[end-1], β))
    push!(vars_2to1_chi3, variance(T, ψs[end-2], β))
end

vars_2to1_plus = Float64[] 
for β in βs[1:18]
    @load "J1J2/dimer_phase_blk2_beta$(β).jld2" fs ψs
    ϕ = ψs[end-2]
    f = fs[end-2]
    
    ϕp = direct_sum(ϕ, T*ϕ; α=f)
    push!(vars_2to1_plus, variance(T, ϕp, β))
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))
ax1 = Axis(fig[1, 1], 
        xlabel = L"T",
        ylabel = L"\text{variance of } T",
        yscale = log10, 
        )
lines!(ax1, 1 ./ βs[1:end], variances_chi3, label=L"\text{cMPO, shift spect }, \chi=3")
lines!(ax1, 1 ./ βs[1:end], variances_chi6, label=L"\text{cMPO, shift spect }, \chi=6")
lines!(ax1, 1 ./ βs[1:end], variances, label=L"\text{cMPO, shift spect }, \chi=9")
scatter!(ax1, 1 ./ βs[1:18], vars_2to1_chi3, label=L"\text{cMPO, blk2 }, \chi=3")
scatter!(ax1, 1 ./ βs[1:18], vars_2to1_chi6, label=L"\text{cMPO, blk2 }, \chi=6")
scatter!(ax1, 1 ./ βs[1:18], vars_2to1, label=L"\text{cMPO, blk2 }, \chi=9")
lines!(ax1, 1 ./ βs[1:18], vars_2to1_plus, linestyle=:dash, label=L"\text{cMPO, blk2, direct sum}, \chi=3")
scatter!(ax1, 1 ./ βs[1:end], vars_lowT, marker='X', label=L"\text{cMPO, shift spect low T}, \chi=3")
axislegend(ax1, position=:rt, framevisible=false)
@show fig

ax2 = Axis(fig[2, 1], 
        xlabel = L"T",
        ylabel = L"\text{variance of } T^2",
        yscale = log10, 
        )
lines!(ax2, 1 ./ βs[1:18], variances_blk_chi3, label=L"\text{cMPO, blk }, \chi=3")
lines!(ax2, 1 ./ βs[1:18], variances_blk_chi6, label=L"\text{cMPO, blk }, \chi=6")
lines!(ax2, 1 ./ βs[1:18], variances_blk, label=L"\text{cMPO, blk }, \chi=9")
axislegend(ax2, position=:rt, framevisible=false)
@show fig

save("J1J2/J1J2_dimer_phase_variances.pdf", fig)