using LinearAlgebra, TensorKit
using CairoMakie
using JLD2, DelimitedFiles 
using Revise
using CircularCMPS

J1, J2 = 1, 0.5
T, Wmat = heisenberg_j1j2_cmpo(J1, J2)

α = 2^(1/4)
βs = 0.64 * α .^ (0:27)

free_energies, energies, variances, entropies = Float64[], Float64[], Float64[], Float64[]
for β in βs 
    @load "J1J2/cooling/dimer_phase_beta$(β).jld2" f E var
    S = (E - f) * β

    push!(free_energies, f)
    push!(energies, E)
    push!(variances, var)
    push!(entropies, S)

end

dtrg_data = readdlm("J1J2/xtrg_pbc_J1_1.000000_J2_0.500000_L_300_bondD_100.txt", '\t', Float64, '\n'; skipstart=33)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (800, 400))

ax1 = Axis(fig[1, 1], 
        xlabel = L"T",
        ylabel = L"F", 
        )
lines!(ax1, 1 ./ dtrg_data[:, 1][end-15:end], dtrg_data[:, 2][end-15:end], color=:grey, label=L"\text{XTRG}")
scatter!(ax1, 1 ./ βs[end-15:end], free_energies[end-15:end], label=L"\text{cMPO}")
#lines!(ax1, 1 ./ βs[1:end], fill(-3/8, length(βs)), color=:red, linestyle=:dash, label=L"\text{gs}")
#lines!(ax1, 1 ./ βs[1:end][end-10:end], free_energies_lowT[end-10:end], linestyle=:dash, label=L"\text{low T tensor}")
#scatter!(ax1, [0], [-3/8], marker=:star5, markersize=15, color=:red, label=L"\text{exact, gs}")
axislegend(ax1, position=:lb, framevisible=false)
@show fig

#ax2 = Axis(fig[1, 2], 
#        xlabel = L"T",
#        ylabel = L"F", 
#        )
#lines!(ax2, 1 ./ dtrg_data[:, 1][end-10:end], dtrg_data[:, 2][end-10:end], color=:grey, label=L"\text{XTRG}")
#scatter!(ax2, 1 ./ βs[end-10:end], free_energies[end-10:end], label=L"\text{cMPO, shift spect }")
#axislegend(ax2, position=:lb, framevisible=false)
#@show fig

ax2 = Axis(fig[1, 2], 
        xlabel = L"T",
        ylabel = L"S", 
        )
lines!(ax2, 1 ./ dtrg_data[:, 1][end-15:end], dtrg_data[:, 4][end-15:end], color=:grey, label=L"\text{XTRG}")
scatter!(ax2, 1 ./ βs[1:end][end-15:end], entropies[end-15:end], label=L"\text{cMPO}")
axislegend(ax2, position=:rb, framevisible=false)
@show fig
#
save("J1J2/J1J2_dimer_phase_result.pdf", fig)

vars_2to1 = Float64[]
vars_2to1_chi6 = Float64[]
vars_2to1_chi3 = Float64[]
for β in βs 
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
lines!(ax1, 1 ./ βs, variances_chi3, label=L"\text{cMPO, shift spect }, \chi=3")
lines!(ax1, 1 ./ βs, variances_chi6, label=L"\text{cMPO, shift spect }, \chi=6")
lines!(ax1, 1 ./ βs, variances, label=L"\text{cMPO, shift spect }, \chi=9")
scatter!(ax1, 1 ./ βs, vars_2to1_chi3, label=L"\text{cMPO, blk2 }, \chi=3")
scatter!(ax1, 1 ./ βs, vars_2to1_chi6, label=L"\text{cMPO, blk2 }, \chi=6")
scatter!(ax1, 1 ./ βs, vars_2to1, label=L"\text{cMPO, blk2 }, \chi=9")
#lines!(ax1, 1 ./ βs, vars_2to1_plus, linestyle=:dash, label=L"\text{cMPO, blk2, direct sum}, \chi=3")
scatter!(ax1, 1 ./ βs, vars_lowT, marker='X', label=L"\text{cMPO, shift spect low T}, \chi=3")
axislegend(ax1, position=:rt, framevisible=false)
@show fig

variances_measblk_chi3 = Float64[]
variances_measblk_chi6 = Float64[]
variances_measblk_chi9 = Float64[]
for β in βs 
    @load "J1J2/dimer_phase_beta$(β).jld2" fs Es vars ψs

    var3 = variance(T2, ψs[end-2], β)
    push!(variances_measblk_chi3, var3)
    var6 = variance(T2, ψs[end-1], β)
    push!(variances_measblk_chi6, var6)
    var9 = variance(T2, ψs[end], β)
    push!(variances_measblk_chi9, var9)
end

ax2 = Axis(fig[2, 1], 
        xlabel = L"T",
        ylabel = L"\text{variance of } T^2",
        yscale = log10, 
        )
lines!(ax2, 1 ./ βs, variances_blk_chi3, label=L"\text{cMPO, blk }, \chi=3")
lines!(ax2, 1 ./ βs, variances_blk_chi6, label=L"\text{cMPO, blk }, \chi=6")
lines!(ax2, 1 ./ βs, variances_blk, label=L"\text{cMPO, blk }, \chi=9")
scatter!(ax2, 1 ./ βs, variances_measblk_chi3, label=L"\text{cMPO, meas blk }, \chi=3")
scatter!(ax2, 1 ./ βs, variances_measblk_chi6, label=L"\text{cMPO, meas blk }, \chi=6")
scatter!(ax2, 1 ./ βs, variances_measblk_chi9, label=L"\text{cMPO, meas blk }, \chi=9")
axislegend(ax2, position=:rt, framevisible=false)
@show fig

save("J1J2/J1J2_dimer_phase_variances.pdf", fig)