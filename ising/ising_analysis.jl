using LinearAlgebra, TensorKit
using CairoMakie
using JLD2, DelimitedFiles 
using Revise
using CircularCMPS

Γ = 1.0
T, Wmat = ising_cmpo(Γ)
ψ = CMPSData(T.Q, T.Ls)

α = 2^(1/2)
βs = 0.32 * α .^ (0:14)
tol = 9

function read_data(βs, tol)
    free_energies, energies, variances, entropies, EEs, EEsyss = Float64[], Float64[], Float64[], Float64[], Float64[], Float64[]
    chis = Int[]
    for β in βs
        @load "ising/results/ising_Gamma1.0_beta$(β)-toles$(tol).jld2" f E var ψ
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

free_energies, enegies, variances, entropies, chis, EEs, EEsyss = read_data(βs, tol);

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 1500))

ax1 = Axis(fig[1, 1], 
        xlabel = L"T",
        ylabel = L"F", 
        )
scatter!(ax1, 1 ./ βs, free_energies, marker='x', label="tolES=1e-$(tol)")
axislegend(ax1, position=:lb, framevisible=false)
@show fig

ax2 = Axis(fig[2, 1], 
        xlabel = L"T",
        ylabel = L"S", 
        )
scatter!(ax2, 1 ./ βs, entropies, marker='x', label="tolES=1e-$(tol)")
axislegend(ax2, position=:rb, framevisible=false)
@show fig
#
ax3 = Axis(fig[3, 1], 
        xlabel = L"T",
        ylabel = L"\text{variance}", 
        yscale = log10,
        )
scatter!(ax3, 1 ./ βs, variances, marker='x', label="tolES=1e-$(tol)")
axislegend(ax3, position=:rt, framevisible=false)
@show fig

ax4 = Axis(fig[4, 1], 
        xlabel = L"T",
        ylabel = L"χ", 
        )
scatter!(ax4, 1 ./ βs, chis, marker='x', label="tolES=1e-$(tol)")
axislegend(ax4, position=:rt, framevisible=false)
@show fig

ax5 = Axis(fig[5, 1], 
        xlabel = L"T",
        ylabel = L"\text{single-sided } S_E", 
        )
scatter!(ax5, 1 ./ βs, EEs, marker='x', label="tolES=1e-$(tol)")
axislegend(ax5, position=:rb, framevisible=false)
@show fig

ax6 = Axis(fig[6, 1], 
        xlabel = L"T",
        ylabel = L"\text{double-sided } S_E", 
        )
scatter!(ax6, 1 ./ βs, EEsyss, marker='x', label="tolES=1e-$(tol)")
axislegend(ax6, position=:rb, framevisible=false)
@show fig

save("ising/results/ising_result_tolES$(tol).pdf", fig)
