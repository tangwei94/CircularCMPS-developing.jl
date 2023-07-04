using LinearAlgebra, TensorKit
using CairoMakie
using JLD2, DelimitedFiles 
using Revise
using CircularCMPS

J1, J2 = 1, 0.5
T, Wmat = heisenberg_j1j2_cmpo(J1, J2)

α = 2^(1/4)
βs = 0.64 * α .^ (0:27)

β = βs[end]
@load "J1J2/cooling1/dimer_phase_beta$(β).jld2" f E var ψ

function read_data(βs, prefix)
    EEs = Float64[]
    chis = Int[]
    ESs = [] 
    for β in βs
        @load "J1J2/cooling1/$(prefix)$(β).jld2" ψ
        χ = dim(space(ψ))
        SE = entanglement_entropy(ψ, β)
        Λβ = real.(diag(CircularCMPS.half_chain_singular_values(ψ, β).data))

        push!(chis, χ)
        push!(EEs, SE)
        push!(ESs, Λβ)
    end
    return chis, EEs, ESs
end

chis, EEs, ESs = read_data(βs, "dimer_phase_beta");

T1s, Λs = Float64[], Float64[]
for (Λ, β) in zip(ESs[end-11:end], βs[end-11:end])
    #T1s = [T1s; (1:length(Λ)) .* 0.01 .+ 1/β]
    T1s = [T1s; fill(1/β, length(Λ))]
    Λs = [Λs; Λ]
end

for (ix, a) in enumerate(ESs[1])
    @show ix, a
end
for (ix, a) in enumerate(ESs[end-6])
    @show ix, a
end
for (ix, a) in enumerate(ESs[end])
    @show ix, a
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 1200))

ax1 = Axis(fig[1, 1], 
        xlabel = L"T",
        ylabel = L"χ", 
        )
scatter!(ax1, 1 ./ βs[1:end][end-11:end], chis[end-11:end], label=L"\text{cMPO, tol=}10^{-6}")
axislegend(ax1, position=:rt, framevisible=false)
@show fig

ax2 = Axis(fig[2, 1], 
        xlabel = L"T",
        ylabel = L"S_E", 
        )
scatter!(ax2, 1 ./ βs[1:end][end-11:end], EEs[end-11:end], label=L"\text{cMPO, tol=}10^{-6}")
axislegend(ax2, position=:rb, framevisible=false)
@show fig

ax3 = Axis(fig[3, 1], 
        xlabel = L"T",
        ylabel = L"ES",
        yscale = log10, 
        )
ylims!(ax3, (1e-4, 1))
scatter!(ax3, T1s, Λs, label=L"\text{cMPO, tol=}10^{-6}")
axislegend(ax3, position=:rb, framevisible=false)
@show fig

save("J1J2/J1J2_dimer_phase_ES_result_cooling1.pdf", fig)
