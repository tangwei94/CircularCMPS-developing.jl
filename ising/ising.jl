using LinearAlgebra, TensorKit, KrylovKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

Γ = 1.0
T, Wmat = ising_cmpo(Γ)
ψ = CMPSData(T.Q, T.Ls)

α = 2^(1/2)
βs = 0.32 * α .^ (0:14)

for β in βs
    global ψ
    ψ, f, E, var = power_iteration(T, Wmat, β, ψ, PowerMethod(tol_ES=1e-14))
    @save "ising/results/ising_Gamma$(Γ)_beta$(β)-toles12.jld2" β f E var ψ
    @show β, f, E, var
end

ΔEs = map(βs) do β 
    @load "ising/results/ising_Gamma1.0_beta$(β)-toles12.jld2" ψ
    abs(eigvals(K_mat(ψ, ψ).data)[end-1])
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))

ax1 = Axis(fig[1, 1], 
        xlabel = L"β",
        ylabel = L"βΔE", 
        )
scatter!(ax1, βs, βs .* ΔEs)
lines!(ax1, βs, βs .* 0 .+ sqrt(2))
#axislegend(ax1, position=:rt, framevisible=false)
@show fig

save("ising/results/ising_plot.pdf", fig)

