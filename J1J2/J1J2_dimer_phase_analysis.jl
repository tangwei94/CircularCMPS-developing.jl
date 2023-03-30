using LinearAlgebra, TensorKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

J1, J2 = 1, 0.5
T, Wmat = heisenberg_j1j2_cmpo(J1, J2)
χs = [4, 6, 9]

β = 32

ts = [0.01:0.01:0.1 ; 0.12:0.02:0.18]
1 ./ ts
free_energies = Float64[]
energies = Float64[] 
variances = Float64[]
entropies = Float64[]
for t in ts 
    @load "J1J2/dimer_phase_temp$(t).jld2" fs Es vars
    S = (Es[end] - fs[end]) / t

    push!(free_energies, fs[end])
    push!(energies, Es[end])
    push!(variances, Es[end])
    push!(entropies, S)
    @show t, fs[end-1], Es[end-1], vars[end-1]
end

ts

for ix in 2:9
    @show ts[ix]
    E_findiff = -(free_energies[ix+1] / ts[ix+1] - free_energies[ix-1] / ts[ix-1]) / 0.02 * ts[ix]^2
    @show E_findiff, energies[ix] 
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))

ax1 = Axis(fig[1, 1], 
        xlabel = L"T",
        ylabel = L"S", 
        )

lines!(ax1, ts, entropies)
#axislegend(ax1, position=:rb, framevisible=false)
@show fig

# ============ check free energy ===========





