using LinearAlgebra, TensorKit
using Revise
using CircularCMPS
using CairoMakie
using JLD2 

J1, J2 = 1, 0.5
T, Wmat = heisenberg_j1j2_cmpo(J1, J2)
χs = [4, 6, 9]

β = 32

ts = [0.01:0.01:0.09 ; 0.12:0.02:0.18]
entropies = Float64[]
for t in ts 
    @load "J1J2/dimer_phase_temp$(t).jld2" fs Es vars
    S = (Es[end-1] - fs[end-1]) / t
    push!(entropies, S)
    @show t, fs[end-1], Es[end-1], vars[end-1]
    @show t, fs[end], Es[end], vars[end]
end

#ψ0 = CMPSData(T.Q, T.Ls)
#steps = 1:100
#
## power method, shift spectrum
#ψ = ψ0
#
#χ = 6
#for ix in steps 
#    Tψ = left_canonical(T*ψ)[2]
#    ψ = left_canonical(ψ)[2]
#    Tψ = direct_sum(Tψ, ψ, 0.95^ix, β) 
#    ψ = compress(Tψ, χ, β; tol=1e-6, init=ψ)
#    ψL = W_mul(Wmat, ψ)
#
#    f = free_energy(T, ψL, ψ, β)
#    E = energy(T, ψL, ψ, β)
#    var = variance(T, ψ, β)
#    @show χ, 0.95^ix, f, E, var
#    @show χ, ix, (E-f)*β
#end
#ψ1, var, _, _, _ = CircularCMPS.boundary_cmps_var(T, ψ, β)
#ψL1 = W_mul(Wmat, ψ1)
#f1 = free_energy(T, ψL1, ψ1, β)
#E1 = energy(T, ψL1, ψ1, β)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))

ax1 = Axis(fig[1, 1], 
        xlabel = L"T",
        ylabel = L"S", 
        )

lines!(ax1, ts, entropies)
#axislegend(ax1, position=:rb, framevisible=false)
@show fig