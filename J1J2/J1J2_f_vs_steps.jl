using LinearAlgebra, TensorKit, KrylovKit
using Revise
using CircularCMPS
using CairoMakie
using JLD2 

J1, J2 = 1, 0.5
TJ1J2, Wmat = heisenberg_j1j2_cmpo(J1, J2)
Tblk = TJ1J2 * TJ1J2

β = 32
χ = 4

ψ0 = CMPSData(T.Q, T.Ls)

steps = 1:40

# simulation 1: power method 
f1s, var1s, varblk1s = Float64[], Float64[], Float64[] 
ψ1 = ψ0
for ix in steps 
    Tψ1 = left_canonical(TJ1J2*ψ1)[2]
    ψ1 = compress(Tψ1, χ, β; tol=1e-6, maxiter=1000, init=ψ1)
    ψL1 = W_mul(Wmat, ψ1)

    f1 = free_energy(T, ψL1, ψ1, β)
    var1 = variance(T, ψ1, β)
    varblk1 = variance(Tblk, ψ1, β)
    push!(f1s, f1)
    push!(var1s, var1)
    push!(varblk1s, varblk1)
    @show ix, f1, var1, varblk1
end

# simulation 2: power method, double unit cell
f2s, var2s, varblk2s = Float64[], Float64[], Float64[] 
ψ2 = ψ0
for ix in steps
    Tψ2 = left_canonical(Tblk*ψ2)[2]
    ψ2 = compress(Tψ2, χ, β; tol=1e-6, maxiter=1000, init=ψ2)
    ψL2 = W_mul(Wmat, ψ2)

    f2 = free_energy(T, ψL2, ψ2, β) / 2
    var2 = variance(T, ψ2, β)
    varblk2 = variance(Tblk, ψ2, β)
    push!(f2s, f2)
    push!(var2s, var2)
    push!(varblk2s, varblk2)
    @show ix, f2, var2, varblk2
end

# simulation 3: power method, shift spectrum
f3s, var3s, varblk3s = Float64[], Float64[], Float64[] 
ψ3 = ψ0
for ix in steps 
    Tψ3 = left_canonical(T*ψ3)[2]
    ψ3 = left_canonical(ψ3)[2]
    Tψ3 = direct_sum(Tψ3, ψ3)
    ψ3 = compress(Tψ3, χ, β; tol=1e-6, maxiter=1000, init=ψ3)
    ψL3 = W_mul(Wmat, ψ3)

    f3 = free_energy(T, ψL3, ψ3, β)
    var3 = variance(T, ψ3, β)
    varblk3 = variance(Tblk, ψ3, β)
    push!(f3s, f3)
    push!(var3s, var3)
    push!(varblk3s, varblk3)
    @show ix, f3, var3, varblk3
end

# plot
fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))

ax1 = Axis(fig[1, 1], 
        xlabel = L"\text{step}",
        ylabel = L"f", 
        )

lines!(ax1, steps, f1s, label=L"\text{power}")
lines!(ax1, steps, f2s, label=L"\text{unit cell}")
lines!(ax1, steps, f3s, label=L"\text{shift spect}")
axislegend(ax1, position=:rt, framevisible=false)
@show fig

ax1 = Axis(fig[2, 1], 
        xlabel = L"\text{step}",
        ylabel = L"\text{var for } T", 
        )

lines!(ax1, steps, var1s, label=L"\text{power}")
lines!(ax1, steps, var2s, label=L"\text{unit cell}")
lines!(ax1, steps, var3s, label=L"\text{shift spect}")
axislegend(ax1, position=:rt, framevisible=false)
@show fig

ax1 = Axis(fig[3, 1], 
        xlabel = L"\text{step}",
        ylabel = L"\text{var for } T^2", 
        )

lines!(ax1, steps, varblk1s, label=L"\text{power}")
lines!(ax1, steps, varblk2s, label=L"\text{unit cell}")
lines!(ax1, steps, varblk3s, label=L"\text{shift spect}")
axislegend(ax1, position=:rt, framevisible=false)
@show fig

save("J1J2/tests_J1_$(J1)_J2_$(J2)_beta$(β)_chi$(χ).pdf", fig)
