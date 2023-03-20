using LinearAlgebra, TensorKit, KrylovKit
using Revise
using CircularCMPS
using CairoMakie
using JLD2 

Δ = 1
T, W = xxz_af_cmpo(Δ)
Tblk = T * T

ψ0 = CMPSData(T.Q, T.Ls)

β = 32
χ = 4

steps = 1:40

# simulation 1: power method 
f1s, fblk1s, fsym1s, fblksym1s = Float64[], Float64[], Float64[], Float64[]
ψ1 = ψ0
for ix in steps 
    Tψ1 = left_canonical(T*ψ1)[2]
    ψ1 = compress(Tψ1, χ, β; tol=1e-6, init=ψ1)
    ψL1 = W_mul(W, ψ1)

    f1 = real(ln_ovlp(ψL1, T, ψ1, β) - ln_ovlp(ψL1, ψ1, β)) / (-β)
    fblk1 = real(ln_ovlp(ψL1, Tblk, ψ1, β) - ln_ovlp(ψL1, ψ1, β)) / (-2*β)
    fsym1 = real(ln_ovlp(ψ1, T, ψ1, β) - ln_ovlp(ψ1, ψ1, β)) / (-β)
    fblksym1 = real(ln_ovlp(ψ1, Tblk, ψ1, β) - ln_ovlp(ψ1, ψ1, β)) / (-2*β)
    push!(f1s, f1)
    push!(fblk1s, fblk1)
    push!(fsym1s, fsym1)
    push!(fblksym1s, fblk1)
    @show ix, f1, fblk1, fsym1, fblksym1
end

# simulation 2: power method, double unit cell
ψ = ψ0
f2s, fsym2s = Float64[], Float64[]
for ix in steps
    Tψ = left_canonical(Tblk*ψ)[2]
    ψ = compress(Tψ, χ, β; tol=1e-6, init=ψ)
    ψL = W_mul(W, ψ)

    f2 = real(ln_ovlp(ψL, Tblk, ψ, β) - ln_ovlp(ψL, ψ, β)) / (-2*β)
    fsym2 = real(ln_ovlp(ψ, Tblk, ψ, β) - ln_ovlp(ψ, ψ, β)) / (-2*β)
    push!(f2s, f2)
    push!(fsym2s, fsym2)
    @show ix, f2, fsym2
end

# simulation 3: power method, shift spectrum
f3s, fblk3s, fsym3s, fblksym3s = Float64[], Float64[], Float64[], Float64[]
ψ3 = ψ0
for ix in steps 
    Tψ3 = left_canonical(T*ψ3)[2]
    ψ3 = left_canonical(ψ3)[2]
    Tψ3 = direct_sum(Tψ3, ψ3, 0.95^ix, β)
    ψ3 = compress(Tψ3, χ, β; tol=1e-6, init=ψ3)
    ψL3 = W_mul(W, ψ3)

    f3 = real(ln_ovlp(ψL3, T, ψ3, β) - ln_ovlp(ψL3, ψ3, β)) / (-β)
    fblk3 = real(ln_ovlp(ψL3, Tblk, ψ3, β) - ln_ovlp(ψL3, ψ3, β)) / (-2*β)
    fsym3 = real(ln_ovlp(ψ3, T, ψ3, β) - ln_ovlp(ψ3, ψ3, β)) / (-β)
    fblksym3 = real(ln_ovlp(ψ3, Tblk, ψ3, β) - ln_ovlp(ψ3, ψ3, β)) / (-2*β)
    push!(f3s, f3)
    push!(fblk3s, fblk3)
    push!(fsym3s, fsym3)
    push!(fblksym3s, fblksym3)
    @show ix, f3, fblk3, fsym3, fblksym3
end

# simulation 4: variational optim
ψ4, f4, grad, _, history = leading_boundary_cmps(T, ψ3, β)
_, fblksym4, grad, _, history = leading_boundary_cmps(Tblk, ψ3, β)
fblksym4 /= 2
fblk4 = real(ln_ovlp(ψ4, Tblk, ψ4, β) - ln_ovlp(ψ4, ψ4, β)) / (-2*β)

# plot
fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))

ax1 = Axis(fig[1, 1], 
        xlabel = L"\text{step}",
        ylabel = L"f", 
        )

lines!(ax1, steps, fsym1s, label=L"\text{power}")
lines!(ax1, steps, fsym2s, label=L"\text{unit cell}")
lines!(ax1, steps, fsym3s, label=L"\text{shift spect}")
lines!(ax1, steps, f3s, label=L"\text{shift spect LR}") # different from fsym3s, no variational principle
lines!(ax1, [steps[1], steps[end]], [f4, f4], color=:black, label=L"\text{variational} T")
axislegend(ax1, position=:rt, framevisible=false)
@show fig

ax2 = Axis(fig[2, 1], 
        xlabel = L"\text{step}",
        ylabel = L"f \text{ all measured with } T^2", 
        )

lines!(ax2, steps, fblksym1s, label=L"\text{power}")
lines!(ax2, steps, fsym2s, label=L"\text{unit cell}")
lines!(ax2, steps, fblksym3s, label=L"\text{shift spect}")
lines!(ax2, [steps[1], steps[end]], [fblk4, fblk4], color=:gray, label=L"\text{variational } T")
lines!(ax2, [steps[1], steps[end]], [fblksym4, fblksym4], color=:black, label=L"\text{variational } T^2")
axislegend(ax2, position=:rt, framevisible=false)
@show fig

save("xxz/AF_heisenberg_tests.pdf", fig)