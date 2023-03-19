using LinearAlgebra, TensorKit
using Revise
using CircularCMPS
using CairoMakie
using JLD2 

J1, J2 = 1, 0.5
TJ1J2, Wmat = heisenberg_j1j2_cmpo(J1, J2)

temperature = 0.1
β = 1/temperature
χ = 4

ψ0 = CMPSData(TJ1J2.Q, TJ1J2.Ls)

steps = 1:100

# simulation 3: power method, shift spectrum
fs = Float64[]
ψ = ψ0
for ix in steps 
    Tψ = left_canonical(TJ1J2*ψ)[2]
    ψ = left_canonical(ψ)[2]
    Tψ = direct_sum(Tψ, ψ)
    ψ = compress(Tψ, χ, β; tol=1e-6, maxiter=1000, init=ψ)
    ψL = W_mul(Wmat, ψ)

    f = real(ln_ovlp(ψL, TJ1J2, ψ, β) - ln_ovlp(ψL, ψ, β)) / (-β)
    push!(fs, f)
    @show ix, f
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
#lines!(ax1, [steps[1], steps[end]], [f4, f4], color=:black, label=L"\text{variational}")
axislegend(ax1, position=:rt, framevisible=false)
@show fig
save("J1J2/J1_$(J1)_J2_$(J2)_beta$(β)_chi$(χ).pdf", fig)

@save "J1J2/J1_$(J1)_J2_$(J2)_beta$(β)_chi$(χ).jld2" f1s f2s f3s ψ1 ψ2 ψ3