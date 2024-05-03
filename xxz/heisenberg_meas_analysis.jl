using LinearAlgebra, TensorKit, KrylovKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

@load "xxz/heisenberg_beta16.0.jld2" ψs fs ss;
@show ss
χs = get_χ.(ψs)
dtrg_results = [-0.4444511048433940, -0.4444566620370233, -0.4444569359048182, -0.4444569531622283, -0.4444569551020382]
dtrg_χs = [36, 64, 100, 144, 256]

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))
ax1 = Axis(fig[1, 1], xlabel=L"\chi = \sqrt{\chi_{dtrg}}", ylabel=L"f")
scatter!(ax1, χs, fs, marker=:circle, markersize=10, label=L"cMPO")
scatter!(ax1, sqrt.(dtrg_χs), dtrg_results, marker=:rect, markersize=10, label=L"dTRG")
@show fig

ax2 = Axis(fig[2, 1], xlabel=L"\chi", ylabel=L"S_K")
scatter!(ax2, χs, ss, marker=:circle, markersize=10)
@show fig

save("xxz/heisenberg_beta16.pdf", fig)

@load "xxz/heisenberg_beta32.0.jld2" ψs fs ss;
@show ss
χs = get_χ.(ψs)
dtrg_results = [-0.4434419354630325, -0.4434701307140408, -0.4434732098594786, -0.4434735813402718, -0.4434736411635757]
dtrg_χs = [36, 64, 100, 144, 256]

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))
ax1 = Axis(fig[1, 1], xlabel=L"\chi = \sqrt{\chi_{dtrg}}", ylabel=L"f")
scatter!(ax1, χs, fs, marker=:circle, markersize=10, label=L"cMPO")
scatter!(ax1, sqrt.(dtrg_χs), dtrg_results, marker=:rect, markersize=10, label=L"dTRG")
@show fig

ax2 = Axis(fig[2, 1], xlabel=L"\chi", ylabel=L"S_K")
scatter!(ax2, χs[3:end], ss[3:end], marker=:circle, markersize=10)
@show fig

save("xxz/heisenberg_beta32.pdf", fig)

@load "xxz/heisenberg_beta64.0.jld2" ψs fs ss;
@show ss
χs = get_χ.(ψs)
dtrg_results = [-0.4431492294786576, -0.4432148254719637, -0.4432259582013656, -0.4432281123556662 , -0.4432286489424274 ]
dtrg_χs = [36, 64, 100, 144, 256]

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))
ax1 = Axis(fig[1, 1], xlabel=L"\chi = \sqrt{\chi_{dtrg}}", ylabel=L"f")
scatter!(ax1, χs, fs, marker=:circle, markersize=10, label=L"cMPO")
scatter!(ax1, sqrt.(dtrg_χs), dtrg_results, marker=:rect, markersize=10, label=L"dTRG")
@show fig

ax2 = Axis(fig[2, 1], xlabel=L"\chi", ylabel=L"S_K")
scatter!(ax2, χs[3:end], ss[3:end], marker=:circle, markersize=10)
@show fig

save("xxz/heisenberg_beta64.pdf", fig)
#@load "xxz/heisenberg_beta64.0.jld2" ψs fs ss


