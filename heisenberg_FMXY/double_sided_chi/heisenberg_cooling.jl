using LinearAlgebra, TensorKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

hz = 1.5
#hz = parse(Float64, ARGS[1])
T, Wmat = xxz_fm_cmpo(1; hz=hz)

α = 2^(1/4)
βs = 0.32 * α .^ (0:32)

ψ = CMPSData(T.Q, T.Ls)

@show (@__FILE__)

for β in βs
    global ψ
    ψ, f, E, var = power_iteration(T, Wmat, β, ψ, PowerMethod(tol_ES=1e-9, spect_shifting=1))
    @save "heisenberg_FMXY/double_sided_chi/data/heisenberg_hz$(hz)_beta$(β)-toles9-spect_shifting1.jld2" β f E var ψ
    println("=== heisenberg FMXY, double sided chi determination, hz=$(hz), β=$(β), tol_ES=1e-9, spect_shifting=1 (!), E=$(E), var=$(var) ===")
end


#β = βs[end-15]
#@load "heisenberg_FMXY/double_sided_chi/data/heisenberg_hz$(hz)_beta$(β)-toles10.jld2" β f E var ψ
#Λ, U = eigen(Hermitian(Wmat)) 
#@show Λ
#function maximize_norm(a, b, c, d)
#    M = ComplexF64[0 0 a+c*im; 0 0 b+d*im; a-c*im b-d*im 0]
#    @assert norm(U * M * U' * Wmat + Wmat * U * M * U') < 1e-14 
#    expM = U * exp(M) * U'
#    gauged_ψ = expM * ψ
#
#    return real(ln_ovlp(gauged_ψ, gauged_ψ, β))
#end
#
#using FiniteDifferences, OptimKit
#
#_f2(x) = maximize_norm(x[1], x[2], x[3], x[4])
#
#x0 = zeros(4)
#_fg2(x) = (_f2(x), grad(central_fdm(5, 1), _f2, x)[1])
#
#x, fx, gx, numfg, normgradhistory = optimize(_fg2, x0, LBFGS(maxiter=100, verbosity=2))
#x
#fx
#
#
## ---> 
## tried to find a way to normalize the cMPO when h>=2.0, but failed
## in this system SRR always larger than SLR, means the the right boundary MPS always have finite entanglement even when the system is a product state
## boundary cMPS describes the environment of the system, in this case contains at least one spin, and possibilities for spin exchange (contribution from excitations). If the boundary MPS becomes a product state, that would be unphysical
## the cMPO is still somehow optimal for the boundary cMPS optimization, see the importance scattering check above
## <---
#
#function show_analysis_results(ψL, ψ, β)
#
#    χ2 = length(ψ.Q.data)
#
#    scattering, reweighted = CircularCMPS.half_chain_singular_values_testtool(ψL, ψ, β);
#
#    scattering_norms = norm.(scattering.data)
#    scattering_norms /= maximum(scattering_norms)
#    reweighted_norms = norm.(reweighted.data)
#    reweighted_norms /= maximum(reweighted_norms)
#    fig1, ax, hm = heatmap(log10.(scattering_norms), colorrange=(-3, 0), colormap=:Blues)
#    #fig1, ax, hm = heatmap(scattering_norms, colorrange=(0, 1), colormap=:Blues)
#    Colorbar(fig1[:, end+1], hm)
#    fig2, ax, hm = heatmap(log10.(reweighted_norms), colorrange=(-6, 0), colormap=:Blues)
#    Colorbar(fig2[:, end+1], hm)
#
#    return fig1, fig2
#end
#
#a, b, c, d = x
#M = ComplexF64[0 0 a+c*im; 0 0 b+d*im; a-c*im b-d*im 0]
#expM = U * exp(M) * U'
#eigen(Hermitian(expM))
#ψ1 = expM * ψ
#ψL1 = Wmat * ψ1
#2*ln_ovlp(ψL, ψ, β) - ln_ovlp(ψL, ψL, β) - ln_ovlp(ψ, ψ, β)
#2*ln_ovlp(ψL1, ψ1, β) - ln_ovlp(ψL1, ψL1, β) - ln_ovlp(ψ1, ψ1, β)
#
#entanglement_entropy(ψ, β)
#entanglement_entropy(ψ1, β)
#
#fig, fig2 = show_analysis_results(ψL1, ψ1, β);
#@show fig
#@show fig2
#
#βs1 = βs[1:end-1]
#function calc_entanglement(index)
#    SLRs, SRRs = Float64[], Float64[]
#    for β in βs1
#        @load "heisenberg_FMXY/data/heisenberg_hz$(hz)_beta$(β)-$(index).jld2" β f E var ψ
#
#        ψ = expM * ψ
#        ψL = Wmat * ψ
#        ΛLR = half_chain_singular_values(ψL, ψ, β)
#        ΛRR = half_chain_singular_values(ψ, β)
#
#        SLR = - real(tr(ΛLR * log(ΛLR)))
#        SRR = - real(tr(ΛRR * log(ΛRR)))
#        push!(SLRs, SLR)
#        push!(SRRs, SRR)
#    end
#    return SLRs, SRRs
#end
#
#SLRs, SRRs = calc_entanglement("toles10");
#
#
#fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (400, 400))
#
#ax1 = Axis(fig[1, 1], 
#        xlabel = L"β",
#        ylabel = L"S", 
#        xscale = log10
#        )
#
##lines!(ax1, βs1, SLRs, label=L"\text{SLR, tolES8}")
#lines!(ax1, βs1, SRRs, label=L"\text{SRR, tolES8}")
#axislegend(ax1, position=:rt, framevisible=false)
#@show fig