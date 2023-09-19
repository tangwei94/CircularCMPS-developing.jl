using LinearAlgebra, TensorKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

hz = 1.5
hz = parse(Float64, ARGS[1])
T, Wmat = xxz_af_cmpo(1; hz=hz)

α = 2^(1/4)
βs = 0.32 * α .^ (0:32)

ψ = CMPSData(T.Q, T.Ls)

for β in βs
    global ψ
    ψ, f, E, var = power_iteration(T, Wmat, β, ψ, PowerMethod(tol_ES=1e-8))
    @save "heisenberg/results/cooling/heisenberg_hz$(hz)_beta$(β)-toles8.jld2" β f E var ψ
    @show β, f, E, var
end

for β in βs
    @load "heisenberg/results/cooling/heisenberg_hz$(hz)_beta$(β)-toles8.jld2" β f E var ψ

    ψL = Wmat * ψ
    ΛLR = half_chain_singular_values(ψL, ψ, β)
    ΛRR = half_chain_singular_values(ψ, β)

    SLR = - real(tr(ΛLR * log(ΛLR)))
    SRR = - real(tr(ΛRR * log(ΛRR)))
    println("β = $β, SLR = $SLR, SRR = $SRR")
end

# ---> 
# when h_z >= 2.0, the system is polarized in the ground state, and there is a gap in the energy spectrum. SLR goes to zero at large beta, and there is a peak at finite temperature.
# however, it appears SRR still has a finite value at large beta, showing different entanglement structure 
# when h_z < 2.0, the ground state is not polarized. the system has gapless spin excitations
# <--- 

# importance scattering check
# importance scattering check
function show_analysis_results(ψL, ψ, β)

    χ2 = length(ψ.Q.data)

    #ΛRR = half_chain_singular_values(ψ, β)
    #ΛLL = half_chain_singular_values(ψL, β)
    #ΛRR_diag = diag(real.(ΛRR.data))
    #ΛLL_diag = diag(real.(ΛLL.data))

    scattering, reweighted = CircularCMPS.half_chain_singular_values_testtool(ψL, ψ, β);

    scattering_norms = norm.(scattering.data)
    scattering_norms /= maximum(scattering_norms)
    reweighted_norms = norm.(reweighted.data)
    reweighted_norms /= maximum(reweighted_norms)
    fig1, ax, hm = heatmap(log10.(scattering_norms), colorrange=(-3, 0), colormap=:Blues)
    Colorbar(fig1[:, end+1], hm)
    fig2, ax, hm = heatmap(log10.(reweighted_norms), colorrange=(-6, 0), colormap=:Blues)
    Colorbar(fig2[:, end+1], hm)

    function meas1(M)
        map(1:χ2) do ix0
            return sum(M[ix0, 1:end]) + sum(M[1:end, ix0])
        end
    end

    fig3, ax2, _ = lines(log10.(meas1(scattering_norms)))
    ax3 = Axis(fig3[2, 1])
    lines!(ax3, log10.(meas1(reweighted_norms)))
    #scatter!(ax3, log10.(norm.(ΛRR_diag)), marker='x')
    #scatter!(ax3, log10.(norm.(ΛLL_diag)), marker='o')

    return fig1, fig2, fig3
end

β = βs[end]
@load "heisenberg/results/cooling/heisenberg_hz$(hz)_beta$(β)-toles8.jld2" β f E var ψ
ψL = Wmat * ψ

fig, fig2, fig3 = show_analysis_results(ψL, ψ, β);
@show fig
@show fig2
#@show fig3

# ---> 
# "scattering matrix".
# <--- 

Λ, U = eigen(Hermitian(Wmat)) 
@show Λ
function normalize(a, b)
    M = ComplexF64[0 0 a; 0 0 b; a b 0]
    expM = U * exp(M) * U'
    return real(ln_ovlp(expM * ψ, ψ, β) + ln_ovlp(ψ, ψ, β) - ln_ovlp(ψ, Wmat * expM * ψ, β) - ln_ovlp(ψ, expM * Wmat * ψ, β))
end

minimal = 1e19
for xa in -2.5:0.1:2.5, xb in -5.5:0.1:5.5
    n = normalize(xa, xb)
    if minimal > n
        @show xa, xb, normalize(xa, xb)
        minimal = n
    end
end

# ---> 
# tried to find a way to normalize the cMPO when h>=2.0, but failed
# in this system SRR always larger than SLR, means the the right boundary MPS always have finite entanglement even when the system is a product state
# boundary cMPS describes the environment of the system, in this case contains at least one spin, and possibilities for spin exchange (contribution from excitations). If the boundary MPS becomes a product state, that would be unphysical
# the cMPO is still somehow optimal for the boundary cMPS optimization, see the importance scattering check above
# <---


