using LinearAlgebra, TensorKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

hz = 4.0
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

# importance scattering check. measure sqrt(ΛRR) * MLR * sqrt(ΛRR) 
β = βs[end]
@load "heisenberg/results/cooling/heisenberg_hz$(hz)_beta$(β)-toles8.jld2" β f E var ψ
ψL = Wmat * ψ
ΛLR = half_chain_singular_values(ψL, ψ, β);
ΛRR = half_chain_singular_values(ψ, β);

ΛLR.data |> diag
ΛRR.data |> diag

scattering, reweighted = CircularCMPS.half_chain_singular_values_testtool(ψL, ψ, β)

fig, ax, hm = heatmap(log.(norm.(scattering.data)), colorrange=(-9, 0), colormap=:Blues)
Colorbar(fig[:, end+1], hm)
@show fig

function importance_meas(M)
    χ2 = size(M, 1)
    map(1:χ2) do ix0
        return sum(M[ix0, 1:end]) + sum(M[1:end, ix0])
    end
end

lines(log10.(importance_meas(norm.(scattering.data))))

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


