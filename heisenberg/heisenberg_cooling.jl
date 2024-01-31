using LinearAlgebra, TensorKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

hz = 1.5
#hz = parse(Float64, ARGS[1])
T, Wmat = xxz_af_cmpo(1; hz=hz)

α = 2^(1/4)
βs = 0.32 * α .^ (0:32)

ψ = CMPSData(T.Q, T.Ls)

for β in βs
    global ψ
    ψ, f, E, var = power_iteration(T, Wmat, β, ψ, PowerMethod(tol_ES=1e-10))
    @save "heisenberg/results/cooling/heisenberg_hz$(hz)_beta$(β)-toles10.jld2" β f E var ψ
    @show β, f, E, var
end

#Λ, U = eigen(Hermitian(Wmat)) 
#@show Λ
#function normalize(a, b)
#    M = ComplexF64[0 0 a; 0 0 b; a b 0]
#    expM = U * exp(M) * U'
#    return real(ln_ovlp(expM * ψ, ψ, β) + ln_ovlp(ψ, ψ, β) - ln_ovlp(ψ, Wmat * expM * ψ, β) - ln_ovlp(ψ, expM * Wmat * ψ, β))
#end
#
#minimal = 1e19
#for xa in -2.5:0.1:2.5, xb in -5.5:0.1:5.5
#    n = normalize(xa, xb)
#    if minimal > n
#        @show xa, xb, normalize(xa, xb)
#        minimal = n
#    end
#end

# ---> 
# tried to find a way to normalize the cMPO when h>=2.0, but failed
# in this system SRR always larger than SLR, means the the right boundary MPS always have finite entanglement even when the system is a product state
# boundary cMPS describes the environment of the system, in this case contains at least one spin, and possibilities for spin exchange (contribution from excitations). If the boundary MPS becomes a product state, that would be unphysical
# the cMPO is still somehow optimal for the boundary cMPS optimization, see the importance scattering check above
# <---


