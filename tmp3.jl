using LinearAlgebra, TensorKit, KrylovKit
using ChainRules, TensorKitAD
using OptimKit

using Revise
using CircularCMPS 

pauli = CircularCMPS.pauli
sp, sm, σz, zero2 = pauli.sp, pauli.sm, pauli.σz, pauli.zero2
Δ = 0.2
Ls = [1/sqrt(2) * sm, 1/sqrt(2) * sp, sqrt(Δ) * σz / 2]
Rs = [1/sqrt(2) * sp, 1/sqrt(2) * sm, sqrt(Δ) * σz / 2]
H = 0
T = CMPO(H*σz, Ls, Rs, fill(zero2, 3, 3))
T2 = CMPO(H*σz, Rs, Ls, fill(zero2, 3, 3))

L = 0.5
CircularCMPS.inner(T, T2, L) + CircularCMPS.inner(T2, T, L)
CircularCMPS.inner(T, T, L) + CircularCMPS.inner(T2, T2, L)

G1 = H * rand(ComplexF64, 3, 3)
G1 = 0.5*(G1 + G1')
ΔG = rand(ComplexF64, 3, 3)
function normal_meas(G::Matrix, L::Real)
    G = G + G'
    Q = exp(G)
    Qinv = exp(-G)
    ln = Q * T * Qinv * T2 * Q 
    rn = T2 * Q * T 

    meas = inner(ln, rn, L) + inner(rn, ln, L) - inner(ln, ln, L) - inner(rn, rn, L)
    return -real(meas)
end

L = 20
function fg(x::ComplexF64)
    f = x->normal_meas(diagm([x, -x, 0]), L) / L
    fG = f(x)
    gG = f'(x)
    return fG, gG
end

α = 1e-6
f = x->normal_meas(diagm([x, -x, 0]), L) / L
(f(0.1 + α) - f(0.1 - α)) / (2*α)
f'(0.1+0im)

for α in 1:20
    @show α, f(α)
end

inner1(G::Matrix, Ga::Matrix, Gb::Matrix) = real(tr(Ga' * Gb))
#retract1(G::Matrix, Ga::Matrix, α::Real) = G + 0.5*α*(Ga + Ga')

optalg_LBFGS = LBFGS(;maxiter=20, gradtol=1e-9, verbosity=2)
result = optimize(fg, H+0im, optalg_LBFGS; inner = (x,a,b)->real(a'*b))#, retract=retract1, transport!=transport1!, )

G1 = result[1]
G1 = 0.5*(G1+G1')
for L in 2 .^ (1:1:10)
    @show L, normal_meas(G1, L) / L
end

T1 = exp(-G1/ 2) * T * exp(G1/2)
norm.(T1.Ls)
norm.(T1.Rs)