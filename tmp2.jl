using LinearAlgebra, TensorKit, KrylovKit
using Revise
using CircularCMPS 

# pauli operators  
Id = TensorMap(ComplexF64[1 0; 0 1], ℂ^2, ℂ^2)
σz = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)
σx = TensorMap(ComplexF64[0 1; 1 0], ℂ^2, ℂ^2)
σy = TensorMap(ComplexF64[0 1im; -1im 0], ℂ^2, ℂ^2)
sp = TensorMap(ComplexF64[0 1; 0 0], ℂ^2, ℂ^2)
sm = TensorMap(ComplexF64[0 0; 1 0], ℂ^2, ℂ^2)
zero2 = zero(Id)

Δ = 1
Txy = CMPO(zero2, [-1/sqrt(2) * sm, -1/sqrt(2) * sp, -sqrt(Δ) * σz / 2], [1/sqrt(2) * sp, 1/sqrt(2) * sm, sqrt(Δ) * σz / 2], fill(zero2, 3, 3))

ψ = CMPSData(zero2, Txy.Ls)

β = 16
for ix in 1:100 
    Tψ = left_canonical(Txy*ψ)[2]
    a = ln_ovlp(Tψ, Tψ, β)
    ψ = left_canonical(ψ)[2]
    Tψ = direct_sum(Tψ, ψ; α = log(1) / β)
    ψ = compress(Tψ, 4, β; tol=1e-6, maxiter=1000, init=ψ)
    Rs_L = [-ψ.Rs[2], -ψ.Rs[1], ψ.Rs[3]]
    ψL = CMPSData(ψ.Q, Rs_L)
    f = real(ln_ovlp(ψL, Txy, ψ, β) - ln_ovlp(ψL, ψ, β)) / (-β)
    @show ix, f
end

Txy_fm = CMPO(zero2, [1/sqrt(2) * sm, 1/sqrt(2) * sp, -Δ * σz / 2], [1/sqrt(2) * sp, 1/sqrt(2) * sm, σz / 2], fill(zero2, 3, 3))

ψ = CMPSData(Txy_fm.Q, Txy_fm.Ls)

for ix in 1:100 
    Tψ = left_canonical(Txy_fm*ψ)[2]
    ψ = left_canonical(ψ)[2]
    Tψ = direct_sum(Tψ, ψ)
    ψ = compress(Tψ, 4, β; tol=1e-6, maxiter=1000, verbosity=1, init=ψ)
    Rs_L = [ψ.Rs[2], ψ.Rs[1], ψ.Rs[3]]
    ψL = CMPSData(ψ.Q, Rs_L)
    f = real(ln_ovlp(ψL, Txy_fm, ψ, β) - ln_ovlp(ψL, ψ, β)) / (-β)
    @show ix, f
end