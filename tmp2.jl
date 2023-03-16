using LinearAlgebra, TensorKit, KrylovKit
using Revise
using CircularCMPS
using CairoMakie 

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
Tblk = Txy * Txy
Txy_fm = CMPO(zero2, [1/sqrt(2) * sm, 1/sqrt(2) * sp, -sqrt(Δ) * σz / 2], [1/sqrt(2) * sp, 1/sqrt(2) * sm, sqrt(Δ) * σz / 2], fill(zero2, 3, 3))

ψ0 = CMPSData(zero2, Txy.Ls)

β = 64

# simulation 1: power method 
f1s = Float64[]
ψ = ψ0
for ix in 1:100 
    Tψ = left_canonical(Txy*ψ)[2]
    ψ = left_canonical(ψ)[2]
    #Tψ = direct_sum(Tψ, ψ)
    ψ = compress(Tψ, 4, β; tol=1e-6, maxiter=1000, init=ψ)
    Rs_L = [-ψ.Rs[2], -ψ.Rs[1], -ψ.Rs[3]]
    ψL = CMPSData(ψ.Q, Rs_L)
    f = real(ln_ovlp(ψL, Txy, ψ, β) - ln_ovlp(ψL, ψ, β)) / (-β)
    push!(f1s, f)
    @show ix, f
end

# simulation 2: power method, double unit cell
ψ = ψ0
f2s = Float64[]
for ix in 1:100
    Tψ = left_canonical(Tblk*ψ)[2]
    ψ = compress(Tψ, 4, β; tol=1e-6, maxiter=1000, init=ψ)
    Rs_L = [-ψ.Rs[2], -ψ.Rs[1], -ψ.Rs[3]]
    ψL = CMPSData(ψ.Q, Rs_L)
    f = real(ln_ovlp(ψL, Tblk, ψ, β) - ln_ovlp(ψL, ψ, β)) / (-β)
    push!(f2s, f / 2)
    @show ix, f / 2
end

# simulation 3: power method, shift spectrum
f3s = Float64[]
ψ = ψ0
for ix in 1:100 
    Tψ = left_canonical(Txy*ψ)[2]
    ψ = left_canonical(ψ)[2]
    Tψ = direct_sum(Tψ, ψ)
    ψ = compress(Tψ, 4, β; tol=1e-6, maxiter=1000, init=ψ)
    Rs_L = [-ψ.Rs[2], -ψ.Rs[1], -ψ.Rs[3]]
    ψL = CMPSData(ψ.Q, Rs_L)
    f = real(ln_ovlp(ψL, Txy, ψ, β) - ln_ovlp(ψL, ψ, β)) / (-β)
    push!(f3s, f)
    @show ix, f
end

# simulation 4: power method, retated hamiltonian
ψ = CMPSData(Txy_fm.Q, Txy_fm.Ls)
f4s = Float64[]
for ix in 1:100 
    Tψ = left_canonical(Txy_fm*ψ)[2]
    ψ = compress(Tψ, 4, β; tol=1e-6, maxiter=1000, verbosity=1, init=ψ)
    Rs_L = [ψ.Rs[2], ψ.Rs[1], -ψ.Rs[3]]
    ψL = CMPSData(ψ.Q, Rs_L)
    f = real(ln_ovlp(ψL, Txy_fm, ψ, β) - ln_ovlp(ψL, ψ, β)) / (-β)
    push!(f4s, f)
    @show ix, f
end