using LinearAlgebra, TensorKit, KrylovKit
using Revise
using CircularCMPS 

# ising 
Id = TensorMap(ComplexF64[1 0; 0 1], ℂ^2, ℂ^2)
σz = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)
σx = TensorMap(ComplexF64[0 1; 1 0], ℂ^2, ℂ^2)
σy = TensorMap(ComplexF64[0 1im; -1im 0], ℂ^2, ℂ^2)
zero2 = zero(Id)

function ising_cmpo(Γ::Real)
    CMPO(Γ*σx, [σz], [σz], fill(zero2, 1, 1))
end

Γ = 1.0
β = 128
T = ising_cmpo(Γ)

ψ2 = CMPSData(Γ*σx, [σz])
for ix in 1:100 
    ψ2 = compress(T*ψ2, 12, β; tol=1e-6, init=ψ2)
    f2 = (ln_ovlp(ψ2, T, ψ2, β) - ln_ovlp(ψ2, ψ2, β)) / (-β)
    @show ix, f2
end
