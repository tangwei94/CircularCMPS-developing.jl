using LinearAlgebra, TensorKit, KrylovKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

T, W = ising_cmpo(1.0)
ψ0 = CMPSData(T.Q, T.Ls)

β = 10
χ = 4

fs = Float64[]
ψs = CMPSData[]
ss = Float64[]

ψ1 = ψ0
for χ in [4, 6, 8, 10]
    for ix in 1:3 
        Tψ1 = left_canonical(T*ψ1)[2]
        ψ1 = compress(Tψ1, χ, β; tol=1e-6, init=ψ1)

        f1 = free_energy(T, ψ1, ψ1, β)
        s1 = klein(ψ1, ψ1, β)
        @show ix, f1, s1
    end
    ψ1, f1, _, _, _ = leading_boundary_cmps(T, ψ1, β; maxiter=100)
    s1 = klein(ψ1, ψ1, β)
    @show χ, f1, s1
    push!(fs, f1)
    push!(ψs, ψ1)
    push!(ss, s1)
end

@show fs
@show ss
