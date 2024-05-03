using LinearAlgebra, TensorKit, KrylovKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

T, W = xxz_fm_cmpo(1.0)
ψ0 = CMPSData(T.Q, T.Ls)

β = parse(Float64, ARGS[1])

ψs = CMPSData[]
fs = Float64[]
ss = Float64[]

ψ1 = ψ0
for χ in [4, 8, 12, 16, 20, 24]
    global ψs, fs, ss
    global ψ1
    for ix in 1:5 
        Tψ1 = left_canonical(T*ψ1)[2]
        ψ1 = left_canonical(ψ1)[2]
        Tψ1 = direct_sum(Tψ1, ψ1)
        ψ1 = compress(Tψ1, χ, β; init=ψ1, maxiter=50)
        ψL1 = W_mul(W, ψ1)

        f1 = free_energy(T, ψL1, ψ1, β)
        s1 = klein(ψL1, ψ1, β)
        @show ix, f1, s1
    end
    ψ1, f1, _, _, _ = leading_boundary_cmps(T, ψ1, β; maxiter=1000)
    s1 = klein(ψ1, ψ1, β)
    @show χ, f1, s1
    push!(fs, f1)
    push!(ψs, ψ1)
    push!(ss, s1)
    @save "xxz/heisenberg_beta$(β).jld2" ψs fs ss
end
