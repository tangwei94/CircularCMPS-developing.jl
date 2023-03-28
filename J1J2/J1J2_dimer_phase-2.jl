using LinearAlgebra, TensorKit
using Revise
using CircularCMPS
using CairoMakie
using JLD2 

J1, J2 = 1, 0.5
T, Wmat = heisenberg_j1j2_cmpo(J1, J2)

temperature = parse(Float64, ARGS[1])
β = 1/temperature
χs = [3, 6, 9]

ψ0 = CMPSData(T.Q, T.Ls)

steps = 1:50

# power method
fs, Es, vars = Float64[], Float64[], Float64[]
ψs = CMPSData[]
ψ = ψ0

T2 = T*T

for χ in χs
    global ψ
    f, E, var = fill(-999, 3)
    for ix in steps 
        Tψ = left_canonical(T2*ψ)[2]
        ψ = compress(Tψ, χ, β; maxiter=100, init=ψ)
        ψL = W_mul(Wmat, ψ)

        f = free_energy(T2, ψL, ψ, β) / 2
        E = energy(T2, ψL, ψ, β) / 2
        var = variance(T2, ψ, β)
        @show χ, ix, f, E, var
        @show χ, ix, (E-f)*β, var
    end
    push!(fs, f)
    push!(Es, E)
    push!(vars, var)
    push!(ψs, ψ)

    if χ > 3
        ψ = CircularCMPS.boundary_cmps_var_optim(T2, ψ, β; maxiter=1000)[1]
        ψL = W_mul(Wmat, ψ)
        f = free_energy(T2, ψL, ψ, β) / 2
        E = energy(T2, ψL, ψ, β) / 2
        var = variance(T2, ψ, β)
        @show f, E, var, (E-f)*β
        push!(fs, f)
        push!(Es, E)
        push!(vars, var)
        push!(ψs, ψ)
    end
end

@save "J1J2/dimer_phase_blk2_temp$(temperature).jld2" fs Es vars ψs