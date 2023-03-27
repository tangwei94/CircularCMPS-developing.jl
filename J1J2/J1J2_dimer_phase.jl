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

# power method, shift spectrum
fs, Es, vars = Float64[], Float64[], Float64[]
ψs = CMPSData[]
ψ = ψ0

T2 = T*T

for χ in χs
    global ψ
    f, E, var = fill(-999, 3)
    for ix in steps 
        Tψ = left_canonical(T*ψ)[2]
        ψ = left_canonical(ψ)[2]
        Tψ = direct_sum(Tψ, ψ, 0.97^ix, β)
        ψ = compress(Tψ, χ, β; tol=1e-6, init=ψ)
        ψL = W_mul(Wmat, ψ)

        f = free_energy(T, ψL, ψ, β)
        E = energy(T, ψL, ψ, β)
        var = variance(T, ψ, β)
        @show χ, ix, f, E, var
        @show χ, ix, (E-f)*β, var
    end
    push!(fs, f)
    push!(Es, E)
    push!(vars, var)
    push!(ψs, ψ)

    if χ > 3
        ψ = CircularCMPS.boundary_cmps_var_optim(T, ψ, β; maxiter=2000)[1]
        ψL = W_mul(Wmat, ψ)
        f = free_energy(T, ψL, ψ, β)
        E = energy(T, ψL, ψ, β)
        var = variance(T, ψ, β)
        @show f, E, var, (E-f)*β
        push!(fs, f)
        push!(Es, E)
        push!(vars, var)
        push!(ψs, ψ)
    end
end

@save "J1J2/dimer_phase_temp$(temperature).jld2" fs Es vars ψs