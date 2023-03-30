using LinearAlgebra, TensorKit
using Revise
using CircularCMPS
using CairoMakie
using JLD2 

J1, J2 = 1, 0.5
T, Wmat = heisenberg_j1j2_cmpo(J1, J2)

χs = [3, 6, 9]

ψ0 = CMPSData(T.Q, T.Ls)

α = 2^(1/4)
βs = 1.28 * α .^ (0:23)

steps = 1:200

# power method, shift spectrum
fs, Es, vars = Float64[], Float64[], Float64[]
ψs = CMPSData[]
ψ = ψ0

for β in βs
    global ψ
    for χ in χs
        f, E, var = fill(-999, 3)
        for ix in steps 
            Tψ = left_canonical(T*ψ)[2]
            ψ = left_canonical(ψ)[2]
            Tψ = direct_sum(Tψ, ψ, 0.5, β)
            ψ = compress(Tψ, χ, β; tol=1e-6, init=ψ)
            ψL = W_mul(Wmat, ψ)

            f = free_energy(T, ψL, ψ, β)
            E = energy(T, ψL, ψ, β)
            var = variance(T, ψ, β)
            @show χ, ix, f, E, (E-f)*β, var
        end
        push!(fs, f)
        push!(Es, E)
        push!(vars, var)
        push!(ψs, ψ)
    end

    @save "J1J2/dimer_phase_beta$(β).jld2" fs Es vars ψs
end