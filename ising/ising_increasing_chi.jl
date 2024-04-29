using LinearAlgebra, TensorKit, KrylovKit 
using CairoMakie 
using JLD2 
using Revise 
using CircularCMPS 

Γ = 1.0
T, Wmat = ising_cmpo(Γ)

βs = 160:16:192
χs = 4:2:20

for β in βs
    ψ = CMPSData(T.Q, T.Ls)
    for χ in χs
        ψ, f, E, var = power_iteration(T, Wmat, β, ψ, PowerMethod(fixχ=χ))
        @save "ising/results/ising_Gamma$(Γ)_beta$(β)-chi$(χ).jld2" β f E var ψ
        @show β, f, E, var
    end
end



