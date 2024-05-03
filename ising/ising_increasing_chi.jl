using LinearAlgebra, TensorKit, KrylovKit 
using CairoMakie 
using JLD2 
using Revise 
using CircularCMPS 

Γ = 1.0
T, Wmat = ising_cmpo(Γ)

βs = 32:16:192
χs = 22:2:32

for β in βs
    @load "ising/results/ising_Gamma$(Γ)_beta$(β)-chi20.jld2" ψ
    for χ in χs
        ψ, f, E, var = power_iteration(T, Wmat, β, ψ, PowerMethod(fixχ=χ))
        @save "ising/results/ising_Gamma$(Γ)_beta$(β)-chi$(χ).jld2" β f E var ψ
        @show β, f, E, var
    end
end



