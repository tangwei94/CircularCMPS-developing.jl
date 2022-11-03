"""
    kinetic(ψ::CircularCMPS)

    Construct the tensor for kinetic energy density `(dψ† / dx)(dψ / dx)`.
"""
function kinetic(ψ::CircularCMPS)
    Q, Rs = get_matrices(ψ) 
    Oψ = Ref(Q) .* Rs - Rs .* Ref(Q)
    return sum(K_otimes.(Oψ, Oψ))
end

"""
    particle_density(ψ::CircularCMPS)

    Construct the tensor for particle density `ψ†ψ`.
"""
function particle_density(ψ::CircularCMPS)
    _, Rs = get_matrices(ψ) 
    return sum(K_otimes.(Rs, Rs))
end

"""
    point_interaction(ψ::CircularCMPS) 

    Construct the tensor for point interaction potential `ψ†ψ†ψψ`. 
"""
function point_interaction(ψ::CircularCMPS)
    _, Rs = get_matrices(ψ) 
    Oψ = Rs .* Rs
    return sum(K_otimes.(Oψ, Oψ))
end

"""
    pairing(ψ::CircularCMPS)

    Construct the tensor for pairint term `ψ†ψ†+ψψ`. 
"""
function pairing(ψ::CircularCMPS)
    Iψ = id(domain(ψ.Q))
    _, Rs = get_matrices(ψ) 
    Oψ = Rs .* Rs
    return sum(K_otimes.(Ref(Iψ), Oψ)) + sum(K_otimes.(Oψ, Ref(Iψ)))
end