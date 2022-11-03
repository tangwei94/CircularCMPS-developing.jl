"""
    kinetic(ψ::CircularCMPS) 

    Construct the tensor for kinetic energy density `(dψ† / dx)(dψ / dx)` measurement.
"""
function kinetic(ψ::CircularCMPS) 
    Q, R = ψ.Q, ψ.R # TODO. AD?
    @tensor O[-1, -2; -3, -4] := Q[-1, 1] * R[1, 3, -3] * Q'[2, -4] * R'[-2, 2, 3] +
                                 R[-1, 3, 1] * Q[1, -3] * R'[2, -4, 3] * Q'[-2, 2] -
                                 Q[-1, 1] * R[1, 3, -3] * R'[2, -4, 3] * Q'[-2, 2] - 
                                 R[-1, 3, 1] * Q[1, -3] * Q'[2, -4] * R'[-2, 2, 3]
    return O
end

"""
    density(ψ::cmps) -> TensorMap{ComplexSpace, 2, 2}

    Construct the tensor for particle density `ψ†ψ` measurement.
"""
function particle_density(ψ::cmps) 
    R = ψ.R
    @tensor O[-1, -2; -3, -4] := R[-1, 1, -3] * R'[-2, -4, 1]
    return O
end

"""
    point_interaction(ψ::cmps) -> TensorMap{ComplexSpace, 2, 2}

    Construct the tensor for point interaction potential `ψ†ψ†ψψ` measurement. 
"""
function point_interaction(ψ::cmps)
    R = ψ.R
    @tensor O[-1, -2; -3, -4] := R[-1, 3, 1] * R[1, 4, -3] * R'[2, -4, 3] * R'[-2, 2, 4]
    return O
end

"""
    build_I_R(ψ::cmps) -> TensorMap{ComplexSpace, 2, 1}
"""
function build_I_R(ψ::cmps)
    χ, d = get_chi(ψ), get_d(ψ)
    I_R = zeros(ComplexF64, (χ, d, χ))
    for ix in 1:d
        I_R[:, ix, :] = Matrix{ComplexF64}(I, (χ, χ))
    end 
    I_R = convert_to_tensormap(I_R, 2)
end

@non_differentiable build_I_R(ψ::cmps)

"""
    pairing(ψ::cmps) -> TensorMap{ComplexSpace, 2, 2}
"""
function pairing(ψ::cmps)
    R = ψ.R
    I_R = build_I_R(ψ)
    @tensor O[-1, -2; -3, -4] := R[-1, 3, 1] * R[1, 4, -3] * I_R'[2, -4, 3] * I_R'[-2, 2, 4] +
                                I_R[-1, 3, 1] * I_R[1, 4, -3] * R'[2, -4, 3] * R'[-2, 2, 4]
    return O
end