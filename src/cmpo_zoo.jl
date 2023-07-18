module pauli
    using TensorKit
    const Id = TensorMap(ComplexF64[1 0; 0 1], ℂ^2, ℂ^2)
    const σz = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)
    const σx = TensorMap(ComplexF64[0 1; 1 0], ℂ^2, ℂ^2)
    const σy = TensorMap(ComplexF64[0 1im; -1im 0], ℂ^2, ℂ^2)
    const zero2 = zero(Id)
    const sp = TensorMap(ComplexF64[0 1; 0 0], ℂ^2, ℂ^2)
    const sm = TensorMap(ComplexF64[0 0; 1 0], ℂ^2, ℂ^2)
end

function cmpo_checker(T::CMPO)
    Id = id(codomain(T.Q))
    K = sum(T.Rs .⊗ T.Ls .⊗ Ref(Id) + Ref(Id) .⊗ T.Rs .⊗ T.Ls + (T.Rs * T.Ps) .⊗ T.Ls) + T.Q ⊗ Id ⊗ Id + Id ⊗ T.Q ⊗ Id + Id ⊗ Id ⊗ T.Q
    return -K
end

function ising_cmpo(Γ::Real)
    σx, σz, zero2 = pauli.σx, pauli.σz, pauli.zero2
    T = CMPO(Γ*σx, [σz], [σz], fill(zero2, 1, 1))
    W = fill(1.0, 1, 1)
    return T, W
end

function xxz_af_cmpo(Δ::Real; hz::Real=0.0)
    sp, sm, σz, zero2 = pauli.sp, pauli.sm, pauli.σz, pauli.zero2
    T = CMPO(hz * σz / 2, [-1/sqrt(2) * sm, -1/sqrt(2) * sp, -sqrt(Δ) * σz / 2], [1/sqrt(2) * sp, 1/sqrt(2) * sm, sqrt(Δ) * σz / 2], fill(zero2, 3, 3))
    W = Float64[0 -1 0 ; -1 0 0 ; 0 0 -1]
    return T, W
end

function xxz_fm_cmpo(Δ::Real; hz::Real=0.0)
    sp, sm, σz, zero2 = pauli.sp, pauli.sm, pauli.σz, pauli.zero2
    T = CMPO(hz * σz / 2, [1/sqrt(2) * sm, 1/sqrt(2) * sp, -sqrt(Δ) * σz / 2], [1/sqrt(2) * sp, 1/sqrt(2) * sm, sqrt(Δ) * σz / 2], fill(zero2, 3, 3))
    W = Float64[0 1 0 ; 1 0 0 ; 0 0 -1]
    return T, W
end

function heisenberg_j1j2_cmpo(J1::Real, J2::Real)
    Id, sp, sm, σz, zero2 = pauli.Id, pauli.sp, pauli.sm, pauli.σz, pauli.zero2

    Ls = [-J1 / sqrt(2) * sm, -J2 / sqrt(2) * sm, -J1 / sqrt(2) * sp, -J2 / sqrt(2) * sp, -J1 * σz / 2, -J2 * σz / 2]
    Rs = [1 / sqrt(2) * sp, zero2, 1 / sqrt(2) * sm, zero2, σz / 2, zero2 ]
    Ps = fill(zero2, 6, 6)
    Ps[1, 2] = Id 
    Ps[3, 4] = Id 
    Ps[5, 6] = Id

    W = zeros(Float64, 6, 6)
    W0 = [0 -1/J2 ; -1/J2 J1/J2^2]
    W[1:2, 3:4] = W0
    W[3:4, 1:2] = W0
    W[5:6, 5:6] = W0

    T = CMPO(zero2, Ls, Rs, Ps)
    #return T, W
    Λ, U = eigen(W)
    UL = U * diagm(sqrt.(abs.(Λ)))
    UR = diagm(sqrt.(abs.(Λ))) * U'
    Λ0 = diagm(sign.(Λ))
    W - UL * Λ0 * UR |> norm

    Ps = UR * T.Ps * inv(W) * UL * Λ0
    Rs = T.Rs * inv(W) * UL * Λ0
    Ls = UR * T.Ls

    T1 = CMPO(T.Q, Ls, Rs, Ps)
    return T1, Λ0
end

function heisenberg_j1j2_cmpo_deprecated(J1::Real, J2::Real)
    Id, sp, sm, σz, zero2 = pauli.Id, pauli.sp, pauli.sm, pauli.σz, pauli.zero2

    Ls = [-J1 / sqrt(2) * sm, -J2 / sqrt(2) * sm, -J1 / sqrt(2) * sp, -J2 / sqrt(2) * sp, -J1 * σz / 2, -J2 * σz / 2]
    Rs = [1 / sqrt(2) * sp, zero2, 1 / sqrt(2) * sm, zero2, σz / 2, zero2 ]
    Ps = fill(zero2, 6, 6)
    Ps[1, 2] = Id 
    Ps[3, 4] = Id 
    Ps[5, 6] = Id

    W = zeros(Float64, 6, 6)
    W0 = [0 -1/J2 ; -1/J2 J1/J2^2]
    W[1:2, 3:4] = W0
    W[3:4, 1:2] = W0
    W[5:6, 5:6] = W0

    T = CMPO(zero2, Ls, Rs, Ps)
    return T, W
end

"""
    construct cMPO for the rydberg atom chain
        H = -Ω/2 ∑ σ^x - Δ ∑ n + V ∑_{j<l} n_j n_l / (j-l)^6
"""
function rydberg_cmpo(Ω::Real, Δ::Real, V::Real)
    Id, σx, σz, zero2 = pauli.Id, pauli.σx, pauli.σz, pauli.zero2
    n = 0.5 * (σz + Id)
    Q = Ω/2 * σx + Δ * n
    Lcoeffs = [0.9178622276427676, 0.310356218294415, -0.19379426188409468, 0.12643720879864367, -0.0778855036290297]
    Pcoeffs = [0.004072756499721004, 0.06032681098634473, 0.19582380494942775, 0.3920164608078191, 0.6078552227771817]
    Ls = -sqrt(V) .* Lcoeffs .* Ref(n)
    Rs = sqrt(V) .* Lcoeffs .* Ref(n)
    Ps = fill(zero2, 5, 5)
    for ix in 1:5
        Ps[ix, ix] = Pcoeffs[ix] * Id 
    end
    T = CMPO(zero2, Ls, Rs, Ps)
    W = - Matrix{Float64}(I, 5, 5)
    return T, W
end