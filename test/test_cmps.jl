χ, d, L = 4, 2, 8
ϕ = CircularCMPS(rand, χ, d, L)

@test get_χ(ϕ) == χ
@test get_d(ϕ) == d

X, ϕL = left_canonical(ϕ)
KL = ϕL.Q + ϕL.Q'
for R in ϕL.Rs 
    KL += R' * R
end
@test norm(KL) / χ^2 < 1e-10

Y, ϕR = right_canonical(ϕ)
KR = ϕR.Q + ϕR.Q'
for R in ϕR.Rs 
    KR += R * R'
end
@test norm(KR) / χ^2 < 1e-10

[expand(ϕ, 12).Q.data[ix, ix] for ix in 1:12]

ψ = expand(ϕ, 6)

fwd, K_mat_pushback = rrule(K_mat, ϕ, ψ)
K_mat_pushback(fwd)

𝕂 = transfer_matrix(ϕ, ψ)
𝕂dag = transfer_matrix_dagger(ϕ, ψ)

Kmat = K_mat(ϕ, ψ) |> K_permute

vr = similar(Kmat, space(ψ) ← space(ϕ)) 
randomize!(vr)
@tensor Kvr[-1; -2] := Kmat[-1, 1, 2, -2] * vr[2, 1]
@test norm(Kvr - 𝕂(vr)) < 1e-10

vl = similar(Kmat, space(ϕ) ← space(ψ)) 
randomize!(vl)
@tensor Kvl[-1; -2] := Kmat[1, -1, -2, 2] * vl[2, 1] 
@test norm(Kvl - 𝕂dag(vl)) < 1e-10

Kmat = K_permute_back(Kmat)
finite_env(Kmat, L);

IK = id(domain(Kmat))
expK, C = finite_env(Kmat, L)

Kmat0 = Kmat - (C/L) * IK
@test norm(exponentiate(x -> Kmat0 * x, L, IK)[1] - expK) < 1e-12

_, nm = finite_env(K_mat(ψ, ψ), ψ.L)
ψ1 = rescale(ψ, -real(nm))

_, nm1 = finite_env(K_mat(ψ1, ψ1), ψ1.L)

# for test
#function finite_env_pushback(ēxpK, l̄n_norm)
#    f̄wd = ēxpK
#    function coeff(a::Number, b::Number) 
#        if a ≈ b
#            return L*exp(a*L)
#        else 
#            return (exp(a*L) - exp(b*L)) / (a - b)
#        end
#    end
#    M = UR' * f̄wd * UL'
#    M1 = similar(M)
#    copyto!(M1.data, M.data .* coeff.(Wvec', conj.(Wvec)))
#    t̄ = UL' * M1 * UR' - L * tr(f̄wd * fwd') * fwd'
#    
#    return NoTangent(), t̄, NoTangent()
#end 

OH = kinetic(ψ) + point_interaction(ψ) - particle_density(ψ)
expK, _ = finite_env(K_mat(ψ, ψ), ψ.L)
real(tr(expK * OH))

function fE(ψ::CircularCMPS)
    OH = kinetic(ψ) + point_interaction(ψ) - particle_density(ψ)
    expK, _ = finite_env(K_mat(ψ, ψ), ψ.L)
    return real(tr(expK * OH))
end

fE(ψ)
dψ = fE'(ψ)

Qr = similar(ψ.Q)
randomize!(Qr)
Rrs = MPSBondTensor[]
for R in ψ.Rs
    Rr = similar(R)
    randomize!(Rr)
    push!(Rrs, Rr)
end
ψr = CircularCMPS(Qr, Rrs, ψ.L)
α = 1e-6

(fE(ψ + α*ψr) - fE(ψ - α*ψr)) / (2*α)
dQ, dRs = get_matrices(dψ)

real(dot(dQ, Qr) + sum(dot.(dRs, Rrs)))