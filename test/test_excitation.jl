ψ = CMPSData(rand, 2, 1)

L = 10

χ = get_χ(ψ)
Id = id(ℂ^χ)
K = K_mat(ψ, ψ)
expK, α = finite_env(K, L)

ψ = rescale(ψ, -real(α), L)
K = K_mat(ψ, ψ)
expK, α = finite_env(K, L)

A1 = similar(K); fill_data!(A1, rand)
A2 = similar(K); fill_data!(A2, rand)
A3 = similar(K); fill_data!(A3, rand)

# test 1 for Coeff2
C2 = Coeff2(K, 0, L)
integral1 = C2(A1, A2)

integral2, err2 = quadgk(τ -> tr(A1 * exp(K*τ) * A2 * exp(K*(L-τ))), 0, L)

@test abs(integral1 - integral2) < err2

# test 2 for Coeff2
C2 = Coeff2(K, 2*pi/L, L)
integral1 = C2(A1, A2)

Id_K = id(domain(K))
integral2, err2 = quadgk(τ -> tr(A1 * exp((K+im*2*pi/L*Id_K)*τ) * A2 * exp(K*(L-τ))), 0, L)

@test abs(integral1 - integral2) < err2

# test 3 for Coeff2. 
# TODO. check d_ph > 1
P = gauge_fixing_map(ψ, L)
ϕX = ExcitationData(P, rand(χ, χ))

@test abs(tr(expK * (K_otimes(ϕX.V, Id) + sum(K_otimes.(ϕX.Ws, ψ.Rs))) )) < 1e-14
@test abs(tr(expK * (K_otimes(Id, ϕX.V) + sum(K_otimes.(ψ.Rs, ϕX.Ws))) )) < 1e-14

N_mat = effective_N(ψ, 0, L)
WN, _ = eigen(Hermitian(N_mat))
@test minimum(real.(WN)) > -1e-16

# test 1 for Coeff3
C3 = Coeff3(K, 0, 0, L)
integral1 = C3(A1, A2, A3)

integral2, err2 = quadgk(τ2 ->
    quadgk(τ1 -> tr(A1 * exp(K*τ1) * A2 * exp(K*(τ2-τ1)) * A3 * exp(K*(L-τ2)) ), 0, τ2)[1],
    0, L
)

@test abs(integral1 - integral2) < err2

# test 2 for Coeff3
Id_K = id(domain(K))
p1, p2 = 4*pi/L, -4*pi/L
C3 = Coeff3(K, p1, p2, L)
integral1 = C3(A1, A2, A3)

integral2, err2 = quadgk(τ2 ->
    quadgk(τ1 -> tr(A1 * exp((K+im*p1*Id_K)*τ1) * A2 * exp((K+im*p2*Id_K)*(τ2-τ1)) * A3 * exp(K*(L-τ2)) ), 0, τ2)[1],
    0, L
)

@test abs(integral1 - integral2) < err2

# test excitation energies
# TODO. ψ has to be normalized. Why???
_, α = finite_env(K_mat(ψ2, ψ2), L)
ψ2 = rescale(ψ2, -real(α), L)

ψ = ψ2
function fE(ψ::CMPSData)
    OH = kinetic(ψ) + c*point_interaction(ψ) - μ * particle_density(ψ)
    expK, _ = finite_env(K_mat(ψ, ψ), L)
    return real(tr(expK * OH))
end
@show fE(ψ)

N1 = effective_N(ψ, -2*pi/L, L)
H1 = effective_H(ψ, -2*pi/L, L; c=c, μ=μ)

@assert norm(H1 - H1') < 1e-10
@assert norm(N1 - N1') < 1e-10

H̃1 = sqrt(inv(N1)) * H1 * sqrt(inv(N1))
Es, _ = eigen(Hermitian(H̃1))

Es ./ L
