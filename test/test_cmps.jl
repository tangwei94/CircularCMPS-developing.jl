@testset "basics" begin
    χ, d, L = 4, 1, 8
    ϕ = CMPSData(rand, χ, d)

    @test get_χ(ϕ) == χ
    @test get_d(ϕ) == d
end

@testset "canonical forms" for ix in 1:10
    χ, d, L = 4, 2, 8
    ϕ = CMPSData(rand, χ, d)

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

end

@testset "fidelity for l/r canonical form and expand" for ix in 1:10
    χ, d, L = 4, 2, 8
    ϕ = CMPSData(rand, χ, 2)
    ϵ = 1e-16
    ϕ2 = expand(ϕ, 6, L; perturb=ϵ)
    ϕl = left_canonical(ϕ)[2]
    ϕr = right_canonical(ϕ)[2]
    # TODO. function ln_ovlp
    function flnovlp(a, b, L)
        _, α = finite_env(K_mat(a, b), L) 
        return α
    end
    fidel2 = flnovlp(ϕ, ϕ2, L) + flnovlp(ϕ2, ϕ, L) - flnovlp(ϕ, ϕ, L) - flnovlp(ϕ2, ϕ2, L)
    @test norm(fidel2) < 1e-10
    fidel_L = flnovlp(ϕ, ϕl, L) + flnovlp(ϕl, ϕ, L) - flnovlp(ϕl, ϕl, L) - flnovlp(ϕ, ϕ, L)
    @test norm(fidel_L) < 1e-10
    fidel_R = flnovlp(ϕ, ϕr, L) + flnovlp(ϕr, ϕ, L) - flnovlp(ϕr, ϕr, L) - flnovlp(ϕ, ϕ, L)
    @test norm(fidel_R) < 1e-10
end

@testset "transfer matrix object" for ix in 1:10
    χ, d, L = 4, 2, 8
    ϕ = CMPSData(rand, χ, d)
    ψ = CMPSData(rand, χ, d)

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
end

@testset "finite_env, rescale" for ix in 1:10

    χ, d, L = 4, 2, 8
    ϕ = CMPSData(rand, χ, d)
    ψ = CMPSData(rand, χ, d)
    Kmat = K_mat(ϕ, ψ) 
    
    IK = id(domain(Kmat))
    expK, C = finite_env(Kmat, L);
    Kmat0 = Kmat - (C/L) * IK

    @test norm(exponentiate(x -> Kmat0 * x, L, IK)[1] - expK) / χ^2 < 1e-12

    _, nm = finite_env(K_mat(ψ, ψ), L)
    ψ1 = rescale(ψ, -real(nm), L)
    _, nm1 = finite_env(K_mat(ψ1, ψ1), L)
    @test norm(nm1) < 1e-10

end

@testset "energy gradient" for ix in 1:10
    function fE(ψ::CMPSData)
        OH = kinetic(ψ) + point_interaction(ψ) - particle_density(ψ)
        expK, _ = finite_env(K_mat(ψ, ψ), L)
        return real(tr(expK * OH))
    end

    χ, d, L = 4, 2, 8
    ψ = CMPSData(rand, χ, d)
    dψ = fE'(ψ)
    ψr = CMPSData(rand, χ, d) 

    α = 1e-6
    grad_fin = (fE(ψ + α*ψr) - fE(ψ - α*ψr)) / (2*α)

    dQ, dRs = get_matrices(dψ)
    Qr, Rrs = get_matrices(ψr)
    grad_auto = real(dot(dQ, Qr) + sum(dot.(dRs, Rrs)))

    @test norm(grad_fin - grad_auto) < α
end

#ϕ = ψ
#E, gϕ = fgE(ϕ)
#dϕ = precondition(ϕ, gϕ)
#α = 1e-3
#Δα = 1e-6
#ϕ0, _ = retract(ϕ, dϕ, α)
#ϕ1, _ = retract(ϕ, dϕ, α-Δα)
#ϕ2, _ = retract(ϕ, dϕ, α+Δα)
#
#E0, gϕ0 = fgE(ϕ0)
#E1, gϕ1 = fgE(ϕ1)
#E2, gϕ2 = fgE(ϕ2)
#
#(E2 - E1) / (2*Δα)
#
#inner(ϕ0, gϕ0, dϕ)