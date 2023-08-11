@testset "test basic utility functions for MultiBosonCMPSData" for ix in 1:10
    ψa = MultiBosonCMPSData(rand, 2, 3)

    @test norm(ψa - ψa) < 1e-12
    @test norm(ψa) / norm(2*ψa) ≈ 0.5
    @test norm(ψa) / norm(ψa + ψa * 0.5) ≈ 2/3
    @test norm(ψa) / norm(ψa - ψa * 0.5) ≈ 2

    ψb = similar(ψa)
    randomize!(ψb)
    @test get_χ(ψb) == 2 
    @test get_d(ψb) == 3 

    @test norm(CMPSData(ψa)) ≈ norm(ψa) 
end

@testset "test multibosoncmps <-> cmps " for ix in 1:10

    χ, d = 2, 3
    ψ = MultiBosonCMPSData(rand, χ, d)
    ϕn = CMPSData(rand, χ, d)
    function _F1(ψ)
        ψn = CMPSData(ψ)

        TM1 = TransferMatrix(ϕn, ψn)
        TM2 = TransferMatrix(ψn, ϕn)
        vl1 = left_env(TM1)
        vr2 = right_env(TM2)
   
        return norm(tr(vl1)) / norm(vl1) + norm(tr(vr2))/norm(vr2) + norm(tr(vl1 * vr2)) / norm(vl1) / norm(vr2)
    end
    function _F2(ψ)
        ψn = CMPSData(ψ)

        TM1 = TransferMatrix(ϕn, ψn)
        TM2 = TransferMatrix(ψn, ψn)
        vl1 = left_env(TM1)
        vr2 = right_env(TM2)
   
        return norm(tr(vl1)) / norm(vl1) + norm(tr(vr2))/norm(vr2) + norm(tr(vl1 * vr2)) / norm(vl1) / norm(vr2)
    end

    test_ADgrad(_F1, ψ)
    test_ADgrad(_F2, ψ)
end

@testset "test Kmat_pseudo_inv by numerical integration" for ix in 1:10
    χ, d = 8, 2
    ψ = MultiBosonCMPSData(rand, χ, d)

    ψn = CMPSData(ψ);
    K = K_permute(K_mat(ψn, ψn));
    λ, EL = left_env(K);
    λ, ER = right_env(K);
    Kinv = Kmat_pseudo_inv(K, λ);

    VL = Tensor(rand, ComplexF64, (ℂ^χ)'⊗ℂ^χ)
    VR = Tensor(rand, ComplexF64, (ℂ^χ)'⊗ℂ^χ)

    IdK = K_permute(id((ℂ^χ)'⊗ℂ^χ))
    K_nm = K - λ * IdK
    K0_nm = K_permute_back(K_nm)

    Kinf = exp(1e4*K0_nm)
    @test norm(Kinf) ≈ 1/norm(tr(EL * ER))

    Λ0, U0 = eigen(K0_nm)
    @test norm(exp(12*K0_nm) - U0 * exp(12*Λ0) * inv(U0)) < 1e-12
    
    δ = isometry(ℂ^(χ^2), ℂ^(χ^2-1))
    Λr, Ur, invUr = δ' * Λ0 * δ, U0 * δ, δ' * inv(U0)

    a1, err = quadgk(τ -> tr(VL' * Ur * exp(τ * Λr) * invUr * VR), 0, 1e4)
    a2 = tr(VL' * K_permute_back(Kinv) * VR)
    @test norm(a1 - a2) < 100 * err

end

@testset "tangent_map should be hermitian and positive definite" for ix in 1:10
    χ, d = 8, 2
    ψ = MultiBosonCMPSData(rand, χ, d)

    ψn = CMPSData(ψ);
    K = K_permute(K_mat(ψn, ψn));
    λ, EL = left_env(K);
    λ, ER = right_env(K);
    Kinv = Kmat_pseudo_inv(K, λ);

    M = zeros(ComplexF64, χ^2+d*χ, χ^2+d*χ)
    for ix in 1:(χ^2+d*χ)
        v = zeros(χ^2+d*χ)
        v[ix] = 1

        X = MultiBosonCMPSData(v, χ, d)
        v1 = vec(tangent_map(ψ, X, EL, ER, Kinv))
        M[:, ix] = v1
        @show norm(v1)
    end

    @test norm(M - M') < 1e-12
    a, _ = eigen(Hermitian(M))
    @test a[1] > -1e-12
end
