@testset "test right dominant eigenvector" for ix in 1:10

    χ, d = 4, 2
    ψn = CMPSData(rand, χ, d)
    ϕn = CMPSData(rand, χ, d)

    TM = TransferMatrix(ψn, ϕn)
    K = K_permute(K_mat(ψn, ϕn))

    TM1 = TransferMatrix(ψn, ψn)
    K1 = K_permute(K_mat(ψn, ψn))
    
    TM2 = TransferMatrix(ϕn, ϕn)
    K2 = K_permute(K_mat(ϕn, ϕn))

    # test TransferMatrix itself 
    v = TensorMap(rand, ComplexF64, ℂ^χ, ℂ^χ)
    @test norm(Kact_R(K, v) - TM(v)) < 1e-12
    @test norm(Kact_R(K1, v) - TM1(v)) < 1e-12
    @test norm(Kact_R(K2, v) - TM2(v)) < 1e-12

    # test right_env
    function test_right_env(Kx, TMx)
        λ, vr = right_env(Kx)
        vr1 = right_env(TMx)
        @test norm(TMx(vr) - λ * vr) < 1e-12
        @test norm(TMx(vr1) - λ * vr1) < 1e-12
        @test norm(vr[1] / vr1[1] * vr1 - vr) < 1e-12
    end
    test_right_env(K, TM)
    test_right_env(K1, TM1)
    test_right_env(K2, TM2)

    # test AD for right_env
    function _F1(Kx) 
        vr = right_env(Kx)[2]
        return norm(tr(vr)) / norm(vr)
    end

    function _TMF1(TMx)
        va = right_env(TMx)
        return norm(tr(va)) / norm(va)
    end

    function test_AD_F1_TMF1(Kx0, TM0)
        dK = similar(Kx0)
        randomize!(dK)
        α = 1e-5
        gα1 = (_F1(Kx0 + α * dK) - _F1(Kx0 - α * dK)) / (2 * α)
        α = 1e-6
        gα2 = (_F1(Kx0 + α * dK) - _F1(Kx0 - α * dK)) / (2 * α)

        dTM = _TMF1'(TM0)
        dTMmat = zero(Kx0)
        for (VL, VR) in zip(dTM.VLs, dTM.VRs)
            @tensor dTMmat[-1 -2; -3 -4] += VL[-1; -4] * VR[-2; -3];
        end
        gαa = real(dot(dTMmat, dK))

        @test abs(gα1 - gαa) < 1e-5
        @test abs(gα2 - gαa) < 1e-6
    end

    test_AD_F1_TMF1(K, TM)
    test_AD_F1_TMF1(K1, TM1)
    test_AD_F1_TMF1(K2, TM2)

end

@testset "test left dominant eigenvector" for ix in 1:10
    χ, d = 4, 2
    ψn = CMPSData(rand, χ, d)
    ϕn = CMPSData(rand, χ, d)

    TM = TransferMatrix(ψn, ϕn)
    K = K_permute(K_mat(ψn, ϕn))

    TM1 = TransferMatrix(ψn, ψn)
    K1 = K_permute(K_mat(ψn, ψn))

    TM2 = TransferMatrix(ϕn, ϕn)
    K2 = K_permute(K_mat(ϕn, ϕn))

    # test TransferMatrix itself 
    v = TensorMap(rand, ComplexF64, ℂ^χ, ℂ^χ)
    @test norm(Kact_L(K, v) - flip(TM)(v)) < 1e-12
    @test norm(Kact_L(K1, v) - flip(TM1)(v)) < 1e-12
    @test norm(Kact_L(K2, v) - flip(TM2)(v)) < 1e-12

    # test left_env
    function test_left_env(Kx, TMx)
        λ, vl = left_env(Kx)
        vl1 = left_env(TMx)
        @test norm(flip(TMx)(vl) - λ * vl) < 1e-12
        @test norm(flip(TMx)(vl1) - λ * vl1) < 1e-12
        @test norm(vl[1] / vl1[1] * vl1 - vl) < 1e-12
    end
    test_left_env(K, TM)
    test_left_env(K1, TM1)
    test_left_env(K2, TM2)

    # test AD for left_env
    function _F1(Kx) 
        vl = left_env(Kx)[2]
        return norm(tr(vl)) / norm(vl)
    end

    function _TMF1(TMx)
        va = left_env(TMx)
        return norm(tr(va)) / norm(va)
    end

    function test_AD_F1_TMF1(Kx0, TM0)
        dK = similar(Kx0)
        randomize!(dK)
        α = 1e-5
        gα1 = (_F1(Kx0 + α * dK) - _F1(Kx0 - α * dK)) / (2 * α)
        α = 1e-6
        gα2 = (_F1(Kx0 + α * dK) - _F1(Kx0 - α * dK)) / (2 * α)

        dTM = _TMF1'(TM0)
        dTMmat = zero(Kx0)
        for (VL, VR) in zip(dTM.VLs, dTM.VRs)
            @tensor dTMmat[-1 -2; -3 -4] += VL[-1; -4] * VR[-2; -3];
        end
        gαa = real(dot(dTMmat, dK))

        @test abs(gα1 - gαa) < 1e-5
        @test abs(gα2 - gαa) < 1e-6
    end

    test_AD_F1_TMF1(K, TM)
    test_AD_F1_TMF1(K1, TM1)
    test_AD_F1_TMF1(K2, TM2)

end

@testset "test AD for more complicated cost functions of TM" for ix in 1:10
    χ, d = 4, 2
    ψn = CMPSData(rand, χ, d)
    ϕn = CMPSData(rand, χ, d)
    TM1 = TransferMatrix(ψn, ϕn)
    TM2 = TransferMatrix(ψn, ψn)

    K1 = K_permute(K_mat(ψn, ϕn))
    K2 = K_permute(K_mat(ψn, ψn))

    @test norm(right_env(K1)[1] - left_env(K1)[1]) < 1e-12
    @test norm(right_env(K2)[1] - left_env(K2)[1]) < 1e-12

    _Fend(vl1, vr2) =  norm(tr(vl1)) / norm(vl1) + norm(tr(vr2))/norm(vr2) + norm(tr(vl1 * vr2)) / norm(vl1) / norm(vr2)
    function _F1(Kx)
        vl1 = left_env(Kx)[2]
        vr2 = right_env(Kx)[2]
        return _Fend(vl1, vr2)
    end

    function _F1TM(TMx)
        vl1 = left_env(TMx)
        vr2 = right_env(TMx)
        return _Fend(vl1, vr2)
    end

    function _F2(Kx)
        vl1 = left_env(K1)[2]
        vr2 = right_env(Kx)[2]
        return _Fend(vl1, vr2)
    end

    function _F2TM(TMx)
        vl1 = left_env(TM1)
        vr2 = right_env(TMx)
        return _Fend(vl1, vr2)
    end
    
    function _F3(Kx)
        vl1 = left_env(Kx)[2]
        vr2 = right_env(K1)[2]
        return _Fend(vl1, vr2)
    end

    function _F3TM(TMx)
        vl1 = left_env(TMx)
        vr2 = right_env(TM1)
        return _Fend(vl1, vr2)
    end

    function test_AD_F1_TMF1(_Fy, _FyTM, Kx0, TM0)
        dK = similar(Kx0)
        randomize!(dK)
        α = 1e-5
        gα1 = (_Fy(Kx0 + α * dK) - _Fy(Kx0 - α * dK)) / (2 * α)
        α = 1e-6
        gα2 = (_Fy(Kx0 + α * dK) - _Fy(Kx0 - α * dK)) / (2 * α)

        dTM = _FyTM'(TM0)
        dTMmat = zero(Kx0)
        for (VL, VR) in zip(dTM.VLs, dTM.VRs)
            @tensor dTMmat[-1 -2; -3 -4] += VL[-1; -4] * VR[-2; -3];
        end
        gαa = real(dot(dTMmat, dK))

        @test abs(gα1 - gαa) < 1e-5
        @test abs(gα2 - gαa) < 1e-6
    end
    test_AD_F1_TMF1(_F1, _F1TM, K1, TM1)
    test_AD_F1_TMF1(_F1, _F1TM, K2, TM2)
    test_AD_F1_TMF1(_F2, _F2TM, K1, TM1)
    test_AD_F1_TMF1(_F2, _F2TM, K2, TM2)
    test_AD_F1_TMF1(_F3, _F3TM, K1, TM1)
    test_AD_F1_TMF1(_F3, _F3TM, K2, TM2)

end

@testset "test AD for TransferMatrix constructor" for ix in 1:10
    
    χ, d = 4, 2
    ψn = CMPSData(rand, χ, d)
    ϕn = CMPSData(rand, χ, d)

    function _F1(ψ)
        TM1 = TransferMatrix(ϕn, ψ)
        TM2 = TransferMatrix(ψn, ϕn)
        vl1 = left_env(TM1)
        vr2 = right_env(TM2)
   
        return norm(tr(vl1)) / norm(vl1) + norm(tr(vr2))/norm(vr2) + norm(tr(vl1 * vr2)) / norm(vl1) / norm(vr2)
    end
    function _F2(ψ)
        TM1 = TransferMatrix(ϕn, ψn)
        TM2 = TransferMatrix(ψ, ψn)
        vl1 = left_env(TM1)
        vr2 = right_env(TM2)
   
        return norm(tr(vl1)) / norm(vl1) + norm(tr(vr2))/norm(vr2) + norm(tr(vl1 * vr2)) / norm(vl1) / norm(vr2)
    end

    function test_grad(_F, ψ0)
        sψ = similar(ψ0)
        randomize!(sψ.Q)
        for ix in 1:d
            randomize!(sψ.Rs[ix])
        end

        α = 1e-5
        gα1 = (_F(ψ0 + α * sψ) - _F(ψ0 - α * sψ)) / (2 * α)
        α = 1e-6
        gα2 = (_F(ψ0 + α * sψ) - _F(ψ0 - α * sψ)) / (2 * α)

        dψ = _F'(ψ0);
        gαa = real(dot(dψ, sψ))
        @test abs(gα1 - gαa) < 1e-5
        @test abs(gα2 - gαa) < 1e-6
    end 

    test_grad(_F1, ψn)
    test_grad(_F2, ψn)
    test_grad(_F1, ϕn)
    test_grad(_F2, ϕn)
end