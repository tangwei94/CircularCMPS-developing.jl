@testset "test multibosoncmps -> cmps backward" for ix in 1:10

    χ, d = 2, 3
    ψ = MultiBosonCMPSData(rand, χ, d)
    ϕn = CMPSData(rand, χ^d, d)
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

    function test_grad(_F, ψ0)
        sψ = similar(ψ0)
        randomize!(sψ)

        α = 1e-5
        gα1 = (_F(ψ0 + α * sψ) - _F(ψ0 - α * sψ)) / (2 * α)
        α = 1e-6
        gα2 = (_F(ψ0 + α * sψ) - _F(ψ0 - α * sψ)) / (2 * α)

        dψ = _F'(ψ0);
        gαa = real(dot(dψ, sψ))
        @test abs(gα1 - gαa) < 1e-5
        @test abs(gα2 - gαa) < 1e-6
    end 

    test_grad(_F1, ψ)
    test_grad(_F2, ψ)
end