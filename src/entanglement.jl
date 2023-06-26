function entanglement_entropy(ψ::CMPSData, β::Real)
    MA = permute(finite_env(ψ*ψ, β/2)[1], (1, 3), (2, 4))
    MB = permute(finite_env(convert(TensorMap, (ψ*ψ)'), β/2)[1], (1, 3), (2, 4))

    ρA = sqrt(MA) * MB * sqrt(MA)
    ρA = ρA / tr(ρA)

    ΛA, _ = eigen(ρA)
    SA = - real(tr(ΛA * log(ΛA)))

    return SA 
end

function suggest_χ(ψ::CMPSData, β::Real; tol::Real=1e-9, maxχ::Int=32, minχ::Int=2)
    MA = permute(finite_env(ψ*ψ, β/2)[1], (1, 3), (2, 4))
    MB = permute(finite_env(convert(TensorMap, (ψ*ψ)'), β/2)[1], (1, 3), (2, 4))

    ρA = sqrt(MA) * MB * sqrt(MA)
    ρA = ρA / tr(ρA)

    ΛA, _ = eigen(ρA)
    SA = reverse(norm.(diag(ΛA.data)))
    χ1 = min(floor(Int, sqrt(sum(SA .> tol))), maxχ)
    χ1 = max(χ1, minχ)
    err = 0 
    if χ1^2 < length(SA)
        err = SA[χ1^2 + 1]
    end

    return χ1, err 
end