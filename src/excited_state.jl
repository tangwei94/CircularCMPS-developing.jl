# NOTE. only works for plain tensors

function θ2(L::Real, a::Number, b::Number)
    if a ≈ b
        return L*exp(a*L)
    else 
        return (exp(a*L) - exp(b*L)) / (a - b)
    end
end

function θ3(L::Real, a::Number, b::Number, c::Number)
    if a ≈ b ≈ c 
        return 0.5 * exp(L*b) * L^2
    elseif a ≈ b && !(b ≈ c) 
        return -1 * (exp(L*b) - exp(L*c) - L*exp(L*b)*(b-c) ) / (b - c)^2
    elseif b ≈ c && !(c ≈ a) 
        return -1 * (exp(L*c) - exp(L*a) - L*exp(L*c)*(c-a) ) / (c - a)^2
    elseif c ≈ a && !(a ≈ b) 
        return -1 * (exp(L*a) - exp(L*b) - L*exp(L*a)*(a-b) ) / (a - b)^2
    else
        return (a * (exp(L*b) - exp(L*c)) + b * (exp(L*c) - exp(L*a)) + c * (exp(L*a) - exp(L*b))) / ((a-b)*(b-c)*(c-a))
    end
end

abstract type AbstractCoeffs end

struct Coeff2 <: AbstractCoeffs
    θs::Matrix{<:Number}
    U::TensorMap{ComplexSpace}
    Uinv::TensorMap{ComplexSpace}
end

function Coeff2(T::TensorMap{ComplexSpace}, p1::Real, L::Real)
    Λ, U = eig(T)
    Uinv = inv(U)
    Λs = diag(Λ.data)

    @tullio θs[i1, i2] := θ2(L, Λs[i1] + im*p1, Λs[i2])

    return Coeff2(θs, U, Uinv)
end

function (C2::Coeff2)(A1::TensorMap{ComplexSpace}, A2::TensorMap{ComplexSpace})
    Ã1data = (C2.Uinv * A1 * C2.U).data 
    Ã2data = (C2.Uinv * A2 * C2.U).data 

    @tullio result = C2.θs[i2, i1] * Ã1data[i1, i2] * Ã2data[i2, i1]
    return result
end

struct Coeff3 <: AbstractCoeffs 
    θs::Array{<:Number, 3}
    U::TensorMap{ComplexSpace}
    Uinv::TensorMap{ComplexSpace}
end

function Coeff3(T::TensorMap{ComplexSpace}, p1::Real, p2::Real, L::Real)
    Λ, U = eig(T)
    Uinv = inv(U)
    Λs = diag(Λ.data)

    @tullio θs[i1, i2, i3] := θ3(L, Λs[i1] + im*p1, Λs[i2] + im*p2, Λs[i3])

    return Coeff3(θs, U, Uinv)
end

function (C3::Coeff3)(A1::TensorMap{ComplexSpace}, A2::TensorMap{ComplexSpace}, A3::TensorMap{ComplexSpace})
    Ã1data = (C3.Uinv * A1 * C3.U).data 
    Ã2data = (C3.Uinv * A2 * C3.U).data 
    Ã3data = (C3.Uinv * A3 * C3.U).data 

    @tullio result = C3.θs[i2, i3, i1] * Ã1data[i1, i2] * Ã2data[i2, i3] * Ã3data[i3, i1] 
    return result
end

struct ExcitationData <: AbstractCMPSData
    V::MPSBondTensor
    Ws::Vector{<:MPSBondTensor}
end

function ExcitationData(P::AbstractTensorMap, data::AbstractMatrix)
    χ, d = dims(codomain(P))
    d -= 1
    Pdata = reshape(P.data, χ, d+1, χ)

    V = TensorMap(Pdata[:, 1, :] * data, ℂ^χ, ℂ^χ)
    Ws = [TensorMap(Pdata[:, ix+1, :] * data, ℂ^χ, ℂ^χ) for ix in 1:d] 
    return ExcitationData(V, Ws)
end

function gauge_fixing_map(ψ::CMPSData, L::Real)

    χ = get_χ(ψ)
    d = get_d(ψ)

    K = K_mat(ψ, ψ)
    expK, _ = finite_env(K, L)

    A0 = TensorMap(zeros, ComplexF64, (ℂ^χ)', (ℂ^χ)'*ℂ^(d+1))
    A0data = zeros(ComplexF64, χ, χ, d+1)
    A0data[:, :, 1] = Matrix{ComplexF64}(I, χ, χ)
    A0data[:, :, 2] = conj(ψ.Rs[1].data)

    A0data = reshape(A0data, χ, χ*(d+1))
    A0 = TensorMap(A0data, (ℂ^χ)', (ℂ^χ)'*ℂ^(d+1))

    @tensor T1[-1; -2 -3] := expK[1, -1, 2, -2] * A0[2, 1, -3]
    P = convert(TensorMap, rightnull(T1)')

    return P
end

function effective_N(ψ::CMPSData, p::Real, L::Real)
    P = gauge_fixing_map(ψ, L)
    K = K_mat(ψ, ψ)
    expK, _ = finite_env(K, L)

    χ = get_χ(ψ)

    X = zeros(χ, χ)
    Y = zeros(χ, χ)
    Id = id(ℂ^χ)
    N_mat = zeros(ComplexF64, χ^2, χ^2)

    C2 = Coeff2(K, -p, L)

    for ix in 1:χ^2
        X[(ix-1) ÷ χ + 1, (ix - 1) % χ + 1] = 1
        ϕX = ExcitationData(P, X)
        Polyester.@batch for iy in 1:χ^2
            Y[(iy-1) ÷ χ + 1, (iy - 1) % χ + 1] = 1
            ϕY = ExcitationData(P, Y)
            N_mat[ix, iy] = tr(expK * sum(K_otimes.(ϕX.Ws, ϕY.Ws))) + 
                C2(K_otimes(Id, ϕY.V) + sum(K_otimes.(ψ.Rs, ϕY.Ws)), K_otimes(ϕX.V, Id) + sum(K_otimes.(ϕX.Ws, ψ.Rs))) 
            Y[(iy-1) ÷ χ + 1, (iy - 1) % χ + 1] = 0
        end
        X[(ix-1) ÷ χ + 1, (ix - 1) % χ + 1] = 0
        @printf "N_mat completed %.4f \r" (ix / χ^2) 
    end
    return L*N_mat  
end

function effective_H(ψ::CMPSData, p::Real, L::Real; c=1.0, μ=2.0)
    P = gauge_fixing_map(ψ, L)
    K = K_mat(ψ, ψ)
    expK, _ = finite_env(K, L)

    χ = get_χ(ψ)

    X = zeros(χ, χ)
    Y = zeros(χ, χ)
    Id = id(ℂ^χ)
    H_mat = zeros(ComplexF64, χ^2, χ^2)

    C2a = Coeff2(K, p, L)
    C2z = Coeff2(K, 0, L)
    C2b = Coeff2(K, -p, L)
    C3a = Coeff3(K, 0, p, L)
    C3b = Coeff3(K, 0, -p, L)

    QR_commutator = Ref(ψ.Q) .* ψ.Rs .- ψ.Rs .* Ref(ψ.Q)
    op_Hψ = -μ * sum(K_otimes.(ψ.Rs, ψ.Rs)) + 
            sum(K_otimes.(QR_commutator, QR_commutator)) +
            c * sum(K_otimes.(ψ.Rs .* ψ.Rs, ψ.Rs .* ψ.Rs))

    for ix in 1:χ^2
        X[(ix-1) ÷ χ + 1, (ix - 1) % χ + 1] = 1
        ϕX = ExcitationData(P, X)
        Polyester.@batch for iy in 1:χ^2
            Y[(iy-1) ÷ χ + 1, (iy - 1) % χ + 1] = 1
            ϕY = ExcitationData(P, Y)

            kinY = Ref(ϕY.V) .* ψ.Rs - ψ.Rs .* Ref(ϕY.V) + 
                   Ref(ψ.Q) .* ϕY.Ws - ϕY.Ws .* Ref(ψ.Q) + (im * p) .* ϕY.Ws
            kinX = Ref(ϕX.V) .* ψ.Rs - ψ.Rs .* Ref(ϕX.V) + 
                   Ref(ψ.Q) .* ϕX.Ws - ϕX.Ws .* Ref(ψ.Q) + (im * p) .* ϕX.Ws
            H1 =  -μ * sum(K_otimes.(ϕX.Ws, ϕY.Ws)) + sum(K_otimes.(kinX, kinY)) +
                 c * sum(K_otimes.(ψ.Rs .* ϕX.Ws .+ ϕX.Ws .* ψ.Rs, ψ.Rs .* ϕY.Ws .+ ϕY.Ws .* ψ.Rs)) 

            H_mat[ix, iy] = tr(expK * H1)

            H_mat[ix, iy] += C3a(op_Hψ, 
                                K_otimes(ϕX.V, Id) + sum(K_otimes.(ϕX.Ws, ψ.Rs)),
                                K_otimes(Id, ϕY.V) + sum(K_otimes.(ψ.Rs, ϕY.Ws))
                                )
            
            H_mat[ix, iy] += C3b(op_Hψ, 
                                K_otimes(Id, ϕY.V) + sum(K_otimes.(ψ.Rs, ϕY.Ws)),
                                K_otimes(ϕX.V, Id) + sum(K_otimes.(ϕX.Ws, ψ.Rs))
                                )

            H_mat[ix, iy] += C2z(op_Hψ, sum(K_otimes.(ϕX.Ws, ϕY.Ws))) 
            H_mat[ix, iy] += C2b(-μ * sum(K_otimes.(ψ.Rs, ϕY.Ws)) +
                                sum(K_otimes.(QR_commutator, kinY)) +
                                c * sum(K_otimes.(ψ.Rs .* ψ.Rs, ψ.Rs .* ϕY.Ws + ϕY.Ws .* ψ.Rs)), 
                                K_otimes(ϕX.V, Id) + sum(K_otimes.(ϕX.Ws, ψ.Rs)))
            H_mat[ix, iy] += C2a(-μ * sum(K_otimes.(ϕX.Ws, ψ.Rs)) +
                                sum(K_otimes.(kinX, QR_commutator)) +
                                c * sum(K_otimes.(ψ.Rs .* ϕX.Ws + ϕX.Ws .* ψ.Rs, ψ.Rs .* ψ.Rs)), 
                                K_otimes(Id, ϕY.V) + sum(K_otimes.(ψ.Rs, ϕY.Ws)))

            Y[(iy-1) ÷ χ + 1, (iy - 1) % χ + 1] = 0
        end
        X[(ix-1) ÷ χ + 1, (ix - 1) % χ + 1] = 0
        @printf "H_mat completed %.4f \r" (ix / χ^2) 
    end
    return L*H_mat  
end

function Kac_Moody_gen(ψ::CMPSData, q::Real, L::Real)
    P = gauge_fixing_map(ψ, L)
    K = K_mat(ψ, ψ)
    Id = id(ℂ^get_χ(ψ))
    expK, _ = finite_env(K, L)
    C2 = Coeff2(K, -q, L)

    V = zeros(ComplexF64, χ^2)
    X = zeros(χ, χ)
    for ix in 1:χ^2
        X[(ix-1) ÷ χ + 1, (ix - 1) % χ + 1] = 1
        ϕX = ExcitationData(P, X)

        V[ix] = C2(sum(K_otimes.(ψ.Rs, ψ.Rs)), K_otimes(ϕX.V, Id) + sum(K_otimes.(ϕX.Ws, ψ.Rs)))
        V[ix] += tr(expK * sum(K_otimes.(ϕX.Ws, ψ.Rs)))

        X[(ix-1) ÷ χ + 1, (ix - 1) % χ + 1] = 0
    end
    return L*V
end

function Kac_Moody_gen(ψ::CMPSData, pX::Real, pY::Real, L::Real)

    P = gauge_fixing_map(ψ, L)
    K = K_mat(ψ, ψ)
    expK, _ = finite_env(K, L)

    χ = get_χ(ψ)

    X = zeros(χ, χ)
    Y = zeros(χ, χ)
    Id = id(ℂ^χ)
    N_mat = zeros(ComplexF64, χ^2, χ^2)

    C2a = Coeff2(K, pY, L)
    C2z = Coeff2(K, pY - pX, L)
    C2b = Coeff2(K, -pX, L)
    C3a = Coeff3(K, pY - pX, pY, L)
    C3b = Coeff3(K, pX - pY, -pY, L)

    op_Nψ = sum(K_otimes.(ψ.Rs, ψ.Rs)) 

    for ix in 1:χ^2
        X[(ix-1) ÷ χ + 1, (ix - 1) % χ + 1] = 1
        ϕX = ExcitationData(P, X)
        for iy in 1:χ^2
            Y[(iy-1) ÷ χ + 1, (iy - 1) % χ + 1] = 1
            ϕY = ExcitationData(P, Y)

            N_mat[ix, iy] = tr(expK * sum(K_otimes.(ϕX.Ws, ϕY.Ws))) +
                            C3a(op_Nψ, 
                               K_otimes(ϕX.V, Id) + sum(K_otimes.(ϕX.Ws, ψ.Rs)),
                               K_otimes(Id, ϕY.V) + sum(K_otimes.(ψ.Rs, ϕY.Ws))
                               ) +
                            C3b(op_Nψ, 
                               K_otimes(Id, ϕY.V) + sum(K_otimes.(ψ.Rs, ϕY.Ws)),
                               K_otimes(ϕX.V, Id) + sum(K_otimes.(ϕX.Ws, ψ.Rs))
                               ) +
                            C2z(op_Nψ, sum(K_otimes.(ϕX.Ws, ϕY.Ws))) +
                            C2b(sum(K_otimes.(ψ.Rs, ϕY.Ws)), 
                               K_otimes(ϕX.V, Id) + sum(K_otimes.(ϕX.Ws, ψ.Rs))) +
                            C2a(sum(K_otimes.(ϕX.Ws, ψ.Rs)), 
                               K_otimes(Id, ϕY.V) + sum(K_otimes.(ψ.Rs, ϕY.Ws)))

            Y[(iy-1) ÷ χ + 1, (iy - 1) % χ + 1] = 0
        end 
        X[(ix-1) ÷ χ + 1, (ix - 1) % χ + 1] = 0
    end
    return L*N_mat  

end

## TODO. ψ has to be normalized. Why???
#_, α = finite_env(K_mat(ψ2, ψ2), L)
#ψ2 = rescale(ψ2, -real(α), L)
#
#N1 = effective_N(ψ2, 0, L)
#H1 = effective_H(ψ2, 0, L; c=1.0, μ=2.0)
#
#eigen(Hermitian(N1))

# save data. 
# solve general eigenvalue problem


