# TODO. no need to include L in most of cases.
# consider redefine CircularCMPS -> cMPSdata, and remove the field :L

struct CircularCMPS 
    Q::MPSBondTensor
    Rs::Vector{<:MPSBondTensor}
    L::Real
end

"""
    cmps(f, chi::Integer, d::Integer) -> cmps

    generate a (plain) cmps with vitual dim `chi` and physical dim `d` using function `f` to generate the parameters.
    Example: `cmps(rand, 2, 4)` 
"""
function CircularCMPS(f, χ::Integer, d::Integer, L::Real)
    Q = TensorMap(f, ComplexF64, ℂ^χ, ℂ^χ)
    Rs = MPSBondTensor{ComplexSpace}[]
    for ix in 1:d 
        push!(Rs, TensorMap(f, ComplexF64, ℂ^χ, ℂ^χ))
    end
    return CircularCMPS(Q, Rs, L)
end

# operations on the data. not on the cMPS
Base.:+(ψ::CircularCMPS, ϕ::CircularCMPS) = CircularCMPS(ψ.Q + ϕ.Q, ψ.Rs .+ ϕ.Rs, ψ.L)
Base.:-(ψ::CircularCMPS, ϕ::CircularCMPS) = CircularCMPS(ψ.Q - ϕ.Q, ψ.Rs .- ϕ.Rs, ψ.L)
Base.:*(ψ::CircularCMPS, x::Number) = CircularCMPS(ψ.Q * x, ψ.Rs .* x, ψ.L)
Base.:*(x::Number, ψ::CircularCMPS) = CircularCMPS(ψ.Q * x, ψ.Rs .* x, ψ.L)

function Base.similar(ψ::CircularCMPS) 
    Q = similar(ψ.Q)
    Rs = [similar(R) for R in ψ.Rs]
    return CircularCMPS(Q, Rs, ψ.L)
end

function Base.zero(ψ::CircularCMPS) 
    Q = zero(ψ.Q)
    Rs = [zero(R) for R in ψ.Rs]
    return CircularCMPS(Q, Rs, ψ.L)
end

@inline get_χ(ψ::CircularCMPS) = dim(_firstspace(ψ.Q))
@inline get_d(ψ::CircularCMPS) = length(ψ.Rs)
TensorKit.space(ψ::CircularCMPS) = _firstspace(ψ.Q)

get_matrices(ψ::CircularCMPS) = (ψ.Q, ψ.Rs)

# TODO. define transfer_matrix object as MPSKit.jl did
""" 
    transfer_matrix(ϕ::CircularCMPS, ψ::CircularCMPS) where S<:EuclideanSpace

    The transfer matrix for <ϕ|ψ>.
    target at right vector. 
    T = I + ϵK. returns K<ϕ|ψ>.
    target at right vector. 
    T = I + ϵK. returns K
"""
function transfer_matrix(ϕ::CircularCMPS, ψ::CircularCMPS)
    function fK(v::MPSBondTensor)
        Tv = ψ.Q * v + v * ϕ.Q' 
        for (Rd, Ru) in zip(ψ.Rs, ϕ.Rs)
            Tv += Rd * v * Ru'
        end
        return Tv
    end
    return fK 
end

"""
   transfer_matrix_dagger(ϕ::CircularCMPS{S}, ψ::CircularCMPS{S}) where S<:EuclideanSpace
    
   The Hermitian conjugate of the transfer matrix for <ϕ|ψ>.
   target at left vector. 
   T† = I + ϵK†. returns K†
"""
function transfer_matrix_dagger(ϕ::CircularCMPS, ψ::CircularCMPS) 
    function fK_dagger(v::MPSBondTensor)
        Tv = v * ψ.Q + ϕ.Q' * v 
        for (Rd, Ru) in zip(ψ.Rs, ϕ.Rs)
            Tv += Ru' * v * Rd
        end
        return Tv
    end
    return fK_dagger
end

"""
    left_canonical(ψ::CircularCMPS)

    Convert the input cMPS into the left-canonical form. 
    Return the gauge transformation matrix X and the left-canonicalized cMPS. 
"""
function left_canonical(ψ::CircularCMPS)
    # transfer matrix dagger
    fK_dagger = transfer_matrix_dagger(ψ, ψ)
    
    # solve the fixed-point equation
    init = similar(ψ.Q, _firstspace(ψ.Q)←_firstspace(ψ.Q))
    randomize!(init);
    ws, vls, _ = eigsolve(fK_dagger, init, 1, :LR)
    w = ws[1]
    vl = vls[1]

    # obtain gauge transformation matrix X
    _, s, u = tsvd(vl)
    X = sqrt(s) * u
    Xinv = u' * sqrt(inv(s))

    # update Q and R 
    Q = X * ψ.Q * Xinv - 0.5 * w * id(_firstspace(ψ.Q)) # TODO. better way to normalize? 
    Rs = [X * R * Xinv for R in ψ.Rs]

    return X, CircularCMPS(Q, Rs, ψ.L) 
end

"""
    right_canonical(ψ::CircularCMPS) 

    Convert the input cmps into the right-canonical form. 
    Return the gauge transformation matrix Y and the right-canonicalized cmps. 
"""
function right_canonical(ψ::CircularCMPS)
    # transfer matrix
    fK = transfer_matrix(ψ, ψ)

    # solve the fixed point equation
    init = similar(ψ.Q, _firstspace(ψ.Q)←_firstspace(ψ.Q))
    randomize!(init);
    ws, vrs, _ = eigsolve(fK, init, 1, :LR)
    w = ws[1]
    vr = vrs[1]

    # obtain gauge transformation matrix Yinv
    _, s, u = tsvd(vr)
    Y = sqrt(s) * u
    Yinv = u' * sqrt(inv(s))

    # update Q and R 
    Q = Yinv' * ψ.Q * Y' - 0.5 * w * id(_firstspace(ψ.Q)) # TODO. better way to normalize? 
    Rs = [Yinv' * R * Y' for R in ψ.Rs]
    
    return Y, CircularCMPS(Q, Rs, ψ.L) 
end

"""
    expand(psi::cmps, chi::Integer, perturb::Float64=1e-3) -> cmps

    expand the cmps `psi` to a target bond dimension `chi` by adding small numbers of size `perturb`.
    Only works when the Q, Rs are plain tensors.
"""
function expand(ψ::CircularCMPS, χ::Integer; perturb::Float64=1e-3)
    χ0, d = get_χ(ψ), get_d(ψ)
    if χ <= χ0
        @warn "new χ not bigger than χ0"
        return ψ
    end
    mask = similar(ψ.Q, ℂ^χ ← ℂ^χ)
    fill_data!(mask, randn)
    mask = perturb * mask
    Q = copy(mask)
    Q.data[1:χ0, 1:χ0] += ψ.Q.data
    for ix in χ0+1:χ
        Q.data[ix, ix] += 2*log(perturb)/ψ.L # suppress
    end

    Rs = fill(copy(mask), d)
    for (R, R0) in zip(Rs, ψ.Rs)
        R.data[1:χ0, 1:χ0] += R0.data
    end

    return CircularCMPS(Q, Rs, ψ.L) 
end

"""
    K_mat(ϕ::CircularCMPS, ψ::CircularCMPS)

    calculate the K_mat from two cmpses `phi` and `psi`. order of indices:

        -1 -->--   ϕ'  -->-- -3
                   |
                   ^
                   |
        -2 --<--   ψ   --<-- -4

    such tensor contraction is implemented in `K_otimes`  

    permute as `permute(K, (2, 3), (4, 1))` (implemented as `K_permute`) and get

        -4 -->--   ϕ'  -->-- -2
                   |
                   ^
                   |
        -1 --<--   ψ   --<-- -3
"""
function K_mat(ϕ::CircularCMPS, ψ::CircularCMPS)
    Id_ψ, Id_ϕ = id(space(ψ)), id(space(ϕ)) 
    Kmat = K_otimes(ϕ.Q, Id_ψ) + K_otimes(Id_ϕ, ψ.Q) + sum(K_otimes.(ϕ.Rs, ψ.Rs))
    return Kmat
end

"""
    finite_env(t::TensorMap{ComplexSpace}, L::Real)

    For a cMPS transfer matrix `t` (see `K_mat`), calculate the environment block for the finite length `L`.
    The input `t` matrix should look like 
    ``` 
        -1 -->--  phi' -->-- -3
                   |
                   ^
                   |
        -2 --<--  psi  --<-- -4
    ```
    This function will calculate exp(t) / tr(exp(t)) by diagonalizing `t`.
"""
function finite_env(t::TensorMap{ComplexSpace}, L::Real)
    W, UR = eig(t)
    UL = inv(UR)
    Ws = []

    if W.data isa Matrix 
        Ws = diag(W.data)
    elseif W.data isa TensorKit.SortedVectorDict
        Ws = vcat([diag(values) for (_, values) in W.data]...)
    end
    Wmax = maximum(real.(Ws))
    ln_of_norm = Wmax * L + log(sum(exp.(Ws .* L .- Wmax * L)))

    W = W - (ln_of_norm / L) * id(_firstspace(W)) 
    expW = exp(W * L)
   
    return UR * expW * UL, ln_of_norm
end

"""
    rescale(ψ::CircularCMPS, lnα::Real)

    rescale the cMPS: ψ -> exp(lnα) * ψ
"""
function rescale(ψ::CircularCMPS, lnα::Real)
    Q = ψ.Q + (lnα / 2 / ψ.L) * id(domain(ψ.Q))
    return CircularCMPS(Q, ψ.Rs, ψ.L)
end

"""
    gauge_fixing_proj(ψ::cmps, L::Real; gauge::Symbol=:periodic) -> proj::TensorMap 

    Fix the gauge for a tangent vector tensor. `gauge` can be chosen between `:periodic` and `:left` .
    This function generates a projector `proj` that can be used to
    - parametrize a fixed-gauge tangent vector
    - eleminate the gauge degrees of freedom in an existing tangent vector
"""
function gauge_fixing_proj(ψ::cmps, L::Real; gauge::Symbol=:periodic)
    # tensor `A` is connected to impurity tensor in the tangent vector
    χ = get_chi(ψ)
    A = cmps(id(ℂ^χ), copy(ψ.R))
    A = convert_to_tensormap(A)

    if gauge == :periodic
        # density matrix ρ
        K = K_mat(ψ, ψ)
        ρ = finite_env(K, L)
        ρ = permute(ρ, (2, 3), (4, 1))
   
        # construct the projector
        @tensor Aρ[-1, ; -2, -3] := ρ[-1, 1, -2, 2] * A'[2, 1, -3]
        proj = rightnull(Aρ)'

    elseif gauge == :left
        # left dominant eigenvector 
        lopT = transf_mat_T(ψ, ψ)
        Vl = eigsolve(lopT, TensorMap(rand, ComplexF64, ℂ^χ, ℂ^χ), 1, :LR)[2][1]

        # projector
        @tensor Aρ[-1, ; -2, -3] := Vl'[1, -2] * A'[-1, 1, -3]
        proj = rightnull(Aρ)'
    end
    return proj
end

"""
    tangent_map(ψ::cmps, L::Real; gauge::Symbol=:periodic) -> Function

    tangent map. act on the parameter space obtained after gauge elimination. 
"""
function tangent_map(ψ::cmps, L::Real, p::Real=0; gauge::Symbol=:periodic)

    χ, d = get_chi(ψ), get_d(ψ)

    # diagonalize K matrix, normalize W according to length 
    K = K_mat(ψ, ψ)
    W, UR = eig(K)
    UL = inv(UR)

    # gauge fixing projector
    proj = gauge_fixing_proj(ψ, L; gauge=gauge)    

    # calculate coefficient matrix  
    Wvec = diag(W.data)
    normψ = logsumexp(L .* Wvec)
    Wvec .-= normψ / L
    @tullio coeffW[ix, iy] := theta2(L, Wvec[ix], Wvec[iy] - p*im)
    #coeffW = similar(W)
    #function coeff(a::Number, b::Number)
    #    # when using coeff.(avec, bvec'), avec is column vector, bvec is row vector
    #    # in this way the index ordering is consistent
    #    if a ≈ b'
    #        return L*exp(a*L)
    #    else 
    #        return (exp(a*L) - exp(b'*L)) / (a - b')
    #    end
    #end
    #copyto!(coeffW.data, coeff.(Wvec, (Wvec .- p*im)'))
    
    # A tensor that is connected to G
    A = cmps(id(ℂ^get_chi(ψ)), copy(ψ.R))
    A = convert_to_tensormap(A)

    D = id(ℂ^(d+1))
    D.data[1] = 0

    idW = id(ℂ^(χ^2))

    # calculate the map
    function f(V::TensorMap{ComplexSpace, 1, 1})
        # crossing term
        @tensor M0[-1, -2; -3, -4] := V[1, -4] * proj[-2, 2, 1] * A'[-3, -1, 2]
        M = UL * M0 * UR
        #M = elem_mult(M, coeffW)
        #M = elem_mult_f1(M, (ix, iy) -> theta2(L, Wvec[ix], Wvec[iy] - p*im))
        @tullio M.data[ix, iy] = M.data[ix, iy] * coeffW[ix, iy]
        M = UR * M * UL
        M = permute(M, (2, 3), (4, 1))
        @tensor Vc[-1; -2] := M[1, 3, 2, -2] * A[2, 4, 1] * proj'[-1, 3, 4] 
        # diagonal term
        ρ = UR * exp(W*L - idW*normψ) * UL 
        ρ = permute(ρ, (2, 3), (4, 1))
        @tensor Vd[-1; -2] := ρ[3, 5, 2, -2] * proj[2, 4, 1] * V[1, 3] * D[6, 4] * proj'[-1, 5, 6]
        return L*(Vc + Vd) 
    end

    return f
end
