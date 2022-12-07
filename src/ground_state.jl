# TODO. construct a type for Hamiltonians

function lieb_liniger_ground_state(c::Real, μ::Real, L::Real, ψ0::Union{CircularCMPS, Nothing}=nothing)
    function fE(ψ::CircularCMPS)
        OH = kinetic(ψ) + c*point_interaction(ψ) - μ * particle_density(ψ)
        expK, _ = finite_env(K_mat(ψ, ψ), L)
        return real(tr(expK * OH))
    end

    # TODO. implement gradientcheck: check inner(d, g) = gradient with respect to alpha obtained from finite difference.
    function fgE(ψ::CircularCMPS)
        E = fE(ψ)
        ∂ψ = fE'(ψ) 
        dQ = zero(∂ψ.Q) #- sum(ψ.Rs' .* ∂ψ.Rs)
        dRs = ∂ψ.Rs .- ψ.Rs .* Ref(∂ψ.Q) #the second term makes sure it is a true gradient! 

        return E, CircularCMPS(dQ, dRs, L)
    end

    function inner(ψ, ψ1::CircularCMPS, ψ2::CircularCMPS)
        return real(dot(ψ1.Q, ψ2.Q) + sum(dot.(ψ1.Rs, ψ2.Rs))) #TODO. clarify the cases with or withou factor of 2. depends on how to define the complex gradient
    end

    function retract(ψ::CircularCMPS, dψ::CircularCMPS, α::Real)
        Rs = ψ.Rs .+ α .* dψ.Rs 
        Q = ψ.Q - α * sum(ψ.Rs' .* dψ.Rs) - 0.5 * α^2 * sum(dψ.Rs' .* dψ.Rs)
        ψ1 = CircularCMPS(Q, Rs, L)
        #ψ1 = left_canonical(ψ1)[2]
        return ψ1, dψ
    end

    function scale!(dψ::CircularCMPS, α::Number)
        dψ.Q = dψ.Q * α
        dψ.Rs .= dψ.Rs .* α
        return dψ
    end

    function add!(dψ::CircularCMPS, dψ1::CircularCMPS, α::Number) 
        dψ.Q += dψ1.Q * α
        dψ.Rs .+= dψ1.Rs .* α
        return dψ
    end

    # only for comparison
    #function no_precondition(ψ::CircularCMPS, dψ::CircularCMPS)
    #    return dψ
    #end

    function precondition(ψ::CircularCMPS, dψ::CircularCMPS)
        fK = transfer_matrix(ψ, ψ)

        # solve the fixed point equation
        init = similar(ψ.Q, _firstspace(ψ.Q)←_firstspace(ψ.Q))
        randomize!(init);
        _, vrs, _ = eigsolve(fK, init, 1, :LR)
        vr = vrs[1]

        δ = inner(ψ, dψ, dψ)

        P = herm_reg_inv(vr, max(1e-12, 1e-3*δ)) 

        Q = dψ.Q  
        Rs = dψ.Rs .* Ref(P)

        return CircularCMPS(Q, Rs, ψ.L)
    end

    transport!(v, x, d, α, xnew) = v

    optalg_LBFGS = LBFGS(;maxiter=1000, gradtol=1e-6, verbosity=2)

    ψ = left_canonical(ψ0)[2]
    ψ1, E, grad, numfg, history = optimize(fgE, ψ, optalg_LBFGS; retract = retract,
                                    precondition = precondition,
                                    inner = inner, transport! =transport!,
                                    scale! = scale!, add! = add!
                                    );
    return ψ1, E, grad, numfg, history 

end

c, μ, L = 1, 2, 16
χ, d = 4, 1
ψ = CircularCMPS(rand, χ, d, L)
ψ1, E, grad, numfg, history = lieb_liniger_ground_state(c, μ, L, ψ)

ψ2 = expand(ψ1, 8)
ψ2, E2, grad, numfg, history = lieb_liniger_ground_state(c, μ, L, ψ2)

ψ3 = expand(ψ2, 12)
ψ3, E3, grad, numfg, history = lieb_liniger_ground_state(c, μ, L, ψ3)
