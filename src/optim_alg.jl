abstract type Algorithm end 

struct CircularCMPSRiemannian <: Algorithm
    maxiter::Int
    tol::Real 
    verbosity::Int
end

function minimize(_f, init::CMPSData, alg::CircularCMPSRiemannian)
    
    function _fg(ϕ::CMPSData)
        fvalue = _f(ϕ)
        ∂ϕ = _f'(ϕ)
        dQ = zero(∂ϕ.Q) 
        dRs = ∂ϕ.Rs .- ϕ.Rs .* Ref(∂ϕ.Q)

        return fvalue, CMPSData(dQ, dRs) 
    end
    function inner(ϕ, ϕ1::CMPSData, ϕ2::CMPSData)
        return real(sum(dot.(ϕ1.Rs, ϕ2.Rs)))
    end
    function retract(ϕ::CMPSData, dϕ::CMPSData, α::Real)
        Rs = ϕ.Rs .+ α .* dϕ.Rs 
        Q = ϕ.Q - α * sum(adjoint.(ϕ.Rs) .* dϕ.Rs) - 0.5 * α^2 * sum(adjoint.(dϕ.Rs) .* dϕ.Rs)
        ϕ1 = CMPSData(Q, Rs)
        return ϕ1, dϕ
    end
    function scale!(dϕ::CMPSData, α::Number)
        dϕ.Q = dϕ.Q * α
        dϕ.Rs .= dϕ.Rs .* α
        return dϕ
    end
    function add!(dϕ::CMPSData, dϕ1::CMPSData, α::Number)
        dϕ.Q += dϕ1.Q * α
        dϕ.Rs .+= dϕ1.Rs .* α
        return dϕ
    end
    function precondition(ϕ::CMPSData, dϕ::CMPSData)
        fK = transfer_matrix(ϕ, ϕ)

        # solve the fixed point equation
        init = similar(ϕ.Q, _firstspace(ϕ.Q)←_firstspace(ϕ.Q))
        randomize!(init);
        _, vrs, _ = eigsolve(fK, init, 1, :LR)
        vr = vrs[1]

        δ = inner(ϕ, dϕ, dϕ)

        P = herm_reg_inv(vr, max(1e-12, 1e-3*sqrt(δ))) 

        Q = dϕ.Q  
        Rs = dϕ.Rs .* Ref(P)

        return CMPSData(Q, Rs)
    end
    transport!(v, x, d, α, xnew) = v
    
    optalg_LBFGS = LBFGS(;maxiter=alg.maxiter, gradtol=alg.tol, verbosity=alg.verbosity)

    init = left_canonical(init)[2] # ensure the input is left canonical

    ψopt, fvalue, grad, numfg, history = optimize(_fg, init, optalg_LBFGS; retract=retract, precondition=precondition, inner=inner, transport! =transport!, scale! =scale!, add! =add!)

    return ψopt, fvalue, grad, numfg, history
end