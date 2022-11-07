function ground_state()

end

function fE(ψ::CircularCMPS)
    OH = kinetic(ψ) + point_interaction(ψ) - 2* particle_density(ψ)
    expK, ln_nm = finite_env(K_mat(ψ, ψ), ψ.L)
    return real(tr(expK * OH))
end

# TODO. implement gradientcheck: check inner(d, g) = gradient with respect to alpha obtained from finite difference.
function fgE(ψ::CircularCMPS)
    E = fE(ψ)
    ∂ψ = fE'(ψ) 
    dQ = zero(∂ψ.Q)#- sum(ψ.Rs' .* ∂ψ.Rs)
    dRs = ∂ψ.Rs .- ψ.Rs .* Ref(∂ψ.Q) #the second term makes sure it is a true gradient! 

    return E, CircularCMPS(dQ, dRs, ψ.L)
end

χ, d, L = 4, 1, 16
ψ = CircularCMPS(rand, χ, d, L)
ψ = left_canonical(ψ)[2]
E, ∂ψ = fgE(ψ)

# TODO. modify finalize!. check cmps norm after each step.
# TODO. compare with periodic gauge result. 
# TODO. in periodic gauge: everything projected out?, check the inner(g, d) relation. 

function inner(ψ, ψ1::CircularCMPS, ψ2::CircularCMPS)
    return real(dot(ψ1.Q, ψ2.Q) + sum(dot.(ψ1.Rs, ψ2.Rs))) #TODO. clarify the cases with or withou factor of 2. depends on how to define the complex gradient
end

function retract(ψ::CircularCMPS, dψ::CircularCMPS, α::Real)
    Rs = ψ.Rs .+ α .* dψ.Rs 
    Q = ψ.Q - α * sum(ψ.Rs' .* dψ.Rs) - 0.5 * α^2 * sum(dψ.Rs' .* dψ.Rs)
    ψ1 = CircularCMPS(Q, Rs, ψ.L)
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

function no_precondition(ψ::CircularCMPS, dψ::CircularCMPS)
    return dψ
end

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

dψ = precondition(ψ, ∂ψ)
ψ1 = retract(ψ, dψ, 1e-1)[1]
left_canonical(ψ1)

fE(ψ1) 
optalg_LBFGS = LBFGS(;maxiter=1000, gradtol=1e-6, verbosity=2)

ψ1, ℰ, grad, numfg, history = optimize(fgE, ψ, optalg_LBFGS; retract = retract,
                                precondition = precondition,
                                inner = inner, transport! =transport!,
                                scale! = scale!, add! = add!
                                );

#_, _, _, _, history_no_precond = optimize(fgE, ψ, optalg_LBFGS; retract = retract,
#                                precondition = no_precondition,
#                                inner = inner, transport! =transport!,
#                                scale! = scale!, add! = add!
#                                );

ψ_χ8 = expand(ψ1, 8; perturb=1e-3)
ψ_χ8 = left_canonical(ψ_χ8)[2]
ψ1_χ8,  = optimize(fgE, ψ_χ8, optalg_LBFGS; retract = retract,
                                precondition = precondition,
                                inner = inner, transport! =transport!,
                                scale! = scale!, add! = add!
                                );

expK_χ8, ln_nm_χ8 = finite_env(K_mat(ψ1_χ8, ψ1_χ8), ψ.L)
ln_nm_χ8

ψ_χ12 = expand(ψ1_χ8, 12; perturb=1e-3)
ψ_χ12 = left_canonical(ψ_χ12)[2]
ψ1_χ12,  = optimize(fgE, ψ_χ12, optalg_LBFGS; retract = retract,
                                precondition = precondition,
                                inner = inner, transport! =transport!,
                                scale! = scale!, add! = add!
                                );

expK_χ12, ln_nm_χ12 = finite_env(K_mat(ψ1_χ12, ψ1_χ12), ψ.L)
ln_nm_χ12

ψ_χ16 = expand(ψ1_χ12, 16; perturb=1e-3)
ψ_χ16 = left_canonical(ψ_χ16)[2]
ψ1_χ16,  = optimize(fgE, ψ_χ16, optalg_LBFGS; retract = retract,
                                precondition = precondition,
                                inner = inner, transport! =transport!,
                                scale! = scale!, add! = add!
                                );

expK_χ16, ln_nm_χ16 = finite_env(K_mat(ψ1_χ16, ψ1_χ16), ψ.L)
ln_nm_χ16