@with_kw struct VariationalOptim 
    maxiter::Int = 2000
    tol::Real = 1e-12 # The tolerance for fidelity is preferably not greater than 1e-8, otherwise the results will be inaccurate.
    verbosity::Int = 1
end

function leading_boundary(T::CMPO, β::Real, ψ::CMPSData, alg::VariationalOptim)
    printstyled("\n[ variational optimization: 
        β = $(β)
        maxiter = $(alg.maxiter) 
        tol = $(alg.tol)
        verbosity=$(alg.verbosity) 
        Make sure that the cMPO is hermitian! \n"; bold=true, color=:red)

    # target function 
    function _f(ϕ::CMPSData)
        return real(-ln_ovlp(ϕ, T, ϕ, β) + ln_ovlp(ϕ, ϕ, β)) / β
    end

    # optimization 
    optalg = CircularCMPSRiemannian(alg.maxiter, alg.tol, alg.verbosity)
    ψ, F, grad, numfg, history = minimize(_f, ψ, optalg)
    
    f = free_energy(T, ψ, ψ, β)
    E = energy(T, ψ, ψ, β)
    var = variance(T, ψ, β)
    printstyled("[ variational optimization: f, E, var: $(f) $(E) $(var) \n"; color=:red)

    return ψ, f, E, var
end
