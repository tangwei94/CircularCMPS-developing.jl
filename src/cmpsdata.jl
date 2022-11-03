struct cMPSdata 
    Q::MPSBondTensor
    Rs::Vector{<:MPSBondTensor}
end

Base.:+(ψdata::cMPSdata, ϕdata::cMPSdata) = cMPSdata(ψdata.Q + ϕdata.Q, ψdata.Rs .+ ϕdata.Rs)
Base.:-(ψdata::cMPSdata, ϕdata::cMPSdata) = cMPSdata(ψdata.Q - ϕdata.Q, ψdata.Rs .- ϕdata.Rs)
Base.:*(ψdata::cMPSdata, x::Number) = cMPSdata(ψdata.Q * x, ψdata.Rs .* x)
Base.:*(x::Number, ψdata::cMPSdata) = cMPSdata(ψdata.Q * x, ψdata.Rs .* x)
