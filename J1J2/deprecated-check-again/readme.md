# deprecated-check-again 

This folder contains data obtained with the old (asymmetric) cMPO, which is expected to be suboptimal

- go through all the temperatures by
```julia
α = 2^(1/4)
βs = 1.28 * α .^ (0:23)
```

- data files
    - `dimer_phase_beta$(beta)_cooling.jld2` dimer phase, power method with cooling strategy. not completed. 
    - `dimer_phase_beta$(beta).jld2`. dimer phase, power method with increasing bondD strategy
    - `dimer_phase_blk2_beta$(beta).jld2`. dimer phase, blocked cMPO, power method with increasing bondD strategy

- to get access to the data:
    - you can always check the keys of the jld2 file with `jldload($(name_of_jld2_file))`
    - with cooling strategy, each data file only contains one cMPS `ψ` with flexible bond dimension
    - with increasing bondD strategy, each data file contains a vector `ψs` composed of cMPS's with fixed bond dimensions 3, 6, 9, 12