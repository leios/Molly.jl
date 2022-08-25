# Different ways to simulate molecules

export
    SteepestDescentMinimizer,
    simulate!,
    VelocityVerlet,
    Verlet,
    StormerVerlet,
    Langevin,
    LangevinSplitting,
    TemperatureREMD

"""
    SteepestDescentMinimizer(; <keyword arguments>)

Steepest descent energy minimization.
Not currently compatible with automatic differentiation using Zygote.

# Arguments
- `step_size::D=0.01u"nm"`: the initial maximum displacement.
- `max_steps::Int=1000`: the maximum number of steps.
- `tol::F=1000.0u"kJ * mol^-1 * nm^-1"`: the maximum force below which to
    finish minimization.
- `run_loggers::Bool=false`: whether to run the loggers during minimization.
- `log_stream::L=devnull`: stream to print minimization progress to.
"""
struct SteepestDescentMinimizer{D, F, L}
    step_size::D
    max_steps::Int
    tol::F
    run_loggers::Bool
    log_stream::L
end

function SteepestDescentMinimizer(;
                                    step_size=0.01u"nm",
                                    max_steps=1_000,
                                    tol=1000.0u"kJ * mol^-1 * nm^-1",
                                    run_loggers=false,
                                    log_stream=devnull)
    return SteepestDescentMinimizer(step_size, max_steps, tol,
                                    run_loggers, log_stream)
end

"""
    simulate!(system, simulator, n_steps; n_threads=Threads.nthreads())
    simulate!(system, simulator; n_threads=Threads.nthreads())

Run a simulation on a system according to the rules of the given simulator.
Custom simulators should implement this function.
"""
function simulate!(sys,
                    sim::SteepestDescentMinimizer;
                    n_threads::Integer=Threads.nthreads())
    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    sim.run_loggers && run_loggers!(sys, neighbors, 0; n_threads=n_threads)
    E = potential_energy(sys, neighbors)
    println(sim.log_stream, "Step 0 - potential energy ",
            E, " - max force N/A - N/A")
    hn = sim.step_size

    for step_n in 1:sim.max_steps
        F = forces(sys, neighbors; n_threads=n_threads)
        max_force = maximum(norm.(F))

        coords_copy = sys.coords
        sys.coords += hn * F ./ max_force
        sys.coords = wrap_coords.(sys.coords, (sys.boundary,))

        neighbors_copy = neighbors
        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                    n_threads=n_threads)
        E_trial = potential_energy(sys, neighbors)
        if E_trial < E
            hn = 6 * hn / 5
            E = E_trial
            println(sim.log_stream, "Step ", step_n, " - potential energy ",
                    E_trial, " - max force ", max_force, " - accepted")
        else
            sys.coords = coords_copy
            neighbors = neighbors_copy
            hn = hn / 5
            println(sim.log_stream, "Step ", step_n, " - potential energy ",
                    E_trial, " - max force ", max_force, " - rejected")
        end

        sim.run_loggers && run_loggers!(sys, neighbors, step_n;
                                        n_threads=n_threads)

        if max_force < sim.tol
            break
        end
    end
    return sys
end

# Forces are often expressed per mol but this dimension needs removing for use in the integrator
function remove_molar(x)
    fx = first(x)
    if dimension(fx) == u"𝐋 * 𝐍^-1 * 𝐓^-2"
        T = typeof(ustrip(fx))
        return x / T(Unitful.Na)
    else
        return x
    end
end

"""
    VelocityVerlet(; <keyword arguments>)

The velocity Verlet integrator.

# Arguments
- `dt::T`: the time step of the simulation.
- `coupling::C=NoCoupling()`: the coupling which applies during the simulation.
- `remove_CM_motion::Bool=true`: whether to remove the center of mass motion
    every time step.
"""
struct VelocityVerlet{T, C}
    dt::T
    coupling::C
    remove_CM_motion::Bool
end

function VelocityVerlet(; dt, coupling=NoCoupling(), remove_CM_motion=true)
    return VelocityVerlet(dt, coupling, remove_CM_motion)
end

function simulate!(sys,
                    sim::VelocityVerlet,
                    n_steps::Integer;
                    n_threads::Integer=Threads.nthreads())
    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))

    AT = Array
    if isa(sys.coords, CuArray)
        AT = CuArray
    end

    #accels_t = AT(zeros(length(sys.coords), length(sys.coords[1])))
    accels_t = AT(zero(sys.coords))

    neighbors = compress_neighborlist(find_neighbors(sys, sys.neighbor_finder; parallel=parallel))
    run_loggers!(sys, neighbors, 0; parallel=parallel)
    #accels_t = accelerations(sys, neighbors)
    wait(forces!(accels_t, sys, neighbors, force))
    accels_t_dt = AT(zero(accels_t))

    for step_n in 1:n_steps
        old_coords = copy(sys.coords)
        sys.coords += sys.velocities .* sim.dt .+ (remove_molar.(accels_t) .* sim.dt ^ 2) ./ 2
        
        apply_constraints!(sys, old_coords, sim.dt)
        sys.coords = wrap_coords.(sys.coords, (sys.boundary,))

        #accels_t_dt = accelerations(sys, neighbors; n_threads=n_threads)
        wait(forces!(accels_t_dt, sys, neighbors, force))
        #println(accels_t_dt)

        sys.velocities += remove_molar.(accels_t .+ accels_t_dt) .* sim.dt / 2

        sim.remove_CM_motion && remove_CM_motion!(sys)
        apply_coupling!(sys, sim.coupling, sim)
        
        run_loggers!(sys, neighbors, step_n; n_threads=n_threads)

        if step_n != n_steps
            neighbors = find_neighbors(sys, sys.neighbor_finder,
                                       neighbors, step_n;
                                       n_threads=n_threads)
            neighbors = compress_neighborlist(neighbors)
            accels_t = accels_t_dt
        end
    end

    return sys
end

"""
    Verlet(; <keyword arguments>)

The leapfrog Verlet integrator.
This is a leapfrog integrator, so the velocities are offset by half a time step
behind the positions.

# Arguments
- `dt::T`: the time step of the simulation.
- `coupling::C=NoCoupling()`: the coupling which applies during the simulation.
- `remove_CM_motion::Bool=true`: whether to remove the center of mass motion
    every time step.
"""
struct Verlet{T, C}
    dt::T
    coupling::C
    remove_CM_motion::Bool
end

function Verlet(; dt, coupling=NoCoupling(), remove_CM_motion=true)
    return Verlet(dt, coupling, remove_CM_motion)
end

function simulate!(sys,
                    sim::Verlet,
                    n_steps::Integer;
                    n_threads::Integer=Threads.nthreads())
    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
    sim.remove_CM_motion && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    run_loggers!(sys, neighbors, 0; n_threads=n_threads)

    for step_n in 1:n_steps
        accels_t = accelerations(sys, neighbors; n_threads=n_threads)

        sys.velocities += remove_molar.(accels_t) .* sim.dt
        
        old_coords = copy(sys.coords)
        sys.coords += sys.velocities .* sim.dt
        apply_constraints!(sys, old_coords, sim.dt)
        sys.coords = wrap_coords.(sys.coords, (sys.boundary,))

        sim.remove_CM_motion && remove_CM_motion!(sys)
        apply_coupling!(sys, sim.coupling, sim)

        run_loggers!(sys, neighbors, step_n; n_threads=n_threads)

        if step_n != n_steps
            neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                        n_threads=n_threads)
        end
    end
    return sys
end

"""
    StormerVerlet(; <keyword arguments>)

The Störmer-Verlet integrator.
Does not currently work with coupling methods that alter the velocity.
Does not currently remove the center of mass motion every time step.

# Arguments
- `dt::T`: the time step of the simulation.
- `coupling::C=NoCoupling()`: the coupling which applies during the simulation.
"""
struct StormerVerlet{T, C}
    dt::T
    coupling::C
end

StormerVerlet(; dt, coupling=NoCoupling()) = StormerVerlet(dt, coupling)

function simulate!(sys,
                    sim::StormerVerlet,
                    n_steps::Integer;
                    n_threads::Integer=Threads.nthreads())
    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    run_loggers!(sys, neighbors, 0; n_threads=n_threads)
    coords_last = sys.coords

    for step_n in 1:n_steps
        accels_t = accelerations(sys, neighbors; n_threads=n_threads)

        coords_copy = sys.coords
        if step_n == 1
            # Use the velocities at the first step since there is only one set of coordinates
            sys.coords += sys.velocities .* sim.dt .+ (remove_molar.(accels_t) .* sim.dt ^ 2) ./ 2
        else
            sys.coords += vector.(coords_last, sys.coords, (sys.boundary,)) .+ remove_molar.(accels_t) .* sim.dt ^ 2
        end
        
        apply_constraints!(sys, coords_copy, sim.dt)

        sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
        # This is accurate to O(dt)
        sys.velocities = vector.(coords_copy, sys.coords, (sys.boundary,)) ./ sim.dt

        apply_coupling!(sys, sim.coupling, sim)

        run_loggers!(sys, neighbors, step_n; n_threads=n_threads)

        if step_n != n_steps
            neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                        n_threads=n_threads)
            coords_last = coords_copy
        end
    end
    return sys
end

"""
    Langevin(; <keyword arguments>)

The Langevin integrator, based on the Langevin Middle Integrator in OpenMM.
This is a leapfrog integrator, so the velocities are offset by half a time step
behind the positions.

# Arguments
- `dt::T`: the time step of the simulation.
- `temperature::K`: the equilibrium temperature of the simulation.
- `friction::F`: the friction coefficient of the simulation.
- `remove_CM_motion::Bool=true`: whether to remove the center of mass motion
    every time step.
"""
struct Langevin{S, K, F, T}
    dt::S
    temperature::K
    friction::F
    remove_CM_motion::Bool
    vel_scale::T
    noise_scale::T
end

function Langevin(; dt, temperature, friction, remove_CM_motion=true)
    vel_scale = exp(-dt * friction)
    noise_scale = sqrt(1 - vel_scale^2)
    return Langevin(dt, temperature, friction, remove_CM_motion, vel_scale, noise_scale)
end

function simulate!(sys,
                    sim::Langevin,
                    n_steps::Integer;
                    n_threads::Integer=Threads.nthreads(),
                    rng=Random.GLOBAL_RNG)    
    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
    sim.remove_CM_motion && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    run_loggers!(sys, neighbors, 0; n_threads=n_threads)

    for step_n in 1:n_steps
        accels_t = accelerations(sys, neighbors; n_threads=n_threads)

        sys.velocities += remove_molar.(accels_t) .* sim.dt
        
        old_coords = copy(sys.coords)
        sys.coords += sys.velocities .* sim.dt / 2
        noise = random_velocities(sys, sim.temperature; rng=rng)
        sys.velocities = sys.velocities .* sim.vel_scale .+ noise .* sim.noise_scale

        sys.coords += sys.velocities .* sim.dt / 2
        
        apply_constraints!(sys, old_coords, sim.dt)
        sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
        sim.remove_CM_motion && remove_CM_motion!(sys)

        run_loggers!(sys, neighbors, step_n; n_threads=n_threads)

        if step_n != n_steps
            neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                        n_threads=n_threads)
        end
    end
    return sys
end

"""
    LangevinSplitting(; <keyword arguments>)

The Langevin simulator using a general splitting scheme, consisting of a
succession of **A**, **B** and **O** steps, corresponding respectively to
updates in position, velocity for the potential part, and velocity for the
thermal fluctuation-dissipation part.
The [`Langevin`](@ref) and [`VelocityVerlet`](@ref) simulators without coupling
correspond to the **BAOA** and **BAB** schemes respectively.
For more information on the sampling properties of splitting schemes, see
[Fass et al. 2018](https://doi.org/10.3390/e20050318).
Not currently compatible with automatic differentiation using Zygote.

# Arguments
- `dt::S`: the time step of the simulation.
- `temperature::K`: the equilibrium temperature of the simulation.
- `friction::F`: the friction coefficient. If units are used, it should have a
    dimensionality of mass per time.
- `splitting::W`: the splitting specifier. Should be a string consisting of the
    characters `A`, `B` and `O`. Strings with no `O`s reduce to deterministic
    symplectic schemes.
- `remove_CM_motion::Bool=true`: whether to remove the center of mass motion
    every time step.
"""
struct LangevinSplitting{S, K, F, W}
    dt::S
    temperature::K
    friction::F
    splitting::W
    remove_CM_motion::Bool
end

function LangevinSplitting(; dt, temperature, friction, splitting, remove_CM_motion=true)
    LangevinSplitting{typeof(dt), typeof(temperature), typeof(friction), typeof(splitting)}(
        dt, temperature, friction, splitting, remove_CM_motion)
end

function simulate!(sys,
                    sim::LangevinSplitting,
                    n_steps::Integer;
                    n_threads::Integer=Threads.nthreads(),
                    rng=Random.GLOBAL_RNG)
    M_inv = inv.(masses(sys))
    α_eff = exp.(-sim.friction * sim.dt .* M_inv / count('O', sim.splitting))
    σ_eff = sqrt.((1 * unit(eltype(α_eff))) .- (α_eff .^ 2))

    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
    sim.remove_CM_motion && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    run_loggers!(sys, neighbors, 0; n_threads=n_threads)
    accels_t = accelerations(sys, neighbors; n_threads=n_threads)

    effective_dts = [sim.dt / count(c, sim.splitting) for c in sim.splitting]

    # Determine the need to recompute accelerations before B steps
    forces_known = !occursin(r"^.*B[^B]*A[^B]*$", sim.splitting)

    force_computation_steps = map(collect(sim.splitting)) do op
        if op == 'O'
            return false
        elseif op == 'A'
            forces_known = false
            return false
        elseif op == 'B'
            if forces_known
                return false
            else
                forces_known = true
                return true
            end
        end
    end

    step_arg_pairs = map(enumerate(sim.splitting)) do (j, op)
        if op == 'A'
            return (A_step!, (sys, effective_dts[j]))
        elseif op == 'B'
            return (B_step!, (sys, effective_dts[j], accels_t, force_computation_steps[j], n_threads))
        elseif op == 'O'
            return (O_step!, (sys, α_eff, σ_eff, rng, sim.temperature))
        end
    end

    for step_n in 1:n_steps
        old_coords = copy(sys.coords)
        for (step!, args) in step_arg_pairs
            step!(args..., neighbors)
        end
        
        apply_constraints!(sys, old_coords, sim.dt)
        sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
        sim.remove_CM_motion && remove_CM_motion!(sys)
        run_loggers!(sys, neighbors, step_n)
        
        if step_n != n_steps
            neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                        n_threads=n_threads)
        end
    end
    return sys
end

function O_step!(s, α_eff, σ_eff, rng, temperature, neighbors)
    noise = random_velocities(s, temperature; rng=rng)
    s.velocities = α_eff .* s.velocities + σ_eff .* noise
end

function A_step!(s, dt_eff, neighbors)
    s.coords += s.velocities * dt_eff
    s.coords = wrap_coords.(s.coords, (s.boundary,))
end

function B_step!(s, dt_eff, acceleration_vector, compute_forces::Bool, n_threads::Int, neighbors)
    if compute_forces
        acceleration_vector .= accelerations(s, neighbors, n_threads=n_threads)
    end
    s.velocities += dt_eff * remove_molar.(acceleration_vector)
end

"""
    TemperatureREMD(; <keyword arguments>)

A simulator for a parallel temperature replica exchange (TREX) simulation on a
[`ReplicaSystem`](@ref).
See [Sugita and Okamoto 1999](https://doi.org/10.1016/S0009-2614(99)01123-9).
The corresponding [`ReplicaSystem`](@ref) should have the same number of replicas as
the number of temperatures in the simulator.
When calling [`simulate!`](@ref), the `assign_velocities` keyword argument determines
whether to assign random velocities at the appropriate temperature for each replica.
Not currently compatible with automatic differentiation using Zygote.

# Arguments
- `dt::DT`: the time step of the simulation.
- `temperatures::TP`: the temperatures corresponding to the replicas.
- `simulators::ST`: individual simulators for simulating each replica.
- `exchange_time::ET`: the time interval between replica exchange attempts.
"""
struct TemperatureREMD{N, T, S, DT, TP, ST, ET}
    dt::DT
    temperatures::TP
    simulators::ST
    exchange_time::ET
end

function TemperatureREMD(;
                         dt,
                         temperatures,
                         simulators,
                         exchange_time)
    S = eltype(simulators)
    T = eltype(temperatures)
    N = length(temperatures)
    DT = typeof(dt)
    TP = typeof(temperatures)
    ET = typeof(exchange_time)

    if length(simulators) != length(temperatures)
        throw(ArgumentError("Number of temperatures ($(length(temperatures))) must match " *
                            "number of simulators ($(length(simulators)))"))
    end
    if exchange_time <= dt
        throw(ArgumentError("Exchange time ($exchange_time) must be greater than the time step ($dt)"))
    end

    simulators = Tuple(simulators[i] for i in 1:N)
    ST = typeof(simulators)
    
    return TemperatureREMD{N, T, S, DT, TP, ST, ET}(dt, temperatures, simulators, exchange_time)
end

function simulate!(sys::ReplicaSystem{D, G, T},
                    sim::TemperatureREMD,
                    n_steps::Integer;
                    assign_velocities::Bool=false,
                    rng=Random.GLOBAL_RNG,
                    n_threads::Integer=Threads.nthreads()) where {D, G, T}
    if sys.n_replicas != length(sim.simulators)
        throw(ArgumentError("Number of replicas in ReplicaSystem ($(length(sys.n_replicas))) " *
                "and simulators in TemperatureREMD ($(length(sim.simulators))) do not match."))
    end

    if n_threads > sys.n_replicas
        thread_div = equal_parts(n_threads, sys.n_replicas)
    else # Use 1 thread per replica
        thread_div = equal_parts(sys.n_replicas, sys.n_replicas)
    end

    n_cycles = convert(Int, (n_steps * sim.dt) ÷ sim.exchange_time)
    cycle_length = n_cycles > 0 ? n_steps ÷ n_cycles : 0
    remaining_steps = n_cycles > 0 ? n_steps % n_cycles : n_steps
    n_attempts = 0

    if assign_velocities
        for i in eachindex(sys.replicas)
            random_velocities!(sys.replicas[i], sim.temperatures[i]; rng=rng)
        end
    end

    for cycle in 1:n_cycles
        @sync for idx in eachindex(sim.simulators)
            Threads.@spawn simulate!(sys.replicas[idx], sim.simulators[idx], cycle_length;
                                     n_threads=thread_div[idx])
        end

        # Alternate checking even pairs 2-3/4-5/6-7/... and odd pairs 1-2/3-4/5-6/...
        cycle_parity = cycle % 2
        for n in (1 + cycle_parity):2:(sys.n_replicas - 1)
            n_attempts += 1
            m = n + 1
            if dimension(sys.energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
                k_b = sys.k * T(Unitful.Na)
            else
                k_b = sys.k
            end
            T_n, T_m = sim.temperatures[n], sim.temperatures[m]
            β_n, β_m = inv(k_b * T_n), inv(k_b * T_m)
            neighbors_n = find_neighbors(sys.replicas[n], sys.replicas[n].neighbor_finder;
                                         n_threads=n_threads)
            neighbors_m = find_neighbors(sys.replicas[m], sys.replicas[m].neighbor_finder;
                                         n_threads=n_threads)
            V_n = potential_energy(sys.replicas[n], neighbors_n)
            V_m = potential_energy(sys.replicas[m], neighbors_m)
            Δ = (β_m - β_n) * (V_n - V_m)
            if Δ <= 0 || rand(rng) < exp(-Δ)
                # Exchange coordinates and velocities
                sys.replicas[n].coords, sys.replicas[m].coords = sys.replicas[m].coords, sys.replicas[n].coords
                sys.replicas[n].velocities, sys.replicas[m].velocities = sys.replicas[m].velocities, sys.replicas[n].velocities
                # Scale velocities to new temperature
                sys.replicas[n].velocities .*= sqrt(T_n / T_m)
                sys.replicas[m].velocities .*= sqrt(T_m / T_n)
                if !isnothing(sys.exchange_logger)
                    log_property!(sys.exchange_logger, sys, nothing, cycle*cycle_length;
                                  indices=(n, m), delta=Δ, n_threads=n_threads)
                end
            end
        end
    end

    if remaining_steps > 0
        @sync for idx in eachindex(sim.simulators)
            Threads.@spawn simulate!(sys.replicas[idx], sim.simulators[idx], remaining_steps;
                                     n_threads=thread_div[idx])
        end
    end

    if !isnothing(sys.exchange_logger)
        finish_logs!(sys.exchange_logger; n_steps=n_steps, n_attempts=n_attempts)
    end

    return sys
end

# Calculate k almost equal patitions of n
@inline function equal_parts(n::Int, k::Int)
    ndiv = n ÷ k
    nrem = n % k
    n_parts = ntuple(i -> (i <= nrem) ? ndiv+1 : ndiv, k)
    return n_parts
end
