struct Parallel <: PairwiseInteraction
    nl_only::Bool
end

Parallel(; nl_only=false) = parallel(nl_only)

@inline @inbounds function force(inter::Parallel,
                                 dr,
                                 coord_i,
                                 coord_j,
                                 atom_i,
                                 atom_j,
                                 boundary)
    println(dr)
    return SVector((0.1, 0.1))
end

#@testset "Parallel Precision" begin
    atoms = [Atom(mass=1.0), Atom(mass=1.00)]
    coords = [SVector(0.3, 0.5), SVector(0.7, 0.5)]
    velocities = [SVector(0.0, 0.0), SVector(0.0, 0.0)]
    pairwise_inters = (Parallel(false),)

    simulator = VelocityVerlet(dt=0.0002)
    boundary = RectangularBoundary(1.0, 1.0)

    sys = System(
        atoms=atoms,
        pairwise_inters=pairwise_inters,
        coords=coords,
        velocities=velocities,
        boundary=boundary,
        loggers=(coords=CoordinateLogger(Float64, 10; dims=2),),
        force_units=NoUnits,
        energy_units=NoUnits,
    )

    simulate!(sys, simulator, 25)

    @test isapprox(vector(coords[1], coords[2], boundary),
                   vector(sys.coords[1], sys.coords[2], boundary))

#end
