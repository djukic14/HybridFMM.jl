using NURBS
using BEAST
using Test
using H2Factory
using LinearAlgebra
using H2FactoryTestUtils
using iFMM
using MLFMA
using SphericalScattering
using HybridFMM
using StaticArrays
using SumTypes

BEASTnurbs = Base.get_extension(BEAST, :BEASTnurbs)
H2NURBS = Base.get_extension(H2Factory, :H2NURBS)
iFMMNURBS = Base.get_extension(iFMM, :iFMMNURBS)

@testset "Hybrid aggregation" begin
    fns = [joinpath(pkgdir(iFMM), "test", "assets", "sphere.dat")]
    m = readMultipatch(fns[1])

    p = 1
    N = 2^3 + p
    X = BsplineBasisDiv(m, p, N)
    X = superspace(X; interfacesonly=false)

    originaltree = ParametricBoundingBallTree(X, 1 / N)

    polyp = 4
    polynomial = iFMM.BarycentricLagrangePolynomial2DEquidistant(polyp)

    λ = originaltree.radius / 1
    k = 2π / λ
    operator = iFMM.MWHyperSingular3D(; wavenumber=k)

    # hybridlevel = 6
    # ishybrid = H2Trees.ishybridlevel(originaltree, hybridlevel)
    hybridsize = originaltree.radius / 7
    ishybrid = H2Trees.ishybridradius(originaltree, hybridsize)
    htree = HybridOcParametricBoundingBall(originaltree, λ / 1.5, ishybrid)

    A = assemble(
        operator, X, X; quadstrat=BEAST.DoubleNumWiltonSauterQStrat(7, 7, 7, 7, 1, 1, 1, 1)
    )

    Ahybrid = HybridFMM.assemble(
        operator,
        X,
        htree;
        polynomial=polynomial,
        NearInteractionsAssembler=H2Factory.DummyNearInteractionsAssembler,
    )

    disaggregationplan = Ahybrid.disaggregationplan
    tfs = Ahybrid.i2otranslator.lowertfs
    i2itranslator = Ahybrid.i2itranslator.iFMMi2itranslator

    lpolynomials = iFMM.leafpolynomials(htree, polynomial)
    momentcollections = Ahybrid.momentcollections

    leaves = H2Trees.leaves(htree)

    #TODO: make this a function in H2FactoryTestUtils
    function interpolatecollection(tree, node, leafcollections, i2itranslator)
        if H2Trees.isleaf(tree, node)
            return leafcollections[node]
        end

        vals = H2Trees.values(tree, node)
        pplusonesquared = size(first(values(leafcollections))[1])[1]
        moments = [Matrix{Float64}(undef, pplusonesquared, length(vals)) for _ in 1:1]

        for child in H2Trees.children(tree, node)
            childcollection = interpolatecollection(
                tree, child, leafcollections, i2itranslator
            )

            ii = [findfirst(isequal(point), vals) for point in H2Trees.values(tree, child)]

            for momentid in eachindex(childcollection)
                moments[momentid][:, ii] .=
                    i2itranslator.translationmatrices[H2Trees.sector(tree, child) + 1] *
                    childcollection[momentid]

                @assert !any(isnan.(childcollection[momentid],))
            end
        end

        return iFMM.TrialMomentCollection(SVector{1}(moments))
    end

    function unpackmoment(m)
        @cases m begin
            [FarField, PolynomialMoment](munpacked) => return munpacked
        end
    end

    x = randn(ComplexF64, numfunctions(X))

    Ahybrid * x

    Ahybridmoments = H2Factory.moments(Ahybrid)

    maxerror = 0
    i = 0
    for node in keys(Ahybridmoments)
        momentcollection = interpolatecollection(
            htree, node, momentcollections, i2itranslator
        )

        interpolatedmoment = momentcollection[1] * x[H2Trees.values(htree, node)]

        aggregatedmoment = (unpackmoment(Ahybridmoments[node])[1])[:]

        relerror =
            norm.(aggregatedmoment - interpolatedmoment) / maximum(norm.(aggregatedmoment))
        println(maximum(relerror))
        maxerror = max(maxerror, maximum(relerror))
        @test maxerror < 1e-14
        i += 1
    end

    println("maxerror: $maxerror")
    @test i == sum(Ahybrid.aggregationplan.storenode)
end
