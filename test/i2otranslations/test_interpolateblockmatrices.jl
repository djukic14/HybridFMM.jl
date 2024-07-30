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

BEASTnurbs = Base.get_extension(BEAST, :BEASTnurbs)
H2NURBS = Base.get_extension(H2Factory, :H2NURBS)
iFMMNURBS = Base.get_extension(iFMM, :iFMMNURBS)

@testset "Hybrid interpolating block matrices" begin
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

    a = interpolatecollection(
        htree, htree.lowertree.nodesatlevel[end - 1][2], momentcollections, i2itranslator
    )

    for level in H2Trees.levels(htree)
        println(level)
        maxrelerror = 0
        for node in H2Trees.LevelIterator(htree, level)
            wellseparatednodes = collect(H2Trees.WellSeparatedIterator(htree, node))

            isempty(wellseparatednodes) && continue

            leafmomentcollection = interpolatecollection(
                htree, node, momentcollections, i2itranslator
            )
            leafvalues = H2Trees.values(htree, node)

            for testleafnode in wellseparatednodes
                testmomentcollection = interpolatecollection(
                    htree, testleafnode, momentcollections, i2itranslator
                )
                testleafvalues = H2Trees.values(htree, testleafnode)

                blockmatrix = A[leafvalues, testleafvalues]

                tf = tfs.dict[node, testleafnode]

                interpolatedblockmatrix = zeros(ComplexF64, size(blockmatrix))

                interpolatedblockmatrix +=
                    operator.operator.β *
                    transpose(leafmomentcollection[1]) *
                    tf *
                    testmomentcollection[1]

                relerror =
                    norm.(blockmatrix - interpolatedblockmatrix) /
                    maximum(norm.(blockmatrix))
                tempmaxrelerror = maximum(relerror)

                @test tempmaxrelerror < 1e-4
                maxrelerror = max(tempmaxrelerror, maxrelerror)
            end
        end
        println("maximum relative error on level $level :", maxrelerror)
    end
end
