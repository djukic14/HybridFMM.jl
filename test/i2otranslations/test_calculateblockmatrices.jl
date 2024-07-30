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

@testset "Hybrid leaf block matrices" begin
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

    lpolynomials = iFMM.leafpolynomials(htree, polynomial)
    momentcollections = Ahybrid.momentcollections

    leaves = H2Trees.leaves(htree)

    maxrelerror = 0
    levels = Set{Int}()
    for leafnode in leaves
        # global maxrelerror

        !H2Trees.isleaf(htree, leafnode) && continue
        farnodes = collect(H2Trees.WellSeparatedIterator(htree, leafnode))

        isempty(farnodes) && continue

        for testleafnode in farnodes
            !H2Trees.isleaf(htree, testleafnode) && continue

            blockmatrix = A[
                H2Trees.values(htree, leafnode), H2Trees.values(htree, testleafnode)
            ]

            tf = tfs.dict[leafnode, testleafnode]

            interpolatedblockmatrix = zeros(ComplexF64, size(blockmatrix))

            interpolatedblockmatrix +=
                operator.operator.β *
                transpose(momentcollections[leafnode].moments[1]) *
                tf *
                momentcollections[testleafnode].moments[1]

            relerror =
                norm.(blockmatrix - interpolatedblockmatrix) / maximum(norm.(blockmatrix))

            push!(levels, H2Trees.level(htree, leafnode))
            push!(levels, H2Trees.level(htree, testleafnode))
            maxlocalrelerror = maximum(relerror)
            @test maxlocalrelerror < 1e-5
            maxrelerror = max(maxrelerror, maxlocalrelerror)
        end
    end

    println("maximum relative error:", maxrelerror)
end
