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

@testset "Hybrid parametric i2otranslations" begin
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

    buffertest = Vector{eltype(polynomial)}(undef, 2)
    buffertrial = Vector{eltype(polynomial)}(undef, 2)

    disaggregationplan = H2Trees.DisaggregationPlan(htree, H2Trees.TranslatingNodesIterator)
    tfs = iFMM.I2OTranslationCollection(
        operator, htree.lowertree, polynomial, disaggregationplan, X
    )

    for receivingnode in H2Trees.disaggregationnodes(disaggregationplan)
        # !H2Trees.isleaf(htree, receivingnode) && continue

        preceivingnode = H2Trees.parametricnode(htree, receivingnode)

        receivingpolynomial = iFMM.shiftedpolynomial(
            htree.lowertree.parametrictree, polynomial, preceivingnode
        )

        TestLinearIndices = collect(LinearIndices(receivingpolynomial))

        patch = X.geo[H2Trees.patchID(htree, receivingnode)]
        testpoints = SVector{3,Float64}[]

        for j in TestLinearIndices
            iFMM.getsamplingpoint!(buffertest, receivingpolynomial, j)
            push!(testpoints, patch([buffertest[1]], [buffertest[2]])[1])
        end

        for translatingnode in H2Trees.translatingnodes(disaggregationplan, receivingnode)
            ptranslatingnode = H2Trees.parametricnode(htree, translatingnode)
            translatingpolynomial = iFMM.shiftedpolynomial(
                htree.lowertree.parametrictree, polynomial, ptranslatingnode
            )
            patch = X.geo[H2Trees.patchID(htree, translatingnode)]

            TrialLinearIndices = collect(LinearIndices(translatingpolynomial))

            trialpoints = SVector{3,Float64}[]
            for j in TrialLinearIndices
                iFMM.getsamplingpoint!(buffertrial, translatingpolynomial, j)
                push!(trialpoints, patch([buffertrial[1]], [buffertrial[2]])[1])
            end

            tfgreen = tfs[(receivingnode, translatingnode)]

            tfmatrix = zeros(ComplexF64, size(tfgreen))
            for j in TrialLinearIndices
                for i in TestLinearIndices
                    tfmatrix[i, j] = iFMM.scalargreen3Dkernel(
                        operator, testpoints[i], trialpoints[j]
                    )
                end
            end

            @test tfmatrix ≈ tfgreen
            # println(receivingnode, " ", translatingnode)
        end
    end
end
