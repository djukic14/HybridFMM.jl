using NURBS
using BEAST
using Test
using H2Factory
using LinearAlgebra
using H2FactoryTestUtils
using iFMM
using MLFMA

BEASTnurbs = Base.get_extension(BEAST, :BEASTnurbs)
H2NURBS = Base.get_extension(H2Factory, :H2NURBS)
iFMMNURBS = Base.get_extension(iFMM, :iFMMNURBS)

λ = 1.0
k = 2π / λ

operator = Maxwell3D.singlelayer(; wavenumber=k, alpha=0.0, beta=1.0)

fnsstep = [joinpath(pkgdir(iFMM), "test", "assets", "Pyramid.stp")]
m = readStep(fnsstep[1])

# fns = [joinpath(pkgdir(iFMM), "test", "assets", "cube.dat")]
# m = readMultipatch(fns[1])

p = 2
N = 2^2 + p
X = BsplineBasisDiv(m, p, N)
X = superspace(X; interfacesonly=false)

A = assemble(
    operator, X, X; quadstrat=BEAST.DoubleNumWiltonSauterQStrat(7, 7, 7, 7, 1, 1, 1, 1)
)

tree = ParametricBoundingBallTree(X, 1 / N)
polyp = 8
polynomial = iFMM.BarycentricLagrangePolynomial2DChebyshev2(polyp)

disaggregationplan = H2Trees.DisaggregationPlan(tree, H2Trees.TranslatingNodesIterator)

momentcollections = Dict(
    zip(
        H2Trees.leaves(tree),
        iFMM.TrialMomentCollections(
            operator,
            tree,
            iFMM.leafpolynomials(tree, polynomial),
            X;
            quadstrat=BEAST.SingleNumQStrat(p + polyp),
        ),
    ),
)

leafindex = 1
leaf = H2Trees.leaves(tree)[leafindex]

testnodeindex = 11
testnode = H2Trees.translatingnodes(disaggregationplan, leaf)[testnodeindex]

leafmomentcollection = momentcollections[leaf]
testnodemomentcollection = momentcollections[testnode]

translationdirection = H2Trees.center(tree, testnode) - H2Trees.center(tree, leaf)

sampling, L = MLFMA.sampling(
    im * k,
    1e-5,
    max(H2Trees.radius(tree, leaf), H2Trees.radius(tree, testnode)) * 2,
    MLFMA.SphericalSampling{Float64},
)

cartsamplingpoints, weights = MLFMA.cartesiansamplepoints(sampling)

translationoperator = zeros(ComplexF64, size(sampling))
MLFMA.translationoperator!(
    translationoperator,
    im * k,
    MLFMA.cartesiansamplepoints(sampling)[1],
    H2Trees.center(tree, leaf),
    H2Trees.center(tree, testnode),
    L,
)
translationoperator = translationoperator[:]
translationoperator .*= weights[:]

pTranslator = iFMMNURBS.ParametriciFMMTranslator(
    operator, polynomial, tree, disaggregationplan, X
)

idsleafu, idsleafv = iFMMNURBS._indicesranges(
    tree.parametrictree, polynomial, H2Trees.parametricnode(tree, leaf)
)
idstestnodeu, idstestnodev = iFMMNURBS._indicesranges(
    tree.parametrictree, polynomial, H2Trees.parametricnode(tree, testnode)
)

leaffieldcollection = MLFMA.RadiationFieldCollection{4,ComplexF64}((
    length(sampling), (polyp + 1)^2
))

for i in eachindex(leaffieldcollection)
    leaffieldcollection[i] .= 0.0
end

testfieldcollection = MLFMA.RadiationFieldCollection{4,ComplexF64}((
    length(sampling), (polyp + 1)^2
))

for i in eachindex(testfieldcollection)
    testfieldcollection[i] .= 0.0
end

leafinterpolationpoints = pTranslator.evaluatedgeometry[H2Trees.patchID(tree, leaf)][H2Trees.levelindex(tree, leaf) - 1][
    1, 1
][
    idsleafu, idsleafv
]

testinterpolationpoints = pTranslator.evaluatedgeometry[H2Trees.patchID(tree, testnode)][H2Trees.levelindex(tree, testnode) - 1][
    1, 1
][
    idstestnodeu, idstestnodev
]

leafconvertmatrix = [
    exp(
        -im *
        k *
        dot(leafinterpolationpoints[i] - H2Trees.center(tree, leaf), cartsamplingpoints[j]),
    ) for j in eachindex(cartsamplingpoints), i in eachindex(leafinterpolationpoints)
]

testnodeconvertmatrix = [
    exp(
        im *
        k *
        dot(
            testinterpolationpoints[i] - H2Trees.center(tree, testnode),
            cartsamplingpoints[j],
        ),
    ) for j in eachindex(cartsamplingpoints), i in eachindex(testinterpolationpoints)
]

b =
    operator.β *
    transpose(leafconvertmatrix * leafmomentcollection[3]) *
    (translationoperator .* testnodeconvertmatrix * testnodemomentcollection[3])

Ablock = A[H2Trees.values(tree, leaf), H2Trees.values(tree, testnode)]

er =
    norm.(A[H2Trees.values(tree, leaf), H2Trees.values(tree, testnode)] - b) ./
    maximum(norm.(A[H2Trees.values(tree, leaf), H2Trees.values(tree, testnode)]))
