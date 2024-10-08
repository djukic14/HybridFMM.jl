using NURBS
using BEAST
using Test
using H2Factory
using LinearAlgebra
using H2FactoryTestUtils
using iFMM
using MLFMA
using SphericalScattering

BEASTnurbs = Base.get_extension(BEAST, :BEASTnurbs)
H2NURBS = Base.get_extension(H2Factory, :H2NURBS)
iFMMNURBS = Base.get_extension(iFMM, :iFMMNURBS)

λ = 1.0
k = 2π / λ

operator = Maxwell3D.singlelayer(; wavenumber=k, alpha=1.0, beta=0.0)

fns = [joinpath(pkgdir(iFMM), "test", "assets", "torus.dat")]
m = readMultipatch(fns[1])

p = 1
N = 2^3 + p
X = BsplineBasisDiv(m, p, N)
X = superspace(X; interfacesonly=false)

A = assemble(
    operator, X, X; quadstrat=BEAST.DoubleNumWiltonSauterQStrat(7, 7, 7, 7, 1, 1, 1, 1)
)

tree = ParametricBoundingBallTree(X, 1 / N)
polyp = 7
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

leafinterpolationpoints = pTranslator.evaluatedgeometry[H2Trees.patchID(tree, leaf)][H2Trees.levelindex(tree, leaf) - 1][
    1, 1
][
    idsleafu, idsleafv
]

leafJu = pTranslator.evaluatedgeometry[H2Trees.patchID(tree, leaf)][H2Trees.levelindex(tree, leaf) - 1][
    2, 1
][
    idsleafu, idsleafv
]

leafJv = pTranslator.evaluatedgeometry[H2Trees.patchID(tree, leaf)][H2Trees.levelindex(tree, leaf) - 1][
    1, 2
][
    idsleafu, idsleafv
]

testinterpolationpoints = pTranslator.evaluatedgeometry[H2Trees.patchID(tree, testnode)][H2Trees.levelindex(tree, testnode) - 1][
    1, 1
][
    idstestnodeu, idstestnodev
]

testJu = pTranslator.evaluatedgeometry[H2Trees.patchID(tree, testnode)][H2Trees.levelindex(tree, testnode) - 1][
    1, 2
][
    idstestnodeu, idstestnodev
]

testJv = pTranslator.evaluatedgeometry[H2Trees.patchID(tree, testnode)][H2Trees.levelindex(tree, testnode) - 1][
    2, 1
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

leafconvertmatrix2 = [
    exp(
        im *
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

leaffieldcollection = MLFMA.RadiationFieldCollection{4,ComplexF64}((
    length(sampling), length(H2Trees.values(tree, leaf))
))
testfieldcollection = MLFMA.ReceiveFieldCollection{4,ComplexF64}((
    length(sampling), length(H2Trees.values(tree, testnode))
))

for i in eachindex(leaffieldcollection)
    leaffieldcollection[i] .= 0.0
    testfieldcollection[i] .= 0.0
end

leaffieldcollection[4] .= leafconvertmatrix * leafmomentcollection[3]
testfieldcollection[4] .= testnodeconvertmatrix * testnodemomentcollection[3]

for i in eachindex(leafinterpolationpoints)
    Ju = leafJu[i]
    Jv = leafJv[i]
    # Ju = SphericalScattering.convertCartesian2Spherical(
    #     leafJu[i], SphericalScattering.cart2sph(leafinterpolationpoints[i])
    # )

    # Jv = SphericalScattering.convertCartesian2Spherical(
    #     leafJv[i], SphericalScattering.cart2sph(leafinterpolationpoints[i])
    # )

    for j in eachindex(H2Trees.values(tree, leaf))
        su = leafmomentcollection[1][i, j] * Ju
        sv = leafmomentcollection[2][i, j] * Jv

        for sindex in eachindex(su)
            leaffieldcollection[sindex][:, j] += su[sindex] .* leafconvertmatrix[:, i]
            leaffieldcollection[sindex][:, j] += sv[sindex] .* leafconvertmatrix[:, i]
        end
    end
end

for i in eachindex(testinterpolationpoints)
    Ju = testJu[i]
    Jv = testJv[i]
    # Ju = SphericalScattering.convertCartesian2Spherical(
    #     testJu[i], SphericalScattering.cart2sph(testinterpolationpoints[i])
    # )

    # Jv = SphericalScattering.convertCartesian2Spherical(
    #     testJv[i], SphericalScattering.cart2sph(testinterpolationpoints[i])
    # )

    for j in eachindex(H2Trees.values(tree, testnode))
        # Ju = SphericalScattering.convertCartesian2Spherical(
        #     testJu[i], SphericalScattering.cart2sph(cartsamplingpoints[i])
        # )

        # Jv = SphericalScattering.convertCartesian2Spherical(
        #     testJv[i], SphericalScattering.cart2sph(cartsamplingpoints[i])
        # )

        su = testnodemomentcollection[1][i, j] * Jv
        sv = testnodemomentcollection[2][i, j] * Ju

        for sindex in eachindex(su)
            testfieldcollection[sindex][:, j] += su[sindex] .* testnodeconvertmatrix[:, i]
            testfieldcollection[sindex][:, j] += sv[sindex] .* testnodeconvertmatrix[:, i]
        end
    end
end

blockmatrix =
    operator.β *
    transpose(leaffieldcollection[4]) *
    (translationoperator .* testfieldcollection[4])

blockmatrix +=
    operator.α *
    transpose(leaffieldcollection[1]) *
    (translationoperator .* testfieldcollection[1])

blockmatrix +=
    operator.α *
    transpose(leaffieldcollection[2]) *
    (translationoperator .* testfieldcollection[2])

blockmatrix +=
    operator.α *
    transpose(leaffieldcollection[3]) *
    (translationoperator .* testfieldcollection[3])

Ablock = A[H2Trees.values(tree, leaf), H2Trees.values(tree, testnode)]

er = norm.(Ablock - blockmatrix) ./ maximum(norm.(Ablock))
