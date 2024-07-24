using PlotlyJS
using NURBS
using BEAST
using Test
using H2Factory
using LinearAlgebra
using H2FactoryTestUtils
using ClusterTrees
using iFMM
using MLFMA

BEASTnurbs = Base.get_extension(BEAST, :BEASTnurbs)
fns = [joinpath(pkgdir(H2Factory), "test", "assets", "step", "sphere.dat")]

λ = 1.0
k = 2π / λ
γ = k * im

polynomial = iFMM.BarycentricLagrangePolynomial2DChebyshev2(4)

op = Maxwell3D.singlelayer(; wavenumber=k)

m = readMultipatch(fns[1])
p = 1
N = 2^3 + p
X = BsplineBasisDiv(m, p, N)
X = superspace(X)
tree = ParametricBoundingBallTree(X, 1 / N)

hybridlevel = 4

ishybrid = H2Trees.ishybridlevel(tree, hybridlevel)

htree = H2Factory.HybridOcParametricBoundingBall(tree, tree.radius / 8, ishybrid)
lowertree = htree.lowertree
uppertree = htree.uppertree

disaggregationplan = H2Trees.DisaggregationPlan(htree, H2Trees.TranslatingNodesIterator)

up, low = H2Trees.DisaggregationPlan(htree, disaggregationplan)

@test sort(disaggregationplan.disaggregationnodes) ==
    sort(vcat(up.disaggregationnodes, low.disaggregationnodes))

tfsamplings = [
    MLFMA.SphericalSampling{Float64}(8, 16), MLFMA.SphericalSampling{Float64}(4, 8)
]

tfs = MLFMA.TranslationCollection(1.0 * im, htree.uppertree, up, tfsamplings)

itfs = iFMM.I2OTranslationCollection(op, htree.lowertree, polynomial, low, X);

olditfs = iFMM.I2OTranslationCollection(
    op,
    tree,
    polynomial,
    H2Trees.DisaggregationPlan(tree, H2Trees.TranslatingNodesIterator),
    X,
);
