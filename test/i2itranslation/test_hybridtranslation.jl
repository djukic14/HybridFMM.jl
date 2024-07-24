using PlotlyJS
using NURBS
using BEAST
using Test
using H2Factory
using LinearAlgebra
using H2FactoryTestUtils
using ClusterTrees
using HybridFMM
using iFMM
using MLFMA

BEASTnurbs = Base.get_extension(BEAST, :BEASTnurbs)
iFMMNURBS = Base.get_extension(iFMM, :iFMMNURBS)
fns = [
    # joinpath(pkgdir(H2Factory), "test", "assets",  "cube.dat"),
    # joinpath(pkgdir(HybridFMM), "test", "assets", "fichera.dat"),
    joinpath(pkgdir(HybridFMM), "test", "assets", "toy_boat.dat"),
]

m = readMultipatch(fns[1])
p = 3
N = 2^5 + p
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

polyp = 10
polynomial = iFMM.BarycentricLagrangePolynomial2DChebyshev2(polyp)
op = Maxwell3D.singlelayer(; wavenumber=1.0)

ptranslator = iFMMNURBS.ParametriciFMMTranslator(
    op, polynomial, lowertree, disaggregationplan, X
)

sampling = MLFMA.SphericalSampling{Float64}(40, 80)

hi2i = HybridFMM.HybridI2ITranslator(ptranslator, htree, sampling);

farfield = MLFMA.FarField{4}(1.0 * im, sampling)
polmoment = iFMM.iFMM.PolynomialMoment{3,ComplexF64}((polyp + 1, polyp + 1))

hi2i(farfield, polmoment, htree, 1, collect(values(htree.hybridnodes))[2])

ho2o = HybridFMM.HybridO2OTranslator(hi2i);

ho2o(polmoment, farfield, htree, 1, collect(values(htree.hybridnodes))[2])
