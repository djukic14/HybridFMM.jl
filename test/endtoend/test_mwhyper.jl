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

BEASTnurbs = Base.get_extension(BEAST, :BEASTnurbs)
H2NURBS = Base.get_extension(H2Factory, :H2NURBS)
iFMMNURBS = Base.get_extension(iFMM, :iFMMNURBS)

fns = [joinpath(pkgdir(iFMM), "test", "assets", "sphere.dat")]
m = readMultipatch(fns[1])

p = 1
N = 2^4 + p
X = BsplineBasisDiv(m, p, N)
X = superspace(X; interfacesonly=false)

tree = ParametricBoundingBallTree(X, 1 / N)

polyp = 4
polynomial = iFMM.BarycentricLagrangePolynomial2DChebyshev2(polyp)

λ = tree.radius / 16
k = 2π / λ
# operator = iFMM.MWHyperSingular3D(; wavenumber=k)
operator = Maxwell3D.singlelayer(; wavenumber=k)

# hybridlevel = 6
# ishybrid = H2Trees.ishybridlevel(tree, hybridlevel)
hybridsize = λ / 4
ishybrid = H2Trees.ishybridradius(tree, hybridsize)
htree = HybridOcParametricBoundingBall(tree, 2 * λ, ishybrid)
@assert H2Trees.testwellseparatedness(htree)

A = assemble(
    operator, X, X; quadstrat=BEAST.DoubleNumWiltonSauterQStrat(7, 7, 7, 7, 1, 1, 1, 1)
)

# Ahybrid = HybridFMM.assemble(
#     operator,
#     X,
#     htree;
#     polynomial=polynomial,
#     NearInteractionsAssembler=H2Factory.DummyNearInteractionsAssembler,
# );

Ahybrid = MLFMA.assemble(
    operator, X; NearInteractionsAssembler=H2Factory.DummyNearInteractionsAssembler
)

Afar = farinteractions(A, Ahybrid)

x = randn(ComplexF64, numfunctions(X))
# x = zeros(ComplexF64, numfunctions(X))
# x[1] = 1

y = Ahybrid * x;
yA = Afar * x;

er = norm.(yA - y) / maximum(norm.(yA))

@show maximum(er)
