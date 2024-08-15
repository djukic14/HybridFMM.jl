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
using SparseArrays
using ProgressMeter

BEASTnurbs = Base.get_extension(BEAST, :BEASTnurbs)
H2NURBS = Base.get_extension(H2Factory, :H2NURBS)
iFMMNURBS = Base.get_extension(iFMM, :iFMMNURBS)

fns = [joinpath(pkgdir(iFMM), "test", "assets", "platewithholes.dat")]
fnsStep = [joinpath(pkgdir(iFMM), "test", "assets", "MigClosed.stp")]

m = readMultipatch(fns[1])
# m = readStep(fnsStep[1])

p = 1
N = 2^3 + p
X = BsplineBasisDiv(m, p, N)
X = superspace(X; interfacesonly=false)

tree = ParametricBoundingBallTree(X, 1 / N)

polyp = 6
polynomial = iFMM.BarycentricLagrangePolynomial2DChebyshev2(polyp)

λ = tree.radius / 1
k = 2π / λ
operator = iFMM.MWWeaklySingular3D(; wavenumber=k)

# hybridlevel = 6
# ishybrid = H2Trees.ishybridlevel(tree, hybridlevel)
hybridsize = λ / 5
ishybrid = H2Trees.ishybridradius(tree, hybridsize)
htree = HybridOcParametricBoundingBall(tree, λ / 2, ishybrid)
@assert H2Trees.testwellseparatedness(htree)

A = BEAST.assemble(
    operator, X, X; quadstrat=BEAST.DoubleNumWiltonSauterQStrat(7, 7, 7, 7, 1, 1, 1, 1)
)

Ahybrid = HybridFMM.assemble(
    operator,
    X,
    htree;
    polynomial=polynomial,
    # NearInteractionsAssembler=H2Factory.SparseBEASTNearInteractionsAssembler,
    NearInteractionsAssembler=H2Factory.DummyNearInteractionsAssembler,
);

Afar = farinteractions(A, Ahybrid)
Anear = sparse(nearinteractions(A, Ahybrid))

x = randn(ComplexF64, numfunctions(X))

y = Ahybrid * x;
yA = Afar * x;

er = norm.(yA - y) / maximum(norm.(yA))

@show maximum(er)
