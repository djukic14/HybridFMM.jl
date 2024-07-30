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

fns = [joinpath(pkgdir(iFMM), "test", "assets", "sphere.dat")]
m = readMultipatch(fns[1])

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
hybridsize = tree.radius / 7
ishybrid = H2Trees.ishybridradius(tree, hybridsize)
htree = HybridOcParametricBoundingBall(tree, λ / 1.5, ishybrid)
@assert H2Trees.testwellseparatedness(htree)

A = assemble(
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
# x[a] .= 0
# x = Array(1:numfunctions(X))
# x[a] .= 0

# x = zeros(numfunctions(X))
# leaf = 9
# x[H2Trees.values(htree, leaf)] .= randn(Float64, length(H2Trees.values(htree, leaf)))

y = Ahybrid * x;
# y = Afar * x + Anear * x
yA = Afar * x;

er = norm.(yA - y) / maximum(norm.(yA))

@show maximum(er)

# a = findall(x -> x > 1e-5, er)
# ls = Int[]
# for value in a
#     push!(ls, H2Trees.findleafnode(htree, value))
# end
# unique!(sort!(ls))

# # # level 2: 1115

# # level 3: 701

# for node in htree.lowertree.nodesatlevel[end]
#     ptree = H2Trees.parametrictree(tree)
#     paramnode = H2Trees.parametricnode(htree, node)

#     @test H2Trees.sector(ptree, paramnode) == H2Trees.sector(htree, node)
# end

leafnodelevel2 = filter(x -> H2Trees.level(htree, x) == 2, collect(H2Trees.leaves(htree)))
for leafnodelevel2 in leafnodelevel2
    leafcollection = Ahybrid.momentcollections[leafnodelevel2]
    parentnodelevel2 = filter(
        x -> !H2Trees.isleaf(htree, x),
        collect(H2Trees.WellSeparatedIterator(htree, leafnodelevel2)),
    )#[1]
    for parentnodelevel2 in parentnodelevel2
        tf = Ahybrid.i2otranslator.lowertfs.dict[(leafnodelevel2, parentnodelevel2)]

        children = collect(H2Trees.children(htree, parentnodelevel2))
        child = children[2]
        for child in children
            childcollection = Ahybrid.momentcollections[child]
            o2omatrix = Ahybrid.o2otranslator.iFMMo2otranslator.translationmatrices[H2Trees.sector(htree, child) + 1]

            blockmatrix = A[
                H2Trees.values(htree, leafnodelevel2), H2Trees.values(htree, child)
            ]
            interpolatedblockmatrix = zeros(ComplexF64, size(blockmatrix))

            interpolatedblockmatrix +=
                operator.operator.β *
                transpose(leafcollection[1]) *
                tf *
                transpose(o2omatrix) *
                childcollection[1]

            relerror =
                norm.(blockmatrix - interpolatedblockmatrix) / maximum(norm.(blockmatrix))
            println(maximum(relerror))
        end
    end
end
##

leafnodelevel2 = 8
leafcollection = Ahybrid.momentcollections[leafnodelevel2]
testleafnodelevel2 = filter(
    x -> H2Trees.isleaf(htree, x),
    collect(H2Trees.WellSeparatedIterator(htree, leafnodelevel2)),
)[1]

testleafcollection = Ahybrid.momentcollections[testleafnodelevel2]

tf = Ahybrid.i2otranslator.lowertfs.dict[(testleafnodelevel2, leafnodelevel2)]

blockmatrix = A[
    H2Trees.values(htree, leafnodelevel2), H2Trees.values(htree, testleafnodelevel2)
]

interpolatedblockmatrix = zeros(ComplexF64, size(blockmatrix))

interpolatedblockmatrix +=
    operator.operator.β * transpose(testleafcollection[1]) * tf * leafcollection[1]

relerror = norm.(blockmatrix - interpolatedblockmatrix) / maximum(norm.(blockmatrix))

for node in H2Trees.DepthFirstIterator(htree)
    nodes = Int[]
    append!(nodes, collect(H2Trees.FarNodeIterator(htree, node)))
    append!(nodes, collect(H2Trees.NearNodeIterator(htree, node)))

    sort!(nodes)

    @test nodes == collect(H2Trees.SameLevelIterator(htree, node))

    vals = Int[]
end

for leaf in H2Trees.leaves(htree)
    vals = Int[]

    append!(vals, H2Trees.farnodevalues(htree, leaf))
    append!(vals, H2Trees.nearnodevalues(htree, leaf))

    sort!(vals)

    @test vals == 1:numfunctions(X)
end

fakeAfar = zeros(ComplexF64, size(A))

@showprogress for i in 1:numfunctions(X)
    x = zeros(numfunctions(X))
    x[i] = 1
    fakeAfar[:, i] .= Ahybrid * x
end

er = norm.(fakeAfar - transpose(fakeAfar))
@show maximum(er)

fakeAnear = A - fakeAfar

for i in eachindex(fakeAnear)
    norm(fakeAnear[i]) < 1e-5 && (fakeAnear[i] = 0)
end
fakeAnear = sparse(fakeAnear)
