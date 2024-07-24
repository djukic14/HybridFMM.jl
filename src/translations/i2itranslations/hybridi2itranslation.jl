struct HybridI2ITranslator{C,U,V}
    convertmatrices::C
    Jus::U
    Jvs::V
end

function HybridI2ITranslator(γ, ptranslator, tree, sampling::MLFMA.AbstractSampling)
    convertmatrices = Vector{Matrix{complex(eltype(sampling))}}(
        undef, length(H2Trees.hybridnodes(tree))
    )
    Jus = Vector{Matrix{SVector{3,eltype(sampling)}}}(
        undef, length(H2Trees.hybridnodes(tree))
    )
    Jvs = Vector{Matrix{SVector{3,eltype(sampling)}}}(
        undef, length(H2Trees.hybridnodes(tree))
    )

    cartesiansamplingpoints, _ = MLFMA.cartesiansamplepoints(sampling)

    hnodes = collect(values(H2Trees.hybridnodes(tree)))

    for (i, node) in enumerate(hnodes)
        parent = H2Trees.parent(tree, node)
        evaluatedgeometry = ptranslator.evaluatedgeometry[H2Trees.patchID(tree, node)][H2Trees.levelindex(tree.lowertree, node) - 1]

        idsu, idsv = iFMMNURBS._indicesranges(
            H2Trees.parametrictree(tree.lowertree),
            ptranslator.originalpolynomial,
            H2Trees.parametricnode(tree, node),
        )

        interpolationpoints = evaluatedgeometry[1, 1][idsu, idsv]

        Jus[i] = evaluatedgeometry[2, 1][idsu, idsv]
        Jvs[i] = evaluatedgeometry[1, 2][idsu, idsv]

        convertmatrix = Matrix{complex(eltype(sampling))}(
            undef, length(cartesiansamplingpoints), length(interpolationpoints)
        )

        for h in eachindex(interpolationpoints), j in eachindex(cartesiansamplingpoints)
            convertmatrix[j, h] = exp(
                γ * dot(
                    interpolationpoints[h] - H2Trees.center(tree, parent),
                    cartesiansamplingpoints[j],
                ),
            )
        end

        convertmatrices[i] = convertmatrix
    end

    return HybridI2ITranslator(
        Dict(zip(hnodes, convertmatrices)), Dict(zip(hnodes, Jus)), Dict(zip(hnodes, Jvs))
    )
end

# function (hybridtranslator::HybridI2ITranslator)(
#     parentmoment, childmoment, tree, parent::Int, child::Int
# )
#     LinearAlgebra.mul!(
#         reshape(parentmoment[4], :),
#         hybridtranslator.convertmatrices[child],
#         reshape(childmoment[3], :),
#     )

#     Jus = hybridtranslator.Jus[child]
#     Jvs = hybridtranslator.Jvs[child]

#     for i in 1:3
#         parentmoment[i] .= zero(eltype(parentmoment[i]))
#     end

#     for momentindex in 1:2
#         for i in eachindex(Jus)
#             su = Jus[i] * childmoment[momentindex][i]
#             sv = Jvs[i] * childmoment[momentindex][i]

#             for sindex in eachindex(su)
#                 LinearAlgebra.mul!(
#                     reshape(parentmoment[sindex], :),
#                     su[sindex],
#                     view(hybridtranslator.convertmatrices[child], :, i),
#                     true,
#                     true,
#                 )

#                 LinearAlgebra.mul!(
#                     reshape(parentmoment[sindex], :),
#                     sv[sindex],
#                     view(hybridtranslator.convertmatrices[child], :, i),
#                     true,
#                     true,
#                 )

#                 # view(parentmoment[sindex], :) .+=
#                 #     su[sindex] *view(hybridtranslator.convertmatrices[child], :, i)
#                 # view(parentmoment[sindex], :) .+=
#                 #     sv[sindex] * view(hybridtranslator.convertmatrices[child], :, i)
#             end
#         end
#     end

#     return parentmoment
# end

function (hybridtranslator::HybridI2ITranslator)(
    parentmoment, childmoment, tree, parent::Int, child::Int
)
    LinearAlgebra.mul!(
        reshape(parentmoment[4], :),
        hybridtranslator.convertmatrices[child],
        reshape(childmoment[3], :),
    )

    Jus = hybridtranslator.Jus[child]
    Jvs = hybridtranslator.Jvs[child]

    for i in 1:3
        parentmoment[i] .= zero(eltype(parentmoment[i]))
    end

    parentmoments = [reshape(parentmoment[i], :) for i in eachindex(parentmoment)]
    convertmatrices = [
        view(hybridtranslator.convertmatrices[child], :, i) for i in eachindex(Jus)
    ]

    for i in eachindex(Jus)
        convertmatrix = convertmatrices[i]
        childmomentu = childmoment[1][i]
        childmomentv = childmoment[2][i]
        for sindex in eachindex(Jus[i])
            su = Jus[i][sindex] * childmomentu
            sv = Jvs[i][sindex] * childmomentv

            LinearAlgebra.mul!(parentmoments[sindex], su, convertmatrix, true, true)

            LinearAlgebra.mul!(parentmoments[sindex], sv, convertmatrix, true, true)
        end
    end

    return parentmoment
end
