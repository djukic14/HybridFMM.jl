
function cartesianfarfielddims(::BEAST.MWSingleLayer3D)
    return 4
end

function cartesianfarfielddims(::iFMM.MWWeaklySingular3D)
    return 3
end

function cartesianfarfielddims(::iFMM.MWHyperSingular3D)
    return 1
end

function moments(γ, tree, relevantlevels, operator, polynomial, aggregationplan, samplings)
    return H2Trees.storagemomentsplan(
        aggregationplan,
        tree,
        moment(γ, tree, relevantlevels, operator, polynomial, samplings),
    )
end

function storagemoments(
    γ, tree, relevantlevels, operator, polynomial, samplings; threading=Val{:single}()
)
    return H2Trees.storagemoments(
        relevantlevels,
        tree,
        moment(γ, tree, relevantlevels, operator, polynomial, samplings);
        threading=threading,
    )
end

function moment(γ::T, tree, relevantlevels, operator, polynomial, samplings) where {T}
    tmoment = iFMM.trialmoment(operator, polynomial)

    return moment(
        γ,
        tree,
        relevantlevels,
        polynomial,
        samplings,
        Val{MLFMA.FarField{cartesianfarfielddims(operator),Matrix{T}}}(),
        Val{iFMM.PolynomialMoment{iFMM.dims(tmoment),Array{T,1}}}(),
    )
end

function moment(
    γ::T, tree, relevantlevels, polynomial, samplings, ::Val{F}, ::Val{P}
) where {T,DF,MF,DP,MP,F<:MLFMA.FarField{DF,MF},P<:iFMM.PolynomialMoment{DP,MP}}
    minrelevantlevel = relevantlevels[begin]

    function moment(tree, level)::HybridMoment{F,P}
        if H2Trees.isuppertreelevel(tree, level)
            return HybridMoment'.FarField(
                MLFMA.FarField{DF}(γ, samplings[level - minrelevantlevel + 1])
            )
        else
            a = HybridMoment'.PolynomialMoment(
                iFMM.PolynomialMoment{DP,eltype(MP)}((
                    prod(size(LinearIndices(polynomial))),
                )),
            )

            return a
        end
    end

    return moment
end
