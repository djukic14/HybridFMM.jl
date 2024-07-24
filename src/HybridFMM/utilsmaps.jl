function moments(γ, tree, relevantlevels, operator, polynomial, aggregationplan, samplings)
    return H2Trees.storagemomentsplan(
        aggregationplan,
        tree,
        moment(γ, tree, relevantlevels, operator, polynomial, samplings),
    )
end

function storagemoments(γ, tree, relevantlevels, operator, polynomial, samplings)
    return H2Trees.storagemoments(
        relevantlevels,
        tree,
        moment(γ, tree, relevantlevels, operator, polynomial, samplings),
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
        Val{MLFMA.FarField{MLFMA.farfielddims(operator) + 1,Matrix{T}}}(),
        Val{iFMM.PolynomialMoment{iFMM.dims(tmoment),Array{T,2}}}(),
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
                iFMM.PolynomialMoment{DP,eltype(MP)}(size(LinearIndices(polynomial)))
            )

            return a
        end
    end

    return moment
end
