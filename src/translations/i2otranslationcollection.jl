struct I2OTranslationCollection{U,L}
    uppertfs::U
    lowertfs::L
end

function I2OTranslationCollection(
    operator,
    tree,
    translationsamplings,
    polynomial,
    X;
    disaggregationplan=nothing,
    upperdisaggregationplan=nothing,
    lowerdisaggregationplan=nothing,
    translator=nothing,
)
    up, low = if isnothing(upperdisaggregationplan) || isnothing(lowerdisaggregationplan)
        H2Trees.DisaggregationPlan(tree, disaggregationplan)
    else
        upperdisaggregationplan, lowerdisaggregationplan
    end

    γ = BEAST.gamma(operator)

    uppertfs = MLFMA.TranslationCollection(
        γ,
        tree.uppertree,
        up,
        translationsamplings;
        levelscales=MLFMA.getweights(translationsamplings),
    )
    lowertfs = iFMM.I2OTranslationCollection(
        operator, tree.lowertree, polynomial, low, X; translator=translator
    )

    return I2OTranslationCollection(uppertfs, lowertfs)
end

function H2Factory.i2otranslation!(
    out::MO, in::MI, receivingnode::Int, translatingnode::Int, tfs::I2OTranslationCollection
) where {MO<:HybridMoment,MI<:HybridMoment}
    @cases out begin
        FarField(out) => @cases in begin
            FarField(in) => begin
                # maxout = maximum(abs.(out[4]))
                # maxin = maximum(abs.(in[4]))

                # if maxout > 1e2
                #     println("maxout farfield = $maxout")
                # end

                # if maxin > 1e2
                #     println("maxin farfield = $maxin")
                # end
                H2Factory.i2otranslation!(out, in, receivingnode, translatingnode, tfs.uppertfs)
            end

            PolynomialMoment(in) => error(
                "i2otranslation with $(typeof(out)) and $(typeof(in)) not allowed"
            )
        end
        PolynomialMoment(out) => @cases in begin
            FarField(in) => error(
                "i2otranslation with $(typeof(out)) and $(typeof(in)) not allowed"
            )
            PolynomialMoment(in) => begin
                # maxout = maximum(abs.(out[3]))
                # maxin = maximum(abs.(in[3]))

                # if maxout > 1e2
                #     println("maxout poly= $maxout")
                # end

                # if maxin > 1e2
                #     println("maxin poly= $maxin")
                # end

                H2Factory.i2otranslation!(out, in, receivingnode, translatingnode, tfs.lowertfs)
            end
        end
    end

    return out
end
