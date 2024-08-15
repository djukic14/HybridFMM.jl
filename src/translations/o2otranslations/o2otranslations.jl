
struct O2OTranslator{I,P,H,IP}
    MLFMAanterpolator::I
    MLFMAphaseshifter::P
    hybrido2otranslator::H
    iFMMo2otranslator::IP
end

function O2OTranslator(i2itranslator)
    anterpolator = MLFMA.anterpolator(i2itranslator.MLFMAinterpolator)
    phaseshifter = i2itranslator.MLFMAphaseshifter
    hybrido2otranslator = HybridO2OTranslator(i2itranslator.hybridi2itranslator)
    iFMMo2otranslator = iFMM.O2OTranslator(i2itranslator.iFMMi2itranslator)

    return O2OTranslator(anterpolator, phaseshifter, hybrido2otranslator, iFMMo2otranslator)
end

function (o2otranslatorfunctor::O2OTranslatorFunctor{<:GalerkinHybridFMM})(
    parentmoment, childmoment, tree, parent, child
)
    @cases parentmoment begin
        FarField(parentmoment) => @cases childmoment begin
            [FarField, PolynomialMoment](childmoment) => farfieldo2otranslation!(
                parentmoment, childmoment, tree, parent, child, o2otranslatorfunctor
            )
        end
        PolynomialMoment(parentmoment) => @cases childmoment begin
            FarField(childmoment) => hybrido2otranslation!(
                parentmoment, childmoment, tree, parent, child, o2otranslatorfunctor
            )
            PolynomialMoment(childmoment) => polynomialo2otranslation!(
                parentmoment, childmoment, tree, parent, child, o2otranslatorfunctor
            )
        end
    end

    return parentmoment
end

function o2ostoragemoment(H2map, level)
    return o2ostoragemoment(H2map, level, H2Factory.threading(H2map))
end

function o2ostoragemoment(H2map, level, ::Val{:single})
    return H2Factory.storagemoment(H2map, level - 1)
end

function o2ostoragemoment(H2map, level, ::Val{:multi})
    return H2Factory.storagemoment(H2map, level - 1)[Threads.threadid()]
end

function farfieldo2otranslation!(
    parentmoment::MLFMA.FarField,
    childmoment::MLFMA.FarField,
    tree,
    parent,
    child,
    o2otranslatorfunctor,
)
    o2otranslator = o2otranslatorfunctor.H2map.o2otranslator
    H2map = o2otranslatorfunctor.H2map
    level = H2Trees.level(tree, child)

    storagemoment = o2ostoragemoment(H2map, level)

    @cases storagemoment begin
        FarField(storagemoment) => begin
            MLFMA.shifttochildphase!(
                storagemoment,
                childmoment,
                o2otranslator.MLFMAphaseshifter,
                tree.uppertree,
                child,
            )

            o2otranslator.MLFMAanterpolator(
                parentmoment,
                storagemoment,
                sampling(H2map, level - 1),
                sampling(H2map, level);
                interpolationtrait=HybridFMM.defaultinterpolationtrait(parentmoment), #TODO: make this configurable
            )
        end
        PolynomialMoment(storagemoment) => error("Storagemoment is not a FarField")
    end

    return nothing
end

function hybrido2otranslation!(
    parentmoment::iFMM.PolynomialMoment,
    childmoment::MLFMA.FarField,
    tree,
    parent,
    child,
    o2otranslatorfunctor,
)
    o2otranslator = o2otranslatorfunctor.H2map.o2otranslator

    o2otranslator.hybrido2otranslator(parentmoment, childmoment, tree, parent, child)

    return nothing
end

function polynomialo2otranslation!(
    parentmoment::iFMM.PolynomialMoment,
    childmoment::iFMM.PolynomialMoment,
    tree,
    parent,
    child,
    o2otranslatorfunctor,
)
    o2otranslator = o2otranslatorfunctor.H2map.o2otranslator.iFMMo2otranslator

    o2otranslator(parentmoment, childmoment, tree, parent, child)

    return nothing
end
