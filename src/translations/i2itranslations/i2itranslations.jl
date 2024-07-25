struct I2ITranslator{I,P,H,IP}
    MLFMAinterpolator::I
    MLFMAphaseshifter::P
    hybridi2itranslator::H
    iFMMi2itranslator::IP
end

function I2ITranslator(
    operator,
    tree,
    samplings,
    interpolationsamplings,
    relevantoctreelevels,
    polynomial,
    ptranslator;
    InterpolatorType=MLFMA.SphericalLagrangeInterpolator{15,ComplexF64},
    PhaseShifterType=MLFMA.PhaseShifter,
)
    intersamplings = samplings[2:end]
    interparentsamplings = interpolationsamplings[2:end]

    #TODO: make this prettier with the nothings
    interpolator = if isempty(intersamplings)
        nothing
    else
        interpolator = InterpolatorType(
            intersamplings,
            interparentsamplings;
            interpolationtrait=MLFMA.interpolationtrait(operator),
        )
    end

    phaseshifter = if isempty(interparentsamplings)
        nothing
    else
        PhaseShifterType(
            BEAST.gamma(operator),
            tree.uppertree,
            relevantoctreelevels,
            interparentsamplings,
        )
    end

    hybridtranslator = if isempty(samplings)
        nothing
    else
        HybridI2ITranslator(BEAST.gamma(operator), ptranslator, tree, samplings[end])
    end

    return I2ITranslator(
        interpolator, phaseshifter, hybridtranslator, iFMM.I2ITranslator(polynomial)
    )
end

function (i2itranslatorfunctor::I2ITranslatorFunctor{<:GalerkinHybridFMM})(
    parentmoment, childmoment, tree, parent, child
)
    @cases parentmoment begin
        FarField(parentmoment) => @cases childmoment begin
            FarField(childmoment) => farfieldi2itranslation!(
                parentmoment, childmoment, tree, parent, child, i2itranslatorfunctor
            )
            PolynomialMoment(childmoment) => hybridi2itranslation!(
                parentmoment, childmoment, tree, parent, child, i2itranslatorfunctor
            )
        end
        PolynomialMoment(parentmoment) => @cases childmoment begin
            [FarField, PolynomialMoment](childmoment) => polynomiali2itranslation!(
                parentmoment, childmoment, tree, parent, child, i2itranslatorfunctor
            )
        end
    end
    return parentmoment
end

function farfieldi2itranslation!(
    parentmoment::MLFMA.FarField,
    childmoment::MLFMA.FarField,
    tree,
    parent,
    child,
    i2itranslatorfunctor,
)
    i2itranslator = i2itranslatorfunctor.H2map.i2itranslator

    level = H2Trees.level(tree, child)
    H2map = i2itranslatorfunctor.H2map

    i2itranslator.MLFMAinterpolator(
        parentmoment,
        childmoment,
        sampling(H2map, level),
        sampling(H2map, level - 1);
        interpolationtrait=HybridFMM.defaultinterpolationtrait(parentmoment), #TODO: make this configurable
    )

    MLFMA.shifttoparentphase!(parentmoment, i2itranslator.MLFMAphaseshifter, tree, child)

    return nothing
end

function hybridi2itranslation!(
    parentmoment::MLFMA.FarField,
    childmoment::iFMM.PolynomialMoment,
    tree,
    parent,
    child,
    i2itranslatorfunctor,
)
    i2itranslator = i2itranslatorfunctor.H2map.i2itranslator

    i2itranslator.hybridi2itranslator(parentmoment, childmoment, tree, parent, child)

    return nothing
end

function polynomiali2itranslation!(
    parentmoment::iFMM.PolynomialMoment,
    childmoment::iFMM.PolynomialMoment,
    tree,
    parent,
    child,
    i2itranslatorfunctor,
)
    i2itranslator = i2itranslatorfunctor.H2map.i2itranslator

    i2itranslator.iFMMi2itranslator(parentmoment, childmoment, tree, parent, child)

    return nothing
end
