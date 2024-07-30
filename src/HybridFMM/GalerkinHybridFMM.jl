struct GalerkinHybridFMM{
    T,
    OperatorType,
    TreeType,
    NearInteractionsType,
    SpaceType,
    MomentCollectionDictType,
    MomentsVectorType,
    StorageMomentsVectorType,
    I2ITranslatorType,
    I2OTranslatorType,
    O2OTranslatorType,
    SamplingType,
    AggregationPlanType,
    DisaggregationPlanType,
    ISNEAR,
    αType,
} <: GalerkinH2Map{T}
    γ::T
    operator::OperatorType
    tree::TreeType
    nearinteractions::NearInteractionsType
    space::SpaceType
    momentcollections::MomentCollectionDictType
    moments::MomentsVectorType
    incomingmoments::StorageMomentsVectorType
    storagemoments::StorageMomentsVectorType
    collectedmoments::StorageMomentsVectorType
    i2itranslator::I2ITranslatorType
    i2otranslator::I2OTranslatorType
    o2otranslator::O2OTranslatorType
    samplings::Vector{SamplingType}
    aggregationplan::AggregationPlanType
    disaggregationplan::DisaggregationPlanType
    minrelevantlevel::Int
    verbose::Bool
    isnear::ISNEAR
    α::αType
end

function assemble(operator, space, tree; kwargs...)
    return GalerkinHybridFMM(operator, space, tree; kwargs...)
end

function GalerkinHybridFMM(
    operator,
    space,
    tree;
    ϵ=1e-4,
    polynomial=nothing,
    nearinteractionsquadstrategy=quadstrat = BEAST.DoubleNumWiltonSauterQStrat(
        7, 7, 7, 7, 1, 1, 1, 1
    ),
    momentquadstrategy=iFMM.MOMENTSQUADSTRATEGYRECOM,
    verbose=false,
    NearInteractionsAssembler=iFMM.NEARINTERACTIONSASSEMBLERRECOM,
    MomentAssembler=iFMM.MOMENTASSEMBLERRECOM,
    I2ITranslator=I2ITranslator,
    I2OTranslator=I2OTranslationCollection,
    O2OTranslator=O2OTranslator,
    Interpolator=MLFMA.INTERPOLATORECOM{15,typeof(MLFMA.γ(operator))},
    PhaseShifter=MLFMA.PHASESHIFTERRECOM,
    samplingfunction=MLFMA.SAMPLINGFUNCTIONRECOM,
    isnear=H2Trees.isnear(),
    translatingnodesiterator=iFMM.TRANSLATINGNODESRECOM(; isnear=isnear),
    aggregatenode=iFMM.AGGREGATENODES(; TranslatingNodesIterator=translatingnodesiterator),
)
    @assert H2Trees.numberoflevels(tree) > 1
    @assert iFMM.trialmoment(operator, polynomial) == iFMM.testmoment(operator, polynomial)
    γ = BEAST.gamma(operator)
    verbose && @info "Assembling near interactions"

    nearassembler = NearInteractionsAssembler{typeof(γ)}(
        operator, space, space, nearinteractionsquadstrategy
    )
    nearinteractions = nearassembler(tree, isnear(tree); verbose=verbose)

    verbose && @info "Assembling moments"

    galerkinplans = H2Trees.galerkinplans(tree, aggregatenode, translatingnodesiterator)
    aggregationplan = galerkinplans.aggregationplan
    disaggregationplan = galerkinplans.disaggregationplan
    upperdisaggregationplan, lowerdisaggregationplan = H2Trees.DisaggregationPlan(
        tree, disaggregationplan
    )

    relevantlevels = galerkinplans.relevantlevels
    minrelevantlevel = relevantlevels[begin]

    leafpolynomials = iFMM.leafpolynomials(tree.lowertree, polynomial)
    momentcollections = Dictionary(
        H2Trees.leaves(tree),
        iFMM.TrialMomentCollections(
            operator,
            tree.lowertree,
            leafpolynomials,
            space;
            quadstrat=momentquadstrategy,
            MomentAssembler=MomentAssembler,
            verbose=verbose,
            leaves=H2Trees.leaves(tree),
        ),
    )

    relevantoctreelevels = relevantlevels[begin]:H2Trees.levels(tree.uppertree)[end]

    samplings, parentsamplings = MLFMA.getsamplingsparentsamplings(
        γ, ϵ, tree.uppertree, relevantoctreelevels, samplingfunction
    )
    translationsamplings = samplings[(H2Trees.mintranslationlevel(disaggregationplan) - relevantlevels[begin] + 1):end]

    ptranslator = iFMMNURBS.ParametriciFMMTranslator(
        operator, polynomial, tree.lowertree, lowerdisaggregationplan, space
    )

    i2otranslator = I2OTranslator(
        operator,
        tree,
        translationsamplings,
        polynomial,
        space;
        upperdisaggregationplan=upperdisaggregationplan,
        lowerdisaggregationplan=lowerdisaggregationplan,
        translator=ptranslator,
    )

    i2itranslator = I2ITranslator(
        operator,
        tree,
        samplings,
        parentsamplings,
        relevantoctreelevels,
        polynomial,
        ptranslator;
        InterpolatorType=Interpolator,
        PhaseShifterType=PhaseShifter,
    )

    o2otranslator = O2OTranslator(i2itranslator)

    moments = HybridFMM.moments(
        γ, tree, relevantlevels, operator, polynomial, aggregationplan, samplings
    )

    storagemoments = HybridFMM.storagemoments(
        γ, tree, relevantlevels, operator, polynomial, samplings
    )
    incomingmoments = deepcopy(storagemoments)
    collectedmoments = deepcopy(storagemoments)

    return GalerkinHybridFMM(
        γ,
        operator,
        tree,
        nearinteractions,
        space,
        momentcollections,
        moments,
        incomingmoments,
        storagemoments,
        collectedmoments,
        i2itranslator,
        i2otranslator,
        o2otranslator,
        samplings,
        aggregationplan,
        disaggregationplan,
        minrelevantlevel,
        verbose,
        isnear(tree),
        iFMM.α(operator, polynomial),
    )
end

function samplings(A::GalerkinHybridFMM)
    return A.samplings
end

function sampling(A::GalerkinHybridFMM, level::Int)
    return A.samplings[levelindex(A, level)]
end
