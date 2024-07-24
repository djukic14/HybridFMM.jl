@sum_type HybridMoment{F,P} :hidden begin
    FarField{F}(::F)
    PolynomialMoment{P}(::P)
end

function LinearAlgebra.mul!(moment::M, momentcollection, y, α, β) where {M<:HybridMoment}
    @cases moment begin
        [FarField, PolynomialMoment](m) =>
            return LinearAlgebra.mul!(m, momentcollection, y, α, β)
    end
end

function Base.copyto!(storagemoment::HybridMoment, moment)
    @cases storagemoment begin
        [FarField, PolynomialMoment](m) => return copyto!(m, moment)
    end
end

function Base.copyto!(storagemoment::H2Factory.AbstractMoment, moment::HybridMoment)
    @cases moment begin
        [FarField, PolynomialMoment](m) => return copyto!(storagemoment, m)
    end
end

function H2Factory.add!(momentA::HybridMoment, momentB)
    @cases momentA begin
        [FarField, PolynomialMoment](m) => return H2Factory.add!(m, momentB)
    end
end

function H2Factory.add!(momentA::H2Factory.AbstractMoment, momentB::HybridMoment)
    @cases momentB begin
        [FarField, PolynomialMoment](m) => return H2Factory.add!(momentA, m)
    end
end

function LinearAlgebra.mul!(y, momentcollection, incomingmoment::HybridMoment, α, β)
    @cases incomingmoment begin
        [FarField, PolynomialMoment](m) =>
            return LinearAlgebra.mul!(y, momentcollection, m, α, β)
    end
end

function H2Factory.conjmul!(moment::HybridMoment, momentcollection, y)
    @cases moment begin
        [FarField, PolynomialMoment](m) => return H2Factory.conjmul!(m, momentcollection, y)
    end
end

function Base.conj!(moment::HybridMoment)
    @cases moment begin
        [FarField, PolynomialMoment](m) => return conj!(m)
    end
end
