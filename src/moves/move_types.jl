struct Birth <: AbstractRJMCMCMove end
struct Death <: AbstractRJMCMCMove end
struct Move <: AbstractRJMCMCMove end
struct Allocate <: AbstractRJMCMCMove end
struct UpdateLatent <: AbstractRJMCMCMove end

inverse_move(::Type{Birth}) = Death
inverse_move(::Type{Death}) = Birth