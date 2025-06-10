struct Birth <: AbstractRJMCMCMove end
struct Death <: AbstractRJMCMCMove end
struct Split <: AbstractRJMCMCMove end
struct Merge <: AbstractRJMCMCMove end
struct Move <: AbstractRJMCMCMove end
struct Allocate <: AbstractRJMCMCMove end

inverse_move(::Type{Birth}) = Death
inverse_move(::Type{Death}) = Birth
inverse_move(::Type{Split}) = Merge
inverse_move(::Type{Merge}) = Split