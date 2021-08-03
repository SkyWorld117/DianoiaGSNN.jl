module DianoiaGSNN
    using LinearAlgebra
    BLAS.set_num_threads(1)
    
    include("./network.jl")
    include("./flatten.jl")
    include("./oneHot.jl")
    include("./ga.jl")

    export Network, oneHot, flatten, GA
end