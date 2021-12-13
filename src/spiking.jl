function scale(num::Int64, input::Float32)
    u = 1.0/(2*(num-1))
    for i in 1:2:2*(num-1)
        if input<=i*u
            return (i-1)*u
        elseif i*u<input<=(i+1)*u
            return (i+1)*u
        end
    end
end

function Spiking!(input::Array{Float32}, spiking_trains::Array{Float32}, scales::Int64, r, T)
    V₊ = maximum(input)
    V₋ = minimum(input)
    @batch for i in 1:length(input)
        freq = Int64(ceil((r-T-1)*scale(scales, (input[i]-V₋)/(V₊-V₋))+T+1))
        
        #=@avx=# for t in freq:freq:T
            spiking_trains[t,i] = 1
        end
    end

    θ = 0.0
    @avxt for t in 1:T
        s = 0
        for i in 1:length(input)
            s += spiking_trains[t,i]
        end
        θ = max(θ, s)
    end
    #println(θ)
    return θ
end