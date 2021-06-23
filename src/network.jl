using LoopVectorization, Polyester

mutable struct Network
    input_size::Int64
    output_size::Int64
    hidden_neurons::Int64

    threshold::Float64
    droprate::Float64
    max_runtime::Int64

    eweights::Array{Int8}
    cweights::Array{Float32}

    value::Array{Float32}
    output::Array{Int8}

    loss::Int64

    activate::Any
    update::Any
    train::Any

    function Network(;input_size::Int64, output_size::Int64, hidden_neurons::Int64, threshold::Float64, droprate::Float64, max_runtime::Int64)
        s = input_size + output_size + hidden_neurons
        eweights = zeros(Int8, s,s)
        #regression = 1.0f0/s
        regression = Float32(threshold/s)
        cweights = fill(regression, s,s)
        @avxt for i in 1:s
            cweights[i,i] = 0.0f0
        end
        value = zeros(Float32, s,1)
        output = zeros(Int8, s,max_runtime)
        new(input_size, output_size, hidden_neurons, threshold, droprate, max_runtime, eweights, cweights, value, output, 0, activate, update, train)
    end

    function activate(nn::Network, input_data::Array{Float32})
        @avxt for i in eachindex(input_data)
            nn.value[i] = input_data[i]
        end

        t = 0
        active = true
        while active && t<nn.max_runtime-1
            t += 1
            active = false
            @avxt for i in eachindex(nn.value)
                s = 0.0f0
                for j in axes(nn.cweights, 2)
                    s += nn.cweights[i,j] * nn.output[j,t]
                end
                nn.value[i] = s + nn.value[i]*(1-nn.droprate)
                nn.value[i] = ifelse(nn.value[i]>1.0f-5, nn.value[i], 0.0f0)

                nn.value[i] = ifelse(nn.value[i]>nn.threshold, nn.value[i]-nn.threshold, nn.value[i])
                nn.output[i,t+1] = ifelse(nn.value[i]>nn.threshold, 1, 0)
            end

            for i in eachindex(nn.value)
                if nn.value[i] >= nn.threshold
                    active = true
                    break
                end
            end
        end
    end

    function update(nn::Network, output_data::Array{Int8}, α::Float64, γ::Float64, label::Int64, monitor::String)
        range = nn.max_runtime÷2
        STDP(nn, output_data, range, α, label, monitor)
        forgetting(nn, γ)
    end

    @inline function STDP(nn::Network, output_data::Array{Int8}, range::Int64, α::Float64, label::Int64, monitor::String)
        s = nn.input_size+nn.hidden_neurons
        loss = 0
        #=@avxt for i in 1:length(output_data)
            loss += ifelse(nn.output[s+i, range+1]==output_data[i], 1, 0)
            nn.output[s+i, range+1] = output_data[i]
        end=#
        @avxt for i in 1:length(output_data)
            loss += ifelse(nn.output[s+i, label]==output_data[i], 1, 0)
            nn.output[s+i, label] = output_data[i]
        end
        if monitor == "absolute"
            nn.loss += loss
        elseif monitor == "examples"
            nn.loss += loss>0 ? 1 : 0
        end

        @batch for i in 2:nn.max_runtime
            for j in i-range:i+range
                try
                    @assert 2<=j<=nn.max_runtime && i!=j
                    Δw = -α/(j-i)
                    t1 = findall(w->w==1, nn.output[:,i])
                    t2 = findall(w->w==1, nn.output[:,j])
                    for x in t1, y in t2
                        if x[1]!=y[1]
                            nn.cweights[x[1],y[1]] += Δw
                        end
                    end
                    #=
                    @avx for x in axes(nn.output, 1), y in axes(nn.output, 1)
                        nn.cweights[i,j] += ifelse(x!=y && nn.output[x,i]==nn.output[y,j]==1, Δw, 0.0f0)
                        nn.cweights[i,j] = ifelse(nn.cweights[i,j]<0.0f0, 0.0f0, nn.cweights[i,j])
                    end
                    =#
                catch AssertionError
                end
            end
        end
    end

    @inline function forgetting(nn::Network, γ::Float64)
        s = nn.input_size + nn.output_size + nn.hidden_neurons
        regression = 1.0f0/s
        @avxt for i in axes(nn.cweights,1), j in axes(nn.cweights,2)
            nn.cweights[i,j] += ifelse(i!=j, (regression-nn.cweights[i,j])*γ, 0)
            nn.cweights[i,j] = ifelse(nn.cweights[i,j]<0.0f0, 0.0f0, nn.cweights[i,j])
        end
    end

    function train(nn::Network; input_data::Array{Float32}, output_data::Array{Int8}, epochs::Int64, α::Float64, γ::Float64, label::Int64, batch::Real=32, monitor::String="absolute")
        batch_size = ceil(Int64, size(input_data,2)/batch)
        current_input_data = zeros(Float32, size(input_data,1))
        current_output_data = zeros(Int8, size(output_data,1))

        for e in 1:epochs
            print("Epoch ", e, "\n[")
            @time begin
                for t in 1:batch_size
                    index = rand(1:size(input_data)[end])
                    @avxt current_input_data .= input_data[:,index]
                    @avxt current_output_data .= output_data[:,index]

                    if t%ceil(batch_size/50)==0
                        print("=")
                    end
                    nn.activate(nn, current_input_data)
                    nn.update(nn, current_output_data, α, γ, label, monitor)
                end
                print("] with loss ", nn.loss, "\nTime usage: ")
                nn.loss = 0
            end
        end
    end
end