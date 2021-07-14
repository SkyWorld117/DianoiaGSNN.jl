using LoopVectorization, Polyester, VectorizedRNG

mutable struct GA{T<:Int64}
    pool_size::T
    run_time::T
    pass::T
    α::Float64

    input_size::T
    output_size::T
    hidden_neurons::T

    max_runtime::T
    label::T

    epochs::T
    batch::T
    monitor::String

    input_data::Array{Float32}
    output_data::Array{Int8}

    gene_pool::Array{Float64}
    losses::Array{Int64}

    construct::Any

    function GA(;pool_size::T, run_time::T, pass::T, α::Float64=0.01, input_size::T, output_size::T, hidden_neurons::T, max_runtime::T, label::T, epochs::T, batch::T, monitor::String, input_data::Array{Float32}, output_data::Array{Int8}) where {T<:Int64}
        gene_pool = rand(0.0:0.00001:1.0, 5,pool_size)
        losses = zeros(Int64, pool_size,1)

        new{Int64}(pool_size, run_time, pass, α, input_size, output_size, hidden_neurons, max_runtime, label, epochs, batch, monitor, input_data, output_data, gene_pool, losses, construct)
    end

    function construct(self::GA)
        temp_pool = zeros(Float64, size(self.gene_pool))
        weights = zeros(Float64, size(self.losses))
        T = self.run_time

        for t in 1:self.run_time
            println("Generation ", t)

            @time begin
                for i in 1:self.pool_size
                    nn = Network(;input_size=self.input_size, output_size=self.output_size, hidden_neurons=self.hidden_neurons, threshold=self.gene_pool[1,i], droprate=self.gene_pool[2,i], max_runtime=self.max_runtime)
                    nn.ga_train(nn; input_data=self.input_data, output_data=self.output_data, epochs=self.epochs, α=self.gene_pool[3,i], γ=self.gene_pool[4,i], θ=self.gene_pool[5,i], label=self.label, batch=self.batch, monitor=self.monitor)
                    self.losses[i,1] = nn.loss
                end
                println("Best result: ", minimum(self.losses))
                if t==self.run_time
                    break
                end

                get_weights!(weights, self.losses)
                for i in 1:self.pass
                    p = argmin(self.losses)[1]
                    @avxt for j in axes(temp_pool, 1)
                        temp_pool[j,i] = self.gene_pool[j,p]
                    end
                    self.losses[p,1] = maximum(self.losses)+1
                end
                for i in self.pass+1:self.pool_size
                    seed₁ = sample(weights)
                    seed₂ = sample(weights)
                    @avxt temp = self.gene_pool[:,seed₂] .- self.gene_pool[:,seed₁]
                    rand!(local_rng(), temp, VectorizedRNG.StaticInt(0), self.gene_pool[:,seed₁], temp)
                    @avxt for j in axes(temp_pool, 1)
                        temp_pool[j,i] = temp[j]
                    end
                    rand!(local_rng(), temp)
                    @avxt for j in eachindex(temp)
                        temp_pool[j,i] = ifelse(temp[j]<=self.α, ifelse(temp[j]<=0.5, (1.0-temp_pool[j,i])*(1.0-temp[j]^(1-t/T)^5), -(temp_pool[j,i]+1.0)*(1.0-temp[j]^(1-t/T)^5)), temp_pool[j,i])
                    end
                end
            end

            @avxt self.gene_pool .= temp_pool
        end

        display(self.gene_pool[:,argmin(self.losses)[1]])
        println()
    end

    function get_weights!(weights::Array{Float64}, losses::Array{Int64})
        @avxt for i in eachindex(losses)
            weights[i] = 1/losses[i]
        end
        s = sum(weights)
        @avxt weights ./= s
    end

    function sample(weights::Array{Float64})
        r = rand()
        for i in 1:length(weights)
            if weights[i]>=r
                return i
            else
                r -= weights[i]
            end
        end
    end
end