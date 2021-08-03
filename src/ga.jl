using LoopVectorization, Polyester, VectorizedRNG, HDF5

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
    losses::Array{Float64}

    construct::Any
    reload::Any

    function GA(;pool_size::T, run_time::T, pass::T, α::Float64=0.01, input_size::T, output_size::T, hidden_neurons::T, max_runtime::T, label::T, epochs::T, batch::T, monitor::String, input_data::Array{Float32}, output_data::Array{Int8}) where {T<:Int64}
        gene_pool = Array{Float64, 2}(undef, 10,pool_size)
        bounds⁺ = [5.0, 1.0, 0.9, 1.0, max_runtime, max_runtime, 1.0, 1.0, 1.0, 1.0]
        bounds⁻ = [1.0e-5, 1.0e-5, 0.0, 0.0, 1.0, 1.0, 1.0e-5, 1.0e-5, 0.0, 0.0]
        for i in 1:10, j in 1:pool_size
            gene_pool[i,j] = rand(bounds⁻[i]:1.0e-5:bounds⁺[i])
        end
        for i in 1:pool_size
            gene_pool[8,i] = rand(gene_pool[3,i]:1.0e-5:1.0f0)
        end

        losses = zeros(Int64, pool_size,1)

        new{Int64}(pool_size, run_time, pass, α, input_size, output_size, hidden_neurons, max_runtime, label, epochs, batch, monitor, input_data, output_data, gene_pool, losses, construct, reload)
    end

    function construct(self::GA)
        temp_pool = zeros(Float64, size(self.gene_pool))
        weights = zeros(Float64, size(self.losses))
        T = self.run_time
        s = self.input_size+self.output_size+self.hidden_neurons
        r₀ = fill(-sqrt(3/s), s,s)
        rₜ = fill(2*sqrt(3/s), s,s)
        bounds⁺ = [5.0, 1.0, 0.9, 1.0, self.max_runtime, self.max_runtime, 1.0, 1.0, 1.0, 1.0]
        bounds⁻ = [1.0e-5, 1.0e-5, 0.0, 0.0, 1.0, 1.0, 1.0e-5, 1.0e-5, 0.0, 0.0]
        nn = Network(;input_size=self.input_size, output_size=self.output_size, hidden_neurons=self.hidden_neurons, max_runtime=self.max_runtime)

        for t in 1:self.run_time
            println("Generation ", t)

            @time begin
                print("[")
                for i in 1:self.pool_size
                    nn.loss = 0.0
                    rand!(local_rng(), nn.weights, VectorizedRNG.StaticInt(0), r₀, rₜ)
                    self.losses[i,1] = nn.ga_train(nn; input_data=self.input_data, output_data=self.output_data, epochs=self.epochs, label=self.label, batch=self.batch, monitor=self.monitor, 
                                                   Cₘ=self.gene_pool[1,i], gₗ=self.gene_pool[2,i], Eₗ=self.gene_pool[3,i], U=self.gene_pool[4,i], τᵣ=self.gene_pool[5,i], τᵤ=self.gene_pool[6,i], f=self.gene_pool[7,i], θ=self.gene_pool[8,i], A₊=self.gene_pool[9,i], A₋=self.gene_pool[10,i])
                    print("=")
                end
                println("]\nBest result: ", minimum(self.losses))
                display(self.gene_pool[:,argmin(self.losses)[1]])
                println()
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
                    bounds⁻[8] = temp_pool[3,i] + 1.0e-5
                    @avxt for j in eachindex(temp)
                        temp_pool[j,i] += ifelse(temp[j]<=self.α, ifelse(temp[j]<=0.5, (bounds⁺[j]-temp_pool[j,i])*(1.0-temp[j]^(1-t/T)^5), (bounds⁻[j]-temp_pool[j,i])*(1.0-temp[j]^(1-t/T)^5)), temp_pool[j,i])
                        temp_pool[j,i] = ifelse(temp_pool[j,i]<0, 0, temp_pool[j,i])
                        temp_pool[j,i] = ifelse(temp_pool[j,i]>bounds⁺[j], bounds⁺[j], temp_pool[j,i])
                        temp_pool[j,i] = ifelse(temp_pool[j,i]<bounds⁻[j], bounds⁻[j], temp_pool[j,i])
                    end
                end
            end

            @avxt self.gene_pool .= temp_pool
            save(self.gene_pool, "./gen_temp.h5")
        end
    end

    function get_weights!(weights::Array{Float64}, losses::Array{Float64})
        m = minimum(losses)-1
        @avxt for i in eachindex(losses)
            weights[i] = 1/(losses[i]-m)
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

    function save(gene_pool::Array{Float64}, path::String)
        h5open(path, "w") do file
            write(file, "gene_pool", gene_pool)
        end
    end

    function reload(self::GA, path::String)
        h5open(path, "r") do file
            self.gene_pool = read(file, "gene_pool")
        end        
    end
end