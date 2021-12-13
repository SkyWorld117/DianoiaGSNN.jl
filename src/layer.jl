using LoopVectorization, Polyester

mutable struct Layer
    input_size::Int64
    layer_size::Int64

    weights::Array{Float32}
    voltage::Array{Float32}
    output::Array{Float32}

    fire_rec::Array{Int64}
    
    Pᵢ::Array{Float32}
    M::Array{Float32}

    guide::Bool
    record::Bool
    sample_output::Array{Float32}

    activate
    reset

    function Layer(;input_size::Int64, layer_size::Int64)
        weights = rand(0.0f0:1.0f-5:4.0f-1, layer_size,input_size)
        voltage = zeros(Float32, layer_size)
        output = zeros(Float32, layer_size)

        fire_rec = zeros(Int64, layer_size)

        Pᵢ = zeros(Float32, layer_size,input_size)
        M = zeros(Float32, layer_size)

        new(input_size, layer_size, weights, voltage, output, fire_rec, Pᵢ, M, false, false, Float32[], activate, reset)
    end

    function activate(self::Layer, input::Array{Float32}, η, A₊, A₋, γᵥ, γₚ, γₘ, w₊, w₋, θ, t, r, first_spike, table)
        @batch for l in 1:self.layer_size
            if t-self.fire_rec[l]>=r
                s = 0
                @avx for i in 1:self.input_size
                    s += self.weights[l,i] * input[i]
                    self.Pᵢ[l,i] *= γₚ
                    self.Pᵢ[l,i] += input[i] * A₊
                    self.weights[l,i] += η * self.M[l] * (w₋-self.weights[l,i])
                end
                self.voltage[l] *= γᵥ
                self.voltage[l] += s
            end
        end

        if self.guide || self.record
            if !first_spike
                if maximum(self.voltage)>θ
                    if self.record
                        first_spike = true
                        winner = argmax(self.voltage)
                        @avxt for i in 1:self.layer_size
                            table[winner, i] += self.sample_output[i]
                        end
                    end
                    if self.guide
                        @avxt for l in 1:self.layer_size
                            self.voltage[l] *= self.sample_output[l]
                        end
                    end
                end
            else
                if self.guide
                    @avxt for l in 1:self.layer_size
                        self.voltage[l] *= self.sample_output[l]
                    end
                end
            end
            if self.guide
                if self.voltage[argmax(self.sample_output)]<θ && t!=0 && t%r==0
                    self.voltage[argmax(self.sample_output)] += θ
                end
            end
        end

        @batch for l in 1:self.layer_size
            self.M[l] *= γₘ
            if self.voltage[l]>=θ
                self.voltage[l] = 0
                self.output[l] = 1
                self.fire_rec[l] = t
                @avx for i in 1:self.input_size
                    self.weights[l,i] += η * self.Pᵢ[l,i] * (w₊-self.weights[l,i])
                    self.Pᵢ[l,i] = 0
                end
                self.M[l] -= A₋
            elseif self.output[l]!=0
                self.output[l] = 0
            end
        end

        return first_spike
    end

    function reset(self::Layer, r)
        @avxt self.voltage .= 0.0f0
        @avxt self.Pᵢ .= 0.0f0
        @avxt self.M .= 0.0f0
        @avxt self.fire_rec .= -r
    end
end