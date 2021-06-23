# DianoiaRNNS

**DianoiaRNNS** is a **r**eal **n**eural **n**etwork **s**imulator. 

It has not been registered yet. You may want to try it like below:

```julia
include("./DianoiaRNNS.jl")
using .DianoiaRNNS
using MLDatasets

train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()

dict = Dict{Int64, Int64}(1=>1, 2=>2, 3=>3, 4=>4, 5=>5, 6=>6, 7=>7, 8=>8, 9=>9, 0=>10)

nn = Network(;input_size=784, output_size=10, hidden_neurons=206, threshold=0.35, droprate=0.1, max_runtime=15)
nn.train(nn; input_data=flatten(train_x,3), output_data=oneHot(train_y,10,dict), epochs=10, α=0.02, γ=0.5, label=8, batch=128, monitor="examples")
```
