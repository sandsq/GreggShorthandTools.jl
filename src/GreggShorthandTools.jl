module GreggShorthandTools
include("Alphabet.jl")
using .Alphabet
include("Drawer/Drawer.jl")
using .Drawer

# using AMDGPU
# AMDGPU.allowscalar(true)
using MLDatasets, Flux, JLD2  # this will install everything if necc.
using Statistics: mean  # standard library
using ImageCore, ImageInTerminal, Images
using FileIO

export run

struct Data
    features
    targets
    split
end
function Base.length(d::Data)
    return length(d.targets)
end

function run()

    folder = "runs"  # sub-directory in which to save
    isdir(folder) || mkdir(folder)
    filename = joinpath(folder, "lenet.jld2")

    #===== DATA =====#

    # train_data = MLDatasets.MNIST()
    # println(train_data.split)
    # exit()


    function ready_picture!(picture_path::String, label::Letter, all_features::Vector{Array{Float32,2}}, all_labels::Vector{Letter})
        img = FileIO.load(picture_path)
        img_array = Gray.(img)

        push!(all_features, img_array)
        push!(all_labels, label)
    end

    # global_subdirs = ["k", "p", "r", "g"]
    letters_to_predict = [_K, _P, _R, _G]
    function load_from_directory(proportion::Float64)
        base_dir = joinpath(@__DIR__, "..", "data")
        # "data"

        # hard coded image size for now
        training_features = Vector{Array{Float32,2}}()
        training_labels = Vector{Letter}()

        testing_features = Vector{Array{Float32,2}}()
        testing_labels = Vector{Letter}()

        for subdir_enum in letters_to_predict
            subdir = to_string(subdir_enum)
            full_dir = "$(base_dir)/$(subdir)"
            num_instances = length(readdir(full_dir))
            train_range = 1:floor(Int, num_instances * proportion)
            test_range = ceil(Int, num_instances * proportion):num_instances
            println("train range $(train_range) and test range $(test_range)")
            for picture_number in train_range
                # println("picture number $picture_number")
                ready_picture!("$(full_dir)/$(picture_number).png", subdir_enum, training_features, training_labels)
            end
            for picture_number in test_range
                # println("picture number $picture_number")
                ready_picture!("$(full_dir)/$(picture_number).png", subdir_enum, testing_features, testing_labels)
            end
        end
        training_features = reshape(reduce(hcat, training_features), 50, 50, :)
        testing_features = reshape(reduce(hcat, testing_features), 50, 50, :)
        # training_data = Dict("features" => training_features, "targets" => training_labels)
        # testing_data = Dict("features" => testing_features, "targets" => testing_labels)
        return Data(training_features, training_labels, :train), Data(testing_features, testing_labels, :test)
    end
    train_data, test_data = load_from_directory(0.8)
    println("training features $(size(train_data.features))")
    println("training labels $(size(train_data.targets))")
    # println(train_data.features)
    # exit()

    # train_data.features is a 28×28×60000 Array{Float32, 3} of the images.
    # Flux needs a 4D array, with the 3rd dim for channels -- here trivial, grayscale.
    # Combine the reshape needed with other pre-processing:

    function loader(data::Data=train_data; batchsize::Int=64)
        x4dim = reshape(data.features, 50, 50, 1, :)   # insert trivial channel dim
        yhot = Flux.onehotbatch(data.targets, letters_to_predict)  # make a 10×60000 OneHotMatrix
        Flux.DataLoader((x4dim, yhot); batchsize, shuffle=true) |> gpu
    end

    loader()  # returns a DataLoader, with first element a tuple like this:

    x1, y1 = first(loader()) # (28×28×1×64 Array{Float32, 3}, 10×64 OneHotMatrix(::Vector{UInt32}))

    # If you are using a GPU, these should be CuArray{Float32, 3} etc.
    # If not, the `gpu` function does nothing (except complain the first time).

    #===== MODEL =====#

    # LeNet has two convolutional layers, and our modern version has relu nonlinearities.
    # After each conv layer there's a pooling step. Finally, there are some fully connected layers:

    lenet = Chain(
        Conv((5, 5), 1 => 6, relu),
        MaxPool((2, 2)),
        Conv((5, 5), 6 => 16, relu),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(1296 => 120, relu),
        Dense(120 => 84, relu),
        Dense(84 => length(letters_to_predict)),
    ) |> gpu

    # Notice that most of the parameters are in the final Dense layers.

    # AMDGPU.@allowscalar
    y1hat = lenet(x1)  # try it out

    sum(softmax(y1hat); dims=1)

    # Each column of softmax(y1hat) may be thought of as the network's probabilities
    # that an input image is in each of 10 classes. To find its most likely answer,
    # we can look for the largest output in each column, without needing softmax first.
    # At the moment, these don't resemble the true values at all:

    @show hcat(Flux.onecold(y1hat, letters_to_predict), Flux.onecold(y1, letters_to_predict))

    #===== METRICS =====#

    # We're going to log accuracy and loss during training. There's no advantage to
    # calculating these on minibatches, since MNIST is small enough to do it at once.



    function loss_and_accuracy(model, data::Data=test_data)
        (x, y) = only(loader(data; batchsize=length(data)))  # make one big batch
        ŷ = model(x)
        loss = Flux.logitcrossentropy(ŷ, y)  # did not include softmax in the model
        acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
        (; loss, acc, split=data.split)  # return a NamedTuple
    end

    @show loss_and_accuracy(lenet)  # accuracy about 10%, before training

    #===== TRAINING =====#

    # Let's collect some hyper-parameters in a NamedTuple, just to write them in one place.
    # Global variables are fine -- we won't access this from inside any fast loops.

    settings = (;
        eta=3e-4,     # learning rate
        lambda=1e-2,  # for weight decay
        batchsize=128,
        epochs=10,
    )
    train_log = []

    # Initialise the storage needed for the optimiser:

    opt_rule = OptimiserChain(WeightDecay(settings.lambda), Adam(settings.eta))
    opt_state = Flux.setup(opt_rule, lenet)

    for epoch in 1:settings.epochs
        # @time will show a much longer time for the first epoch, due to compilation
        @time for (x, y) in loader(batchsize=settings.batchsize)
            grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), lenet)
            Flux.update!(opt_state, lenet, grads[1])
        end

        # Logging & saving, but not on every epoch
        if epoch % 2 == 1
            loss, acc, _ = loss_and_accuracy(lenet)
            test_loss, test_acc, _ = loss_and_accuracy(lenet, test_data)
            @info "logging:" epoch acc test_acc
            nt = (; epoch, loss, acc, test_loss, test_acc)  # make a NamedTuple
            push!(train_log, nt)
        end
        if epoch % 5 == 0
            JLD2.jldsave(filename; lenet_state=Flux.state(lenet) |> cpu)
            println("saved to ", filename, " after ", epoch, " epochs")
        end
    end

    @show train_log

    # We can re-run the quick sanity-check of predictions:
    y1hat = lenet(x1)
    @show hcat(Flux.onecold(y1hat, letters_to_predict), Flux.onecold(y1, letters_to_predict))

    #===== INSPECTION =====#



    xtest, ytest = only(loader(test_data, batchsize=length(test_data)))

    # There are many ways to look at images, you won't need ImageInTerminal if working in a notebook.
    # ImageCore.Gray is a special type, whick interprets numbers between 0.0 and 1.0 as shades:

    xtest[:, :, 1, 5] .|> Gray |> transpose |> cpu

    Flux.onecold(ytest, letters_to_predict)[5]  # true label, should match!

    # Let's look for the image whose classification is least certain.
    # First, in each column of probabilities, ask for the largest one.
    # Then, over all images, ask for the lowest such probability, and its index.

    ptest = softmax(lenet(xtest))
    max_p = maximum(ptest; dims=1)
    _, i = findmin(vec(max_p))

    xtest[:, :, 1, i] .|> Gray |> transpose |> cpu

    Flux.onecold(ytest, letters_to_predict)[i]  # true classification
    ptest[:, i]  # probabilities of all outcomes
    Flux.onecold(ptest[:, i], letters_to_predict)  # uncertain prediction

    #===== ARRAY SIZES =====#

    # A layer like Conv((5, 5), 1=>6) takes 5x5 patches of an image, and matches them to each
    # of 6 different 5x5 filters, placed at every possible position. These filters are here:

    Conv((5, 5), 1 => 6).weight |> summary  # 5×5×1×6 Array{Float32, 4}

    # This layer can accept any size of image; let's trace the sizes with the actual input:

    #=

    julia> x1 |> size
    (28, 28, 1, 64)

    julia> lenet[1](x1) |> size  # after Conv((5, 5), 1=>6, relu),
    (24, 24, 6, 64)

    julia> lenet[1:2](x1) |> size  # after MaxPool((2, 2))
    (12, 12, 6, 64)

    julia> lenet[1:3](x1) |> size  # after Conv((5, 5), 6 => 16, relu)
    (8, 8, 16, 64)

    julia> lenet[1:4](x1) |> size  # after MaxPool((2, 2))
    (4, 4, 16, 64)

    julia> lenet[1:5](x1) |> size  # after Flux.flatten
    (256, 64)

    =#

    # Flux.flatten is just reshape, preserving the batch dimesion (64) while combining others (4*4*16).
    # This 256 must match the Dense(256 => 120). Here is how to automate this, with Flux.outputsize:

    lenet2 = Flux.@autosize (50, 50, 1, 1) Chain(
        Conv((5, 5), 1 => 6, relu),
        MaxPool((2, 2)),
        Conv((5, 5), _ => 16, relu),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(_ => 120, relu),
        Dense(_ => 84, relu),
        Dense(_ => length(letters_to_predict))
    )

    # Check that this indeed accepts input the same size as above:

    @show lenet2(cpu(x1)) |> size

    #===== LOADING =====#

    # During training, the code above saves the model state to disk. Load the last version:

    loaded_state = JLD2.load(filename, "lenet_state")

    # Now you would normally re-create the model, and copy all parameters into that.
    # We can use lenet2 from just above:

    Flux.loadmodel!(lenet2, loaded_state)

    # Check that it now agrees with the earlier, trained, model:

    @show lenet2(cpu(x1)) ≈ cpu(lenet(x1))

    function predict_on_file(path)
        println(path)
        written_r = FileIO.load(path)
        written_r = Gray.(written_r)
        written_r = imresize(written_r, (50, 50))
        written_r = reshape(written_r, 50, 50, 1, :)
        vals = softmax(lenet(written_r))
        println("$(vals), $(letters_to_predict)")
        prediction = Flux.onecold(softmax(lenet2(written_r)), letters_to_predict)
        println(prediction)
    end

    predict_on_file("data/written_k.png")
    predict_on_file("data/k_path.png")
    predict_on_file("data/r-transformed.png")
    predict_on_file("data/g_path.png")

    #===== THE END =====#
end
end
