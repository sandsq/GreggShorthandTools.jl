# custom split layer
struct Split{T}
    paths::T
end

Split(paths...) = Split(paths)

Flux.@layer Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

function create_model9959(window_size::Number, n_classes::Number; misc="")
    m = Chain(
        Split(
            Chain(
                Conv((1, window_size), 1 => 8, relu, pad=SamePad()),
                MaxPool((2, 2)),
                Conv((1, window_size), 8 => 16, relu, pad=SamePad()),
                MaxPool((2, 2)),
                flatten,
                # GlobalMeanPool(),
                # MaxPool((1, 1)),
            ),
            Chain(
                Conv((window_size, 1), 1 => 8, relu, pad=SamePad()),
                MaxPool((2, 2)),
                Conv((window_size, 1), 8 => 16, relu, pad=SamePad()),
                MaxPool((2, 2)),
                flatten,
                # GlobalMaxPool(),
                # MaxPool((1, 1)),
            ),
            Chain(
                Conv((window_size + 2, window_size + 2), 1 => 8, relu, pad=SamePad()),
                MaxPool((2, 2)),
                Conv((window_size + 2, window_size + 2), 8 => 16, relu, pad=SamePad()),
                MaxPool((2, 2)),
                flatten,
                # GlobalMaxPool(),
                # MaxPool((1, 1)),
            )),
        Parallel(vcat,
            Dense(2304 => 32, relu),
            Dense(2304 => 32, relu),
            Dense(2304 => 32, relu),
        ),
        # Conv((window_size, window_size), 1 => 8, relu, pad=SamePad()),
        # MaxPool((2, 2)),
        # Conv((window_size, window_size), 8 => 16, relu, pad=SamePad()),
        # MaxPool((2, 2)),
        # Conv((window_size, window_size), 16 => 32, relu, pad=SamePad()),
        # MaxPool((2, 2)),
        # Conv((window_size, window_size), 32 => 64, relu, pad=SamePad()),
        # MaxPool((2, 2)),
        # flatten,

        Dense(96 => 48, relu),
        # Dense(32 => 16, relu),
        # Dense(120 => 84, relu),
        Dense(48 => n_classes),
    ) |> gpu
    return m, "cnn_$(window_size)x$(window_size)_$misc"
end

function create_model(window_size::Number, n_classes::Number; misc="")
    m = Chain(
        Split(
            Chain(
                Conv((1, window_size), 1 => 8, relu, pad=SamePad()),
                MaxPool((2, 2)),
                Conv((1, window_size), 8 => 16, relu, pad=SamePad()),
                MaxPool((2, 2)),
                Conv((1, window_size), 16 => 32, relu, pad=SamePad()),
                MaxPool((2, 2)),
                GlobalMeanPool(),
                # MaxPool((1, 1)),
                flatten,

            ),
            Chain(
                Conv((window_size, 1), 1 => 8, relu, pad=SamePad()),
                MaxPool((2, 2)),
                Conv((window_size, 1), 8 => 16, relu, pad=SamePad()),
                MaxPool((2, 2)),
                Conv((window_size, 1), 16 => 32, relu, pad=SamePad()),
                MaxPool((2, 2)),
                GlobalMaxPool(),
                # MaxPool((1, 1)),
                flatten,

            ),
            Chain(
                Conv((window_size + 2, window_size + 2), 1 => 8, relu, pad=SamePad()),
                MaxPool((2, 2)),
                Conv((window_size + 2, window_size + 2), 8 => 16, relu, pad=SamePad()),
                MaxPool((2, 2)),
                Conv((window_size + 2, window_size + 2), 16 => 32, relu, pad=SamePad()),
                MaxPool((2, 2)),
                GlobalMaxPool(),
                # MaxPool((1, 1)),
                flatten,

            )),
        Parallel(vcat,
            identity,
            identity,
            identity,
            # Dense(32 => 32, relu),
            # Dense(32 => 32, relu),
            # Dense(32 => 32, relu),
        ),
        # Conv((window_size, window_size), 1 => 8, relu, pad=SamePad()),
        # MaxPool((2, 2)),
        # Conv((window_size, window_size), 8 => 16, relu, pad=SamePad()),
        # MaxPool((2, 2)),
        # Conv((window_size, window_size), 16 => 32, relu, pad=SamePad()),
        # MaxPool((2, 2)),
        # Conv((window_size, window_size), 32 => 64, relu, pad=SamePad()),
        # MaxPool((2, 2)),
        # flatten,

        Dense(96 => 48, relu),
        # Dense(32 => 16, relu),
        # Dense(120 => 84, relu),
        Dense(48 => n_classes),
    ) |> gpu
    return m, "cnn_$(window_size)x$(window_size)_$misc"
end

function run_conv(;param_epochs=10, should_load_model=false)
    println("running with $param_epochs epochs")
    println("loading model? $should_load_model")

    folder = "runs"  # sub-directory in which to save
    isdir(folder) || mkdir(folder)


    #===== DATA =====#

    # train_data = MLDatasets.MNIST()
    # println(train_data.split)
    # exit()

    # global_subdirs = ["k", "p", "r", "g"]
    letters_to_predict = [_K, _G, _R, _L, _P, _B, _F, _V, _T, _D, _N, _M, _A, _E]

    train_data, test_data = load_from_directory(0.8, letters_to_predict)
    println("training features $(size(train_data.features))")
    println("training labels $(size(train_data.targets))")
    # println(train_data.features)
    # exit()

    # train_data.features is a 28×28×60000 Array{Float32, 3} of the images.
    # Flux needs a 4D array, with the 3rd dim for channels -- here trivial, grayscale.
    # Combine the reshape needed with other pre-processing:

    function loader(data::Data=train_data; batchsize::Int=64, shuffle=true)
        x4dim = reshape(data.features, IMAGE_SIZE_X, IMAGE_SIZE_Y, 1, :)   # insert trivial channel dim
        yhot = Flux.onehotbatch(data.targets, letters_to_predict)  # make a 10×60000 OneHotMatrix
        # println("x4dim $(size(x4dim))")
        # println("yhot $(size(yhot))")
        Flux.DataLoader((x4dim, yhot); batchsize, shuffle=shuffle) |> gpu
    end

    loader()  # returns a DataLoader, with first element a tuple like this:

    x1, y1 = first(loader()) # (28×28×1×64 Array{Float32, 3}, 10×64 OneHotMatrix(::Vector{UInt32}))

    # If you are using a GPU, these should be CuArray{Float32, 3} etc.
    # If not, the `gpu` function does nothing (except complain the first time).

    #===== MODEL =====#

    # LeNet has two convolutional layers, and our modern version has relu nonlinearities.
    # After each conv layer there's a pooling step. Finally, there are some fully connected layers:
    window_size = 3
    lenet, file_stem = create_model(window_size, length(letters_to_predict); misc="multilayer_conv_in_split_global_maxpool_with_ae")
    filename = joinpath(folder, "$file_stem.jld2")
    details_file = joinpath(folder, "$file_stem.txt")
    @show lenet
    # lenet = Chain(
    #     Conv((window_size, window_size), 1 => 6, relu),
    #     # MaxPool((2, 2)),
    #     Conv((window_size, window_size), 6 => 16, relu),
    #     MaxPool((2, 2)),
    #     Conv((window_size, window_size), 16 => 32, relu),
    #     MaxPool((2, 2)),
    #     Flux.flatten,
    #     Dense(16928 => 1200, relu),
    #     Dense(1200 => 120, relu),
    #     Dense(120 => 84, relu),
    #     Dense(84 => length(letters_to_predict)),
    # ) |> gpu

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



    function loss_and_accuracy(model, data::Data=test_data; should_print_wrong=false)
        d = loader(data; batchsize=length(data))  # make one big batch
        # d = loader(data; batchsize=100)
        (x, y) = only(d)
        ŷ = model(x)
        loss = Flux.logitcrossentropy(ŷ, y)  # did not include softmax in the model
        acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)

        if should_print_wrong
            small_loader = loader(data; batchsize=1, shuffle=false)
            for (i, (x, y)) in enumerate(small_loader)
                # println(y)
                predicted_y = Flux.onecold(softmax(model(x)), letters_to_predict)
                true_y = Flux.onecold(y, letters_to_predict)
                # println(predicted_y)
                # println(true_y)

                if true_y != predicted_y
                    println("$i: predicted $predicted_y, true $true_y")
                end
            end
        end
        (; loss, acc, split=data.split)  # return a NamedTuple
    end

    @show loss_and_accuracy(lenet)

    #===== TRAINING =====#

    # Let's collect some hyper-parameters in a NamedTuple, just to write them in one place.
    # Global variables are fine -- we won't access this from inside any fast loops.

    settings = (;
        eta=3e-4,     # learning rate
        lambda=1e-2,  # for weight decay
        batchsize=128,
        epochs=param_epochs,
    )
    train_log = []

    # lenet_to_load = Flux.@autosize (IMAGE_SIZE_X, IMAGE_SIZE_Y, 1, 1) Chain(
    #     Conv((window_size, window_size), 1 => 6, relu),
    #     MaxPool((2, 2)),
    #     Conv((window_size, window_size), _ => 16, relu),
    #     MaxPool((2, 2)),
    #     Flux.flatten,
    #     Dense(_ => 120, relu),
    #     Dense(_ => 84, relu),
    #     Dense(_ => length(letters_to_predict))
    # )

    model = if should_load_model
        lenet_to_load, _ = create_model(window_size, length(letters_to_predict))
        @show lenet_to_load(cpu(x1)) |> size
        println("loading from $filename")
        loaded_state = JLD2.load(filename, "lenet_state")
        Flux.loadmodel!(lenet_to_load, loaded_state)
        @show loss_and_accuracy(lenet_to_load)
        lenet_to_load
    else
        lenet
    end
    # model = lenet_to_load
    push!(train_log, model)

    # Initialise the storage needed for the optimiser:

    opt_rule = OptimiserChain(WeightDecay(settings.lambda), Adam(settings.eta))
    opt_state = Flux.setup(opt_rule, lenet)



    # println("@@@@@@@@@@@@@@@@@@ skipping training for now by setting epochs to 0")
    for epoch in 1:settings.epochs
        # @time will show a much longer time for the first epoch, due to compilation
        @time for (x, y) in ProgressBar(loader(batchsize=settings.batchsize))
            grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), model)
            Flux.update!(opt_state, model, grads[1])
        end

        # Logging & saving, but not on every epoch
        # if epoch % 2 == 1
        if true
            loss, acc, _ = loss_and_accuracy(model)
            test_loss, test_acc, _ = loss_and_accuracy(model, test_data)
            @info "logging:" epoch acc test_acc
            nt = (; epoch, loss, acc, test_loss, test_acc)  # make a NamedTuple
            push!(train_log, nt)
        end
        if epoch % 5 == 0 || epoch == settings.epochs
            JLD2.jldsave(filename; lenet_state=Flux.state(model) |> cpu)
            println("saved to ", filename, " after ", epoch, " epochs")
        end
    end


    open(details_file, "w") do fo
        for ele in train_log
            println(fo, ele)
        end
    end
    @show train_log

    # We can re-run the quick sanity-check of predictions:
    y1hat = model(x1)
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

    ptest = softmax(model(xtest))
    max_p = maximum(ptest; dims=1)
    _, i = findmin(vec(max_p))

    xtest[:, :, 1, i] .|> Gray |> transpose |> cpu

    Flux.onecold(ytest, letters_to_predict)[i]  # true classification
    ptest[:, i]  # probabilities of all outcomes
    Flux.onecold(ptest[:, i], letters_to_predict)  # uncertain prediction

    #===== ARRAY SIZES =====#

    # A layer like Conv((5, 5), 1=>6) takes 5x5 patches of an image, and matches them to each
    # of 6 different 5x5 filters, placed at every possible position. These filters are here:

    Conv((window_size, window_size), 1 => 6).weight |> summary  # 5×5×1×6 Array{Float32, 4}

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

    lenet2, _ = create_model(window_size, length(letters_to_predict))

    # lenet2 = Flux.@autosize (IMAGE_SIZE_X, IMAGE_SIZE_Y, 1, 1) Chain(
    #     Conv((window_size, window_size), 1 => 6, relu),
    #     MaxPool((2, 2)),
    #     Conv((window_size, window_size), _ => 16, relu),
    #     MaxPool((2, 2)),
    #     Conv((window_size, window_size), 16 => 32, relu),
    #     MaxPool((2, 2)),
    #     Flux.flatten,
    #     Dense(_ => 1200, relu),
    #     Dense(_ => 120, relu),
    #     Dense(_ => 84, relu),
    #     Dense(_ => length(letters_to_predict))
    # )

    # Check that this indeed accepts input the same size as above:

    @show lenet2(cpu(x1)) |> size

    #===== LOADING =====#

    # During training, the code above saves the model state to disk. Load the last version:

    loaded_state = JLD2.load(filename, "lenet_state")

    # Now you would normally re-create the model, and copy all parameters into that.
    # We can use lenet2 from just above:

    Flux.loadmodel!(lenet2, loaded_state)

    # Check that it now agrees with the earlier, trained, model:

    @show lenet2(cpu(x1)) ≈ cpu(model(x1))

    @show test_loss, test_acc, _ = loss_and_accuracy(lenet2, test_data; should_print_wrong=true)

    function predict_on_file(path)
        written_r = FileIO.load(path)
        written_r = Gray.(written_r)
        written_r = imresize(written_r, (IMAGE_SIZE_X, IMAGE_SIZE_Y))
        written_r = reshape(written_r, IMAGE_SIZE_X, IMAGE_SIZE_Y, 1, :)
        vals = softmax(lenet2(written_r))

        println("@@@@@")
        println("$(vals), $(letters_to_predict)")
        prediction = Flux.onecold(softmax(lenet2(written_r)), letters_to_predict)
        println(path)
        println(prediction)
        println()
    end

    predict_on_file("data/t/1.png")
    predict_on_file("data/d/1.png")
    predict_on_file("data/m/1.png")
    predict_on_file("data/n/1.png")

    predict_on_file("data/written_k.png")
    predict_on_file("data/written_k_background_corrected.png")
    predict_on_file("data/r-transformed.png")
    predict_on_file("data/handwritten/k.png")
    predict_on_file("data/handwritten/g.png")
    predict_on_file("data/handwritten/r.png")
    predict_on_file("data/handwritten/l.png")

    predict_on_file("data/handwritten/p.png")
    predict_on_file("data/handwritten/b.png")
    predict_on_file("data/handwritten/f.png")
    predict_on_file("data/handwritten/v.png")

    predict_on_file("data/handwritten/a.png")
    predict_on_file("data/handwritten/e.png")

    #===== THE END =====#
end
