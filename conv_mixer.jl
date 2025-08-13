using Flux, MLDatasets, Optimisers
using Flux: onehotbatch, onecold, DataLoader, Optimiser, flatten, params
using BSON: @save, @load
include("src/GreggShorthandTools.jl")
using .GreggShorthandTools
using .GreggShorthandTools.Alphabet

const LETTERS_TO_PREDICT = [_K, _G]#, _R, _L, _P, _B, _F, _V, _T, _D, _N, _M]

function ConvMixer(in_channels, kernel_size, patch_size, dim, depth, N_classes)
    f = Chain(
        Conv((patch_size, patch_size), in_channels => dim, gelu; stride=patch_size),
        BatchNorm(dim),
        [
            Chain(
                SkipConnection(Chain(Conv((kernel_size, kernel_size), dim => dim, gelu; pad=SamePad(), groups=dim), BatchNorm(dim)), +),
                Chain(Conv((1, 1), dim => dim, gelu), BatchNorm(dim))
            )
            for i in 1:depth
        ]...,
        AdaptiveMeanPool((1, 1)),
        flatten,
        Dense(dim, N_classes)
    )
    return f
end

function get_data(batchsize; dataset=MLDatasets.CIFAR10, idxs=nothing)
    """
    idxs=nothing gives the full dataset, otherwise (for testing purposes) only the 1:idxs elements of the train set are given.
    """
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Loading Dataset
    if idxs === nothing
        xtrain, ytrain = dataset(:train)[:]
        xtest, ytest = dataset(:test)[:]
    else
        xtrain, ytrain = dataset(:train)[1:idxs]
        xtest, ytest = dataset(:test)[1:Int(idxs / 10)]
    end

    # Reshape Data to comply to Julia's (width, height, channels, batch_size) convention in case there are only 1 channel (eg MNIST)
    if ndims(xtrain) == 3
        w = size(xtrain)[1]
        xtrain = reshape(xtrain, (w, w, 1, :))
        xtest = reshape(xtest, (w, w, 1, :))
    end

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    println("training features $(size(xtrain))")
    println("training labels $(size(ytrain))")

    train_loader = DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest), batchsize=batchsize)

    dev = cpu

    train_data, test_data = GreggShorthandTools.load_from_directory(0.8, LETTERS_TO_PREDICT)
    train_digits, train_labels = train_data.features, train_data.targets
    test_digits, test_labels = test_data.features, test_data.targets

    train_labels_onehot = Flux.onehotbatch(train_labels, LETTERS_TO_PREDICT)
    test_labels_onehot = Flux.onehotbatch(test_labels, LETTERS_TO_PREDICT)

    if ndims(train_digits) == 3
        w = size(train_digits)[1]
        train_digits = reshape(train_digits, (w, w, 1, :))
        test_digits = reshape(test_digits, (w, w, 1, :))
    end

    println("training features $(size(train_digits))")
    println("training labels $(size(train_labels_onehot))")

    train_loader = DataLoader((train_digits |> dev, train_labels_onehot |> dev), batchsize=batchsize, shuffle=true, partial=false)
    test_loader = DataLoader((test_digits |> dev, test_labels_onehot |> dev), batchsize=batchsize, shuffle=true, partial=false)

    return train_loader, test_loader
end

function create_loss_function(dataloader, device)

    function loss(model)
        n = 0
        l = 0.0f0
        acc = 0.0f0

        for (x, y) in dataloader
            x, y = x |> device, y |> device
            z = model(x)
            l += Flux.logitcrossentropy(z, y, agg=sum)
            acc += sum(onecold(z) .== onecold(y))
            n += size(x)[end]
        end
        l / n, acc / n
    end

    return loss

end


function train(n_epochs=100)

    #params: warning, the training can be long with these params
    train_loader, test_loader = get_data(128)
    η = 3e-4
    in_channel = 1
    patch_size = 2
    kernel_size = 7
    dim = 128
    dimPL = 2
    depth = 18
    use_cuda = true

    #logging the losses
    train_save = zeros(n_epochs, 2)
    test_save = zeros(n_epochs, 2)

    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    train_loss_fn = create_loss_function(train_loader, device)
    test_loss_fn = create_loss_function(test_loader, device)

    model = ConvMixer(in_channel, kernel_size, patch_size, dim, depth, length(LETTERS_TO_PREDICT)) |> device

    ps = Flux.trainable(model)
    opt = OptimiserChain(
        WeightDecay(1.0f-3),
        ClipNorm(1.0),
        ADAM(η)
    )

    opt_state = Optimisers.setup(Optimisers.Adam(), model)
    # ∇model = gradient(m -> loss(m(x), y), model)[1]
    # opt_state, model = Optimisers.update!(opt_state, model, ∇model)

    for epoch in 1:n_epochs
        println("epoch $epoch")
        for (x, y) in train_loader
            println("loaded one batch")
            x, y = x |> device, y |> device
            ∇model = gradient(m -> Flux.logitcrossentropy(m(x), y, agg=sum), model)[1]
            # gr = gradient(() -> Flux.logitcrossentropy(model(x), y, agg=sum), model)
            Flux.Optimise.update!(opt_state, model, ∇model)
        end

        #logging
        train_loss, train_acc = train_loss_fn(model) |> cpu
        test_loss, test_acc = test_loss_fn(model) |> cpu
        train_save[epoch, :] = [train_loss, train_acc]
        test_save[epoch, :] = [test_loss, test_acc]

        if epoch % 5 == 0
            @info "Epoch $epoch : Train loss = $train_loss || Validation accuracy = $test_acc."
        end

    end

    model = model |> cpu
    @save "model.bson" model
    @save "losses.bson" train_save test_save
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
