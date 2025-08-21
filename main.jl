using Random
using ArgParse
include("src/GreggShorthandTools.jl")
using .GreggShorthandTools
using .GreggShorthandTools.Alphabet

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--run_conv"
        arg_type = Int
        "--load_model"
        default = false
        action = :store_true
        "--generate"
        default = false
        action = :store_true
        "--test"
        default = false
        help = "an option without argument, i.e. a flag"
        action = :store_true
    end

    return parse_args(s)
end


function main()
    args = parse_commandline()
    println(args)
    if args["run_conv"] != nothing
        epochs = args["run_conv"]
        GreggShorthandTools.run_conv(; param_epochs=epochs, should_load_model=args["load_model"])
    elseif "run_st" in keys(args)
        GreggShorthandTools.run_spatial_transformer()
    elseif args["generate"]
        rng = Xoshiro(0)
        num_samples = 5000
        # for letter in [_K, _G, _R, _L, _P, _B, _F, _V]
        #     println("letter is $letter")
        #     for i in 1:num_samples
        #         path = "/home/sand/.julia/dev/GreggShorthandTools/data/$(to_string(letter))/$i.png"
        #         mkpath(dirname(path))
        #         GreggShorthandTools.Drawer.draw_stroke_bezier(rng, letter, path)
        #     end
        # end
        # for letter in [_T, _D, _N, _M]
        #     for i in 1:num_samples
        #         path = "/home/sand/.julia/dev/GreggShorthandTools/data/$(to_string(letter))/$i.png"
        #         mkpath(dirname(path))
        #         GreggShorthandTools.Drawer.draw_line(rng, letter, path)
        #     end
        # end
        for letter in [_A, _E]
            for i in 1:num_samples
                path = "/home/sand/.julia/dev/GreggShorthandTools/data/$(to_string(letter))/$i.png"
                mkpath(dirname(path))
                GreggShorthandTools.Drawer.draw_ae(rng, letter, path)
            end
        end
    elseif args["test"]
        GreggShorthandTools.Drawer.draw_stroke_bezier(_G, "/home/sand/.julia/dev/GreggShorthandTools/data/g_path.png")
        GreggShorthandTools.Drawer.draw_stroke_bezier(_R, "/home/sand/.julia/dev/GreggShorthandTools/data/r_path.png")
        GreggShorthandTools.Drawer.draw_stroke_bezier(_P, "/home/sand/.julia/dev/GreggShorthandTools/data/p_path.png")
        GreggShorthandTools.Drawer.draw_stroke_bezier(_V, "/home/sand/.julia/dev/GreggShorthandTools/data/v_path.png")
        GreggShorthandTools.Drawer.draw_stroke_bezier(_G, "/home/sand/.julia/dev/GreggShorthandTools/data/g_path.png")
        GreggShorthandTools.Drawer.draw_stroke_bezier(_K, "/home/sand/.julia/dev/GreggShorthandTools/data/k_path.png")

        GreggShorthandTools.Drawer.draw_ae(_A, "/home/sand/.julia/dev/GreggShorthandTools/data/a_path.png")
        GreggShorthandTools.Drawer.draw_ae(_E, "/home/sand/.julia/dev/GreggShorthandTools/data/e_path.png")
    end
end

main()

# # GreggShorthandTools.run()

# rng = Xoshiro(0)
# # GreggShorthandTools.Drawer.draw_stroke(_G, "/home/sand/.julia/dev/GreggShorthandTools/data/g_path.png")
# # GreggShorthandTools.Drawer.draw_stroke(_R, "/home/sand/.julia/dev/GreggShorthandTools/data/r_path.png")


# for letter in [_K, _G, _R, _L]
#     println("letter is $letter")
#     for i in 1:1000
#         path = "/home/sand/.julia/dev/GreggShorthandTools/data/$(to_string(letter))/$i.png"
#         mkpath(dirname(path))
#         # println("path is $path")
#         GreggShorthandTools.Drawer.draw_stroke(Random.default_rng(), letter, path)
#     end
# end
