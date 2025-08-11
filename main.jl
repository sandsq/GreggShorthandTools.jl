using Random
using ArgParse
include("src/GreggShorthandTools.jl")
using .GreggShorthandTools
using .GreggShorthandTools.Alphabet


function main(args)
    if "--run" in args
        GreggShorthandTools.run()
    elseif "--generate" in args
        # for letter in [_K, _G, _R, _L, _P, _B, _F, _V]
        #     println("letter is $letter")
        #     for i in 1:2500
        #         path = "/home/sand/.julia/dev/GreggShorthandTools/data/$(to_string(letter))/$i.png"
        #         mkpath(dirname(path))
        #         GreggShorthandTools.Drawer.draw_stroke(Random.default_rng(), letter, path)
        #     end
        # end
        for letter in [_T, _D, _N, _M]
            for i in 1:2500
                path = "/home/sand/.julia/dev/GreggShorthandTools/data/$(to_string(letter))/$i.png"
                mkpath(dirname(path))
                GreggShorthandTools.Drawer.draw_line(Random.default_rng(), letter, path)
            end
        end
    elseif "--test" in args
        GreggShorthandTools.Drawer.draw_stroke(_G, "/home/sand/.julia/dev/GreggShorthandTools/data/g_path.png")
        GreggShorthandTools.Drawer.draw_stroke(_R, "/home/sand/.julia/dev/GreggShorthandTools/data/r_path.png")
        GreggShorthandTools.Drawer.draw_stroke(_P, "/home/sand/.julia/dev/GreggShorthandTools/data/p_path.png")
        GreggShorthandTools.Drawer.draw_stroke(_V, "/home/sand/.julia/dev/GreggShorthandTools/data/v_path.png")
    end
end

main(ARGS)

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
