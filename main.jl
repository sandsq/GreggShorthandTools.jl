using Random
include("src/shorthand_detector.jl")

# shorthand_detector.shorthand_generator.test()
shorthand_detector.run()

# rng = Xoshiro(0)
# shorthand_detector.Drawer.draw_stroke("/home/sand/.julia/dev/shorthand_detector/data/k_path.png")

# for i in 1:1000
#     path = "/home/sand/.julia/dev/shorthand_detector/data/r/$i.png"
#     shorthand_detector.Drawer.draw_stroke(path; rotation=pi)
# end
