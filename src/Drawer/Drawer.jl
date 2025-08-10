module Drawer

export test_func,
       draw_ellipse,
       draw_stroke

using Luxor
import Random: AbstractRNG, default_rng

const BASE_DIR = joinpath(@__DIR__, "..", "..", "data")
const SQUARE_CANVAS_SIZE = 200
const CANVAS_HEIGHT = SQUARE_CANVAS_SIZE
const CANVAS_WIDTH = SQUARE_CANVAS_SIZE
const CENTER_X = CANVAS_WIDTH / 2
const CENTER_Y = CANVAS_HEIGHT / 2

# function draw_ellipse()
#     Drawing(CANVAS_HEIGHT, CANVAS_WIDTH, "$(BASE_DIR)/test.png")
#     text("Hello world", Point(5, 5))
#     circle(CENTER, 15, action = :stroke)
#     finish()
# end

"""
  draw_stroke()

Draw `k` or `g` stroke. Rotate for `r, l, p, b, f, v`
"""
function draw_stroke(rng::AbstractRNG)
    println("saving $(BASE_DIR)/k_path.png")

    Drawing(CANVAS_HEIGHT, CANVAS_WIDTH, "$(BASE_DIR)/k_path.png")
    background("white")

    function line_width_randomizer(min_thickness, max_thickness)
        return min_thickness + rand(rng, Float64) * (max_thickness - min_thickness)
    end
    setline(line_width_randomizer(0.9, 1.5))

    # scale(0.25)

    tiny_offset_base = rand(rng, 16:18)
    small_offset_base = rand(rng, 7:9)
    medium_offset_base = rand(rng, 4:6)
    large_offset_base = rand(rng, 2:3)

    function randomize_ten_percent()
        return 0.5 + rand(rng, Float64) * (1.5 - 0.5)
    end

    tiny_offset_x() = CANVAS_WIDTH / tiny_offset_base * randomize_ten_percent()
    tiny_offset_y() = CANVAS_HEIGHT / tiny_offset_base * randomize_ten_percent()
    small_offset_x = CANVAS_WIDTH / small_offset_base
    small_offset_y = CANVAS_HEIGHT / small_offset_base
    medium_offset_x() = CANVAS_WIDTH / medium_offset_base * randomize_ten_percent()
    medium_offset_y() = CANVAS_HEIGHT / medium_offset_base * randomize_ten_percent()
    large_offset_x() = CANVAS_WIDTH / large_offset_base * randomize_ten_percent()
    large_offset_y = CANVAS_HEIGHT / large_offset_base

    sethue(0.1, 0.6, 0.8)
    # sethue("red")
    line(Point(CANVAS_WIDTH / 2, 0), Point(CANVAS_WIDTH / 2, CANVAS_HEIGHT))
    line(Point(0, CANVAS_HEIGHT / 2), Point(CANVAS_WIDTH, CANVAS_HEIGHT / 2))
    strokepath()

    sethue("black")
    p1 = Point(CENTER_X - medium_offset_x(), CENTER_Y + tiny_offset_y())
    p2 = Point(CENTER_X + medium_offset_x(), CENTER_Y - medium_offset_y())

    should_hook_back = true
    if should_hook_back
        p3_randomized_offset = medium_offset_x()
        p3 = Point(CENTER_X + p3_randomized_offset, CENTER_Y)
        curve(p1, p2, p3)
        strokepath()
        setline(line_width_randomizer(1.5, 2.5))
        p4 = Point(CENTER_X + p3_randomized_offset, CENTER_Y + 0.5 * tiny_offset_y())
        p5 = Point(CENTER_X + p3_randomized_offset * 0.8, CENTER_Y + tiny_offset_y())
        curve(p3, p4, p5)
    else
        p3 = Point(CENTER_X + medium_offset_x(), CENTER_Y + tiny_offset_y())
        curve(p1, p2, p3)
    end


    strokepath()


    finish()
end

function draw_stroke()
    draw_stroke(default_rng())
end


function test_func()
    println("test function")
    return "hi"
end

end
