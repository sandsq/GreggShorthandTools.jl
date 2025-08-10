module Drawer

export test_func,
       draw_ellipse,
       draw_stroke

using Luxor

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

function draw_stroke()
    println("saving $(BASE_DIR)/k_path.png")

    Drawing(CANVAS_HEIGHT, CANVAS_WIDTH, "$(BASE_DIR)/k_path.png")
    background("white")


    setline(1)

    # scale(0.25)

    tiny_offset_x = CANVAS_WIDTH / 16
    tiny_offset_y = CANVAS_HEIGHT / 16
    small_offset_x = CANVAS_WIDTH / 8
    small_offset_y = CANVAS_HEIGHT / 8
    medium_offset_x = CANVAS_WIDTH / 4
    medium_offset_y = CANVAS_HEIGHT / 4
    large_offset_x = CANVAS_WIDTH / 2
    large_offset_y = CANVAS_HEIGHT / 2

    # sethue((40, 150, 200))
    sethue("red")
    line(Point(large_offset_x, 0), Point(large_offset_x, CANVAS_HEIGHT))
    line(Point(0, large_offset_y), Point(CANVAS_WIDTH, large_offset_y))
    strokepath()

    sethue("black")
    p1 = Point(CENTER_X - medium_offset_x, CENTER_Y + tiny_offset_y)
    p2 = Point(CENTER_X + medium_offset_x, CENTER_Y - medium_offset_y)
    p3 = Point(CENTER_X + medium_offset_x, CENTER_Y + tiny_offset_y)
    curve(p1, p2, p3)
    p4 = Point(CENTER_X + medium_offset_x, CENTER_Y + medium_offset_y)
    p5 = Point(CENTER_X - medium_offset_x, CENTER_Y - tiny_offset_y)
    curve(p3, p4, p5)


    strokepath()


    finish()
end


function test_func()
    println("test function")
    return "hi"
end

end
