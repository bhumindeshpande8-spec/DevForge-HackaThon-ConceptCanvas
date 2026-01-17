from manim import *

class GeneratedScene(Scene):
    def construct(self):
        circle = Circle(color=BLUE)
        self.play(GrowFromCenter(circle))
        self.wait(1)
