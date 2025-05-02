from manim import *

class VectorAddition(Scene):
    def construct(self):
        # Configure FFmpeg path (uncomment and update if FFmpeg is not in PATH)
        config.ffmpeg_executable = "C:\ffmpeg\bin\ffmpeg.exe"

        # Create a coordinate plane
        plane = NumberPlane(
            x_range=(-5, 5, 1),
            y_range=(-4, 4, 1),
            axis_config={"include_numbers": True, "font_size": 24},
            background_line_style={"stroke_opacity": 0.5}
        )
        self.add(plane)
        self.wait(0.5)

        # Define vectors a = (2, 1) and b = (1, 2)
        vector_a = Arrow(start=ORIGIN, end=np.array([2, 1, 0]), buff=0, color=BLUE)
        vector_b = Arrow(start=ORIGIN, end=np.array([1, 2, 0]), buff=0, color=YELLOW)

        # Add labels for vectors
        label_a = MathTex(r"\vec{a}", font_size=36).next_to(vector_a.get_end(), RIGHT)
        label_b = MathTex(r"\vec{b}", font_size=36).next_to(vector_b.get_end(), UP)

        # Animate vector a
        self.play(Create(vector_a), Write(label_a), run_time=1)
        self.wait(0.5)

        # Animate vector b
        self.play(Create(vector_b), Write(label_b), run_time=1)
        self.wait(0.5)

        # Show vector b translated to the tip of vector a
        vector_b_copy = Arrow(
            start=vector_a.get_end(),
            end=vector_a.get_end() + np.array([1, 2, 0]),
            buff=0,
            color=YELLOW
        )
        label_b_copy = MathTex(r"\vec{b}", font_size=36).next_to(vector_b_copy.get_end(), UP)
        self.play(Create(vector_b_copy), Write(label_b_copy), run_time=1)
        self.wait(0.5)

        # Create and animate the resultant vector (a + b)
        resultant_end = vector_a.get_end() + vector_b.get_end()
        resultant = Arrow(start=ORIGIN, end=resultant_end, buff=0, color=GREEN)
        label_result = MathTex(r"\vec{a} + \vec{b}", font_size=36).next_to(resultant.get_end(), RIGHT)
        self.play(Create(resultant), Write(label_result), run_time=1)
        self.wait(0.5)

        # Draw the parallelogram for vector addition
        parallelogram = Polygon(
            ORIGIN,
            vector_a.get_end(),
            resultant_end,
            vector_b.get_end(),
            stroke_color=WHITE,
            stroke_width=2,
            fill_color=WHITE,
            fill_opacity=0.2
        )
        self.play(Create(parallelogram), run_time=1)
        self.wait(1)

        # Final pause to view the animation
        self.wait(2)
