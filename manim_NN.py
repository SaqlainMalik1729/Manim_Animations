from manim import *

class NeuralNetwork(Scene):
    def construct(self):
        # Configure FFmpeg path (uncomment and update if FFmpeg is not in PATH)
        # config.ffmpeg_executable = "C:\\ffmpeg\\ffmpeg-7.1.1-full_build\\bin\\ffmpeg.exe"
        config.ffmpeg_executable = "C:\ffmpeg\bin\ffmpeg.exe"
        # Title
        title = Text("Neural Network: Handwritten Digit Classification", font_size=26).to_edge(UP)
        self.play(Write(title, run_time=2))
        self.wait(2)

        # Step 1: Show handwritten digit (28x28 grid with bright MNIST "7")
        digit_label = Text("Input: Handwritten Digit '7'", font_size=16).shift(UP * 2.5 + LEFT * 4.5)
        # Create 28x28 grid (784 pixels, smaller scale)
        pixel_grid = VGroup(*[Square(side_length=0.11) for _ in range(784)]).arrange_in_grid(28, 28, buff=0).scale(0.9).shift(LEFT * 4.5)
        # Hard-coded MNIST "7" pixel intensities (28x28, 0–255)
        mnist_7 = [
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,84,185,159,151,60,36,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,222,254,254,254,254,241,198,198,198,198,198,198,170,52,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,67,114,72,114,163,227,254,225,254,254,254,250,229,254,254,140,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17,66,14,67,67,67,59,21,236,254,106,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,83,253,209,18,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,233,255,83,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,254,238,44,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,59,249,254,62,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,187,5,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,205,248,58,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,126,254,182,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,251,240,57,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,221,254,166,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,203,254,219,35,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,254,254,77,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,224,254,115,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,254,52,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,242,254,254,52,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,254,219,40,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,207,18,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        ]
        # Flatten 2D array and normalize to opacity (0.2–1.0 for brightness)
        intensities = [0.2 + 0.8 * (mnist_7[i // 28][i % 28] / 255) for i in range(784)]
        # Animate grid creation row-by-row
        for row in range(28):
            row_pixels = VGroup(*[pixel_grid[i] for i in range(row * 28, (row + 1) * 28)])
            for i, pixel in enumerate(row_pixels):
                pixel.set_fill(WHITE, opacity=intensities[row * 28 + i])
            self.play(Create(row_pixels, run_time=0.1), run_time=0.1)
        self.play(Write(digit_label, run_time=2))
        self.wait(2)

        # Step 2: Show input as 28x28 matrix and flatten to 1D array
        matrix_label = Text("Input as 28x28 Matrix", font_size=16).shift(DOWN * 2 + LEFT * 4.5)
        # Create a simplified 28x28 matrix (0 or 1 for clarity)
        simplified_values = [[1 if mnist_7[i][j] > 50 else 0 for j in range(28)] for i in range(28)]
        matrix = Matrix(
            simplified_values,
            element_to_mobject_config={"font_size": 8},
            v_buff=0.1,
            h_buff=0.1
        ).scale(0.4).next_to(matrix_label, DOWN, buff=0.3)
        self.play(Write(matrix_label, run_time=2), Create(matrix, run_time=3))
        self.wait(2)

        # Define input_nodes early for arrow creation
        input_positions = [-3.5 + i for i in range(8)]  # [-3.5, -2.5, ..., 3.5]
        input_nodes = VGroup(*[Circle(radius=0.1, color=BLUE).shift(UP * i * 0.45 + RIGHT * 0) for i in input_positions])

        # Flatten matrix to 1D array
        array_label = Text("Flattened to 784x1 Array", font_size=16).move_to(matrix_label)
        # Create 1D array (show first few and last few elements for brevity)
        flat_values = [simplified_values[i // 28][i % 28] for i in range(784)]
        display_values = flat_values[:5] + ["..."] + flat_values[-5:]
        array = Matrix(
            [[v] for v in display_values],
            element_to_mobject_config={"font_size": 12},
            v_buff=0.2,
            h_buff=0.2
        ).scale(0.5).next_to(array_label, DOWN, buff=0.3)
        # Create a copy of array to avoid transformation issues
        array_copy = array.copy()
        self.play(
            Transform(matrix_label, array_label, run_time=2),
            Transform(matrix, array_copy, run_time=3)
        )
        # Show array feeding into input layer
        array_right = array.get_right()
        input_left = input_nodes[0].get_left()
        array_arrows = VGroup(*[
            Arrow(array_right, input_left, color=WHITE, buff=0.1)
            for _ in range(3)
        ])
        self.play(Create(array_arrows, run_time=2))
        self.wait(2)
        self.play(FadeOut(matrix_label, matrix, array_copy, array_arrows, run_time=2))

        # Step 3: Introduce ANN architecture with two hidden layers
        ann_label = Text("Neural Network Architecture", font_size=16).move_to(digit_label)
        self.play(Transform(digit_label, ann_label, run_time=2))
        self.wait(2)

        # Create remaining layers (input_nodes already defined)
        hidden1_positions = [-1.5 + i for i in range(4)]  # [-1.5, -0.5, 0.5, 1.5]
        hidden2_positions = [-1 + i for i in range(3)]  # [-1, 0, 1]
        output_positions = [i for i in range(10)]  # [0, 1, ..., 9]
        
        hidden1_nodes = VGroup(*[Circle(radius=0.1, color=YELLOW).shift(UP * i * 0.6 + RIGHT * 2) for i in hidden1_positions])
        hidden2_nodes = VGroup(*[Circle(radius=0.1, color=ORANGE).shift(UP * i * 0.6 + RIGHT * 4) for i in hidden2_positions])
        output_nodes = VGroup(*[Circle(radius=0.1, color=GREEN).shift(UP * (i - 4.5) * 0.3 + RIGHT * 6) for i in output_positions])

        # Labels for layers
        input_label = Text("Input\n(784)", font_size=12).next_to(input_nodes, LEFT, buff=0.15)
        hidden1_label = Text("Hidden 1\n(16)", font_size=12).next_to(hidden1_nodes, UP, buff=0.15)
        hidden2_label = Text("Hidden 2\n(8)", font_size=12).next_to(hidden2_nodes, UP, buff=0.15)
        output_label = Text("Output\n(0–9)", font_size=12).next_to(output_nodes, RIGHT, buff=0.15)
        output_digits = VGroup(*[Text(str(i), font_size=10).next_to(output_nodes[i], RIGHT, buff=0.1) for i in range(10)])

        # Draw connections
        input_hidden1_edges = VGroup(*[
            Line(input_nodes[i].get_center(), hidden1_nodes[j].get_center(), color=WHITE, stroke_opacity=0.3, stroke_width=0.7)
            for i in range(len(input_nodes)) for j in range(len(hidden1_nodes))
        ])
        hidden1_hidden2_edges = VGroup(*[
            Line(hidden1_nodes[i].get_center(), hidden2_nodes[j].get_center(), color=WHITE, stroke_opacity=0.3, stroke_width=0.7)
            for i in range(len(hidden1_nodes)) for j in range(len(hidden2_nodes))
        ])
        hidden2_output_edges = VGroup(*[
            Line(hidden2_nodes[i].get_center(), output_nodes[j].get_center(), color=WHITE, stroke_opacity=0.3, stroke_width=0.7)
            for i in range(len(hidden2_nodes)) for j in range(len(output_nodes))
        ])

        # Group all network elements for easy fading
        network_group = VGroup(
            input_nodes, hidden1_nodes, hidden2_nodes, output_nodes,
            input_label, hidden1_label, hidden2_label, output_label,
            input_hidden1_edges, hidden1_hidden2_edges, hidden2_output_edges,
            output_digits
        )

        # Animate ANN creation
        self.play(
            Create(input_nodes, run_time=2), Write(input_label, run_time=2),
            Create(hidden1_nodes, run_time=2), Write(hidden1_label, run_time=2),
            Create(hidden2_nodes, run_time=2), Write(hidden2_label, run_time=2),
            Create(output_nodes, run_time=2), Write(output_label, run_time=2),
            Write(output_digits, run_time=2),
            Create(input_hidden1_edges, run_time=2),
            Create(hidden1_hidden2_edges, run_time=2),
            Create(hidden2_output_edges, run_time=2)
        )
        self.wait(2)

        # Step 4: Forward propagation (slow and detailed)
        forward_label = Text("Forward Propagation", font_size=16).move_to(digit_label)
        self.play(Transform(digit_label, forward_label, run_time=2))
        self.wait(2)

        # Pixels to input layer
        pixel_arrows = VGroup(*[Arrow(pixel_grid[200 + i].get_center(), input_nodes[0].get_center(), color=WHITE, buff=0.1) for i in range(3)])
        weight_label = Text("Weights (W₁)", font_size=12).next_to(pixel_arrows, UP, buff=0.1)
        bias_label = Text("Bias (b₁)", font_size=12).next_to(input_nodes, DOWN, buff=0.1)
        self.play(
            Create(pixel_arrows, run_time=2),
            Write(weight_label, run_time=2),
            Write(bias_label, run_time=2),
            input_nodes.animate.set_fill(BLUE, opacity=0.8),
            run_time=2
        )
        self.wait(2)
        self.play(FadeOut(pixel_arrows, weight_label, bias_label, run_time=2))

        # Input to hidden layer 1
        weight_label = Text("Weights (W₂)", font_size=12).next_to(input_hidden1_edges, UP, buff=0.1)
        bias_label = Text("Bias (b₂)", font_size=12).next_to(hidden1_nodes, DOWN, buff=0.1)
        self.play(
            input_hidden1_edges.animate.set_stroke(opacity=1),
            Write(weight_label, run_time=2),
            Write(bias_label, run_time=2),
            hidden1_nodes.animate.set_fill(YELLOW, opacity=0.8),
            run_time=2
        )
        self.wait(2)
        self.play(FadeOut(weight_label, bias_label, run_time=2))

        # Hidden layer 1 to hidden layer 2
        weight_label = Text("Weights (W₃)", font_size=12).next_to(hidden1_hidden2_edges, UP, buff=0.1)
        bias_label = Text("Bias (b₃)", font_size=12).next_to(hidden2_nodes, DOWN, buff=0.1)
        self.play(
            hidden1_hidden2_edges.animate.set_stroke(opacity=1),
            Write(weight_label, run_time=2),
            Write(bias_label, run_time=2),
            hidden2_nodes.animate.set_fill(ORANGE, opacity=0.8),
            run_time=2
        )
        self.wait(2)
        self.play(FadeOut(weight_label, bias_label, run_time=2))

        # Hidden layer 2 to output
        weight_label = Text("Weights (W₄)", font_size=12).next_to(hidden2_output_edges, UP, buff=0.1)
        bias_label = Text("Bias (b₄)", font_size=12).next_to(output_nodes, DOWN, buff=0.1)
        self.play(
            hidden2_output_edges.animate.set_stroke(opacity=1),
            Write(weight_label, run_time=2),
            Write(bias_label, run_time=2),
            output_nodes.animate.set_fill(GREEN, opacity=0.8),
            run_time=2
        )
        self.wait(2)
        self.play(FadeOut(weight_label, bias_label, run_time=2))

        # Step 5: ReLU Activation (Hidden Layers)
        self.play(
            FadeOut(network_group, pixel_grid, digit_label, run_time=1.5),
            title.animate.shift(UP * 2)
        )
        relu_label = Text("ReLU Activation (Hidden Layers)", font_size=22).shift(UP * 2)
        relu_eq = MathTex(r"f(x) = \max(0, x)", font_size=34).next_to(relu_label, DOWN, buff=0.5)
        relu_desc = Text("Applies non-linearity to hidden neurons", font_size=16).next_to(relu_eq, DOWN, buff=0.5)
        self.play(
            Write(relu_label, run_time=2),
            Write(relu_eq, run_time=2),
            Write(relu_desc, run_time=2)
        )
        self.wait(3)
        self.play(
            FadeOut(relu_label, relu_eq, relu_desc, run_time=1.5),
            FadeIn(network_group, pixel_grid, digit_label, run_time=1.5),
            title.animate.shift(DOWN * 2)
        )
        self.play(
            hidden1_nodes.animate.set_fill(YELLOW, opacity=1),
            hidden2_nodes.animate.set_fill(ORANGE, opacity=1),
            run_time=2
        )
        self.wait(2)

        # Step 6: Perceptron Equation
        self.play(
            FadeOut(network_group, pixel_grid, digit_label, run_time=1.5),
            title.animate.shift(UP * 2)
        )
        perceptron_label = Text("Single Perceptron", font_size=22).shift(UP * 2)
        perceptron_eq = MathTex(r"z = \sum_{i} w_i x_i + b, \quad a = \sigma(z)", font_size=34).next_to(perceptron_label, DOWN, buff=0.5)
        perceptron_desc = Text("Computes weighted sum and applies activation", font_size=16).next_to(perceptron_eq, DOWN, buff=0.5)
        self.play(
            Write(perceptron_label, run_time=2),
            Write(perceptron_eq, run_time=2),
            Write(perceptron_desc, run_time=2)
        )
        self.wait(3)
        self.play(
            FadeOut(perceptron_label, perceptron_eq, perceptron_desc, run_time=1.5),
            FadeIn(network_group, pixel_grid, digit_label, run_time=1.5),
            title.animate.shift(DOWN * 2)
        )
        self.wait(2)

        # Step 7: Softmax Activation (Output Layer)
        self.play(
            FadeOut(network_group, pixel_grid, digit_label, run_time=1.5),
            title.animate.shift(UP * 2)
        )
        softmax_label = Text("Softmax Activation (Output Layer)", font_size=22).shift(UP * 2)
        softmax_eq = MathTex(r"\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}", font_size=34).next_to(softmax_label, DOWN, buff=0.5)
        softmax_desc = Text("Converts outputs to probabilities", font_size=16).next_to(softmax_eq, DOWN, buff=0.5)
        self.play(
            Write(softmax_label, run_time=2),
            Write(softmax_eq, run_time=2),
            Write(softmax_desc, run_time=2)
        )
        self.wait(3)
        self.play(
            FadeOut(softmax_label, softmax_eq, softmax_desc, run_time=1.5),
            FadeIn(network_group, pixel_grid, digit_label, run_time=1.5),
            title.animate.shift(DOWN * 2)
        )
        self.play(output_nodes.animate.set_fill(GREEN, opacity=1), run_time=2)
        
        # Show output probabilities
        probabilities = [0.01, 0.01, 0.02, 0.03, 0.01, 0.05, 0.02, 0.85, 0.03, 0.02]
        prob_labels = VGroup(*[
            Text(f"{prob:.2f}", font_size=10).next_to(output_nodes[i], RIGHT, buff=0.3)
            for i, prob in enumerate(probabilities)
        ])
        prob_bars = VGroup(*[
            Rectangle(height=prob * 1.0, width=0.1, fill_color=GREEN, fill_opacity=0.8)
            .next_to(output_nodes[i], RIGHT, buff=0.5)
            for i, prob in enumerate(probabilities)
        ])
        self.play(
            Write(prob_labels, run_time=2),
            Create(prob_bars, run_time=2),
            output_nodes[7].animate.set_fill(GREEN, opacity=1),
            run_time=2
        )
        self.wait(2)

        # Step 8: Cross-Entropy Loss
        self.play(
            FadeOut(network_group, pixel_grid, digit_label, run_time=1.5),
            title.animate.shift(UP * 2)
        )
        loss_label = Text("Cross-Entropy Loss", font_size=22).shift(UP * 2)
        loss_eq = MathTex(r"L = -\sum_{i=1}^{10} y_i \log(\hat{y}_i)", font_size=34).next_to(loss_label, DOWN, buff=0.5)
        loss_desc = Text("Measures error between prediction and true label", font_size=16).next_to(loss_eq, DOWN, buff=0.5)
        self.play(
            Write(loss_label, run_time=2),
            Write(loss_eq, run_time=2),
            Write(loss_desc, run_time=2)
        )
        self.wait(3)
        self.play(
            FadeOut(loss_label, loss_eq, loss_desc, run_time=1.5),
            FadeIn(network_group, pixel_grid, digit_label, run_time=1.5),
            title.animate.shift(DOWN * 2)
        )
        self.wait(2)

        # Step 9: Highlight prediction
        prediction = Text("Predicted Digit: 7 (Probability: 0.85)", font_size=16).move_to(digit_label)
        self.play(
            Transform(digit_label, prediction, run_time=2),
            output_nodes[7].animate.scale(1.5).set_fill(GREEN, opacity=1),
            prob_bars[7].animate.set_fill(GOLD, opacity=1),
            run_time=2
        )
        self.wait(3)

        # Fade out
        self.play(
            FadeOut(title, digit_label, pixel_grid, network_group, prob_labels, prob_bars),
            run_time=3
        )
        self.wait(2)
