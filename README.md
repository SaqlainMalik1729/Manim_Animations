Neural Network Animation

This Manim script creates an animation illustrating a neural network's process for handwritten digit classification using the MNIST dataset, focusing on a digit "7".

Execution Instructions

Prerequisites





Python 3.8+: Ensure Python is installed.



Manim: Install the community edition of Manim.



FFmpeg: Required for video rendering. Ensure FFmpeg is installed and accessible.

Installation





Install Manim:

pip install manim



Install FFmpeg:





Windows:





Download FFmpeg from ffmpeg.org.



Extract and place the ffmpeg.exe in a directory, e.g., C:\ffmpeg\bin\ffmpeg.exe.



Update the script's config.ffmpeg_executable path to point to ffmpeg.exe:

config.ffmpeg_executable = "C:\\ffmpeg\\bin\\ffmpeg.exe"



macOS (using Homebrew):

brew install ffmpeg



Linux (using apt):

sudo apt update
sudo apt install ffmpeg

Running the Animation





Save the script as neural_network.py.



Run the script using Manim:

manim -pql neural_network.py NeuralNetwork





-p: Plays the animation after rendering.



-ql: Renders in low quality for faster execution. Use -qh for high quality or -qm for medium quality.



The output video will be saved in the media/videos/ directory.

Notes





Ensure the FFmpeg path in the script matches your system’s FFmpeg installation.



Rendering may take a few minutes depending on quality settings and system performance.



Modify quality flags (-ql, -qm, -qh) based on your needs
