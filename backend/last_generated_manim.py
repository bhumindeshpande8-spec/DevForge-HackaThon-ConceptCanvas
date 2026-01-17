from manim import *

class GeneratedScene(Scene):
    def construct(self):
        from manim import *
        class EigenVectorsAnimation(Scene):
            def construct(self):
                # Create a 2x2 matrix
                A = Matrix([[1, 2], [3, 4]])
                # Define the vector V
                V = Vector([1, 1])
                # Calculate the eigenvalues and eigenvectors of A
                eigenvalues, eigenvectors = A.eigenvals(), A.eigenspaces()
                # Create a grid with points representing the vectors and their corresponding eigenvalues and eigenvectors
                grid = VGroup(
                    *[Dot(A.row_vector(i)) for i in range(2)],
                    *[Vector(eigenvector) for eigenvector in eigenvectors],
                    *[Text(f"Eigenvalue {eigenvalue}").next_to(A.row_vector(i), RIGHT) for i, eigenvalue in enumerate(eigenvalues)]
                )
                # Set the initial position of the grid
                self.play(Create(grid))
                # Rotate the first row vector by 45 degrees to highlight an invariant direction
                self.play(Rotate(grid[0], PI / 4))
                self.wait(2)
                # Move the second row vector along the x-axis to highlight another invariant direction
                self.play(Transform(grid[1], grid[1].shift(DOWN * 2)))
                self.wait(2)
