import pygame
import numpy as np
import recognizer as recognizer
from scipy.ndimage import label
import os

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'new_model.h5')
if not os.path.exists(model_path):
    model = recognizer.create_model()

# Initialize pygame
pygame.init()

# Set the dimensions of the drawing board
board_width, board_height = 784, 784
extra_width = 250  # Extra width for displaying the predicted digit

# Create a white canvas
canvas = pygame.display.set_mode((board_width + extra_width, board_height))
canvas.fill((0, 0, 0))

# Create a boolean variable to track if the mouse button is pressed
drawing = False

# Create an empty image array
image_array = np.zeros((board_height, board_width), dtype=np.uint8)

# Initialize font
pygame.font.init()
font = pygame.font.SysFont('Arial', 20)

# Initialize predicted digit
predicted_char = None

# Track previous mouse position
last_pos = None

# Function to save and extract drawn objects as images
def find_drawn_images(image_array):
    # Find connected components (individual drawn objects)
    labeled_array, num_features = label(image_array > 0)
    if num_features == 0:
        print("No drawing found")
        return []  # No drawing found
    
    drawn_images = []
    for i in range(1, num_features + 1):
        # Find the bounding box of each component
        non_black_pixels = np.argwhere(labeled_array == i)
        min_y, min_x = np.min(non_black_pixels, axis=0)
        max_y, max_x = np.max(non_black_pixels, axis=0)

        # Ensure the bounding box is within the image dimensions
        min_y = max(0, min_y - 30)
        min_x = max(0, min_x - 30)
        max_y = min(image_array.shape[0] - 1, max_y + 30)
        max_x = min(image_array.shape[1] - 1, max_x + 30)

        # Extract the bounding box
        bounding_box = image_array[min_y:max_y + 1, min_x:max_x + 1]
        drawn_images.append(bounding_box)

    print(f"{num_features} drawings found")
    return drawn_images

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Start drawing when the mouse button is pressed
            drawing = True
            last_pos = pygame.mouse.get_pos()
        elif event.type == pygame.MOUSEBUTTONUP:
            # Stop drawing and reset the last position
            drawing = False
            last_pos = None
            image_array = pygame.surfarray.array2d(canvas)
        elif event.type == pygame.MOUSEMOTION and drawing:
            # Draw continuously while the mouse is moving
            current_pos = pygame.mouse.get_pos()
            if current_pos[0] < board_width:  # Ensure drawing is within the board
                if last_pos is not None:
                    # Calculate the number of circles to draw
                    distance = max(abs(current_pos[0] - last_pos[0]), abs(current_pos[1] - last_pos[1]))
                    for i in range(distance):
                        x = int(last_pos[0] + (current_pos[0] - last_pos[0]) * i / distance)
                        y = int(last_pos[1] + (current_pos[1] - last_pos[1]) * i / distance)
                        pygame.draw.circle(canvas, (255, 255, 255), (x, y), 14)
                last_pos = current_pos
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Reset the drawing board when the spacebar is pressed
                canvas.fill((0, 0, 0))
                pygame.draw.rect(canvas, (255, 255, 255), (board_width, 0, extra_width, board_height))  # White background for prediction area
                image_array = np.zeros((board_height, board_width), dtype=np.uint8)
                predicted_char = None  # Reset the predicted digit
            elif event.key == pygame.K_RETURN:
                # Predict the digit when the enter key is pressed
                model = recognizer.load_model()
                # Get the current canvas as a Surface
                canvas_surface = pygame.display.get_surface()
                # Extract only the drawing area
                drawing_area = pygame.Surface((board_width, board_height))
                drawing_area.blit(canvas_surface, (0, 0), (0, 0, board_width, board_height))
                
                # Convert the drawing area to a numpy array
                image_array = pygame.surfarray.array2d(drawing_area)
                
                drawn_images = find_drawn_images(image_array)

                # If no drawing has been made, set predicted digit to "-"
                if len(drawn_images) == 0:
                    predicted_char = "-"
                else:
                    # Predict digits for each drawn image
                    predicted_chars = []
                    for img in drawn_images:
                        # Convert the numpy array back to a Surface
                        surface_array = pygame.surfarray.make_surface(img)

                        # Convert the surface to a compatible format for smooth scaling
                        surface_array = surface_array.convert_alpha()

                        # Resize the surface to 28x28 pixels using smoothscale for blending
                        resized_surface = pygame.transform.smoothscale(surface_array, (28, 28))

                        # Convert the resized surface to a grayscale array
                        grayscale_array = np.array(pygame.surfarray.array2d(resized_surface), dtype=np.uint8)

                        # Predict the digit using the model
                        prediction = recognizer.predict(model, grayscale_array)
                        predicted_chars.append(str(prediction))

                    predicted_char = "".join(predicted_chars)  # Update the predicted chars
            elif event.key == pygame.K_q:
                # Quit the game when 'Q' is pressed
                running = False

    # Draw a white background for the prediction area
    pygame.draw.rect(canvas, (255, 255, 255), (board_width, 0, extra_width, board_height))

    # Render the predicted digit text
    text_surface = font.render("Predicted: " + (predicted_char if predicted_char is not None else ""), True, (0, 0, 0))  # Render text in black
    canvas.blit(text_surface, (board_width + 10, board_height // 2))

    # Render instructions
    instructions_text = font.render("Press Enter to predict", True, (0, 0, 0))
    canvas.blit(instructions_text, (board_width + 10, 20))
    restart_text = font.render("Press Space to restart", True, (0, 0, 0))
    canvas.blit(restart_text, (board_width + 10, 50))
    quit_text = font.render("Press Q to quit", True, (0, 0, 0))
    canvas.blit(quit_text, (board_width + 10, 80))

    # Update the display
    pygame.display.flip()

# Quit pygame
pygame.quit()
