import argparse
from PIL import Image
import os


def count_non_black_pixels(image_path):
  """
  Opens an image, counts, and returns the number of non-black pixels.

  A pixel is considered "black" if its R, G, and B values are all 0.
  The alpha (transparency) channel is ignored.

  Args:
      image_path (str): The full path to the image file.

  Returns:
      int: The number of non-black pixels, or None if an error occurs.
  """
  if not os.path.exists(image_path):
    print(f"Error: The file '{image_path}' was not found.")
    return None

  try:
    # Open the image using a context manager
    with Image.open(image_path) as img:
      # Convert the image to RGB mode to standardize pixel format.
      # This handles palettes, RGBA, grayscale, etc.
      rgb_img = img.convert("RGB")

      non_black_count = 0

      # Get all pixel data at once for efficiency
      pixels = list(rgb_img.getdata())

      # Iterate through the pixels
      for r, g, b in pixels:
        # Check if the pixel is not pure black
        if r != 0 or g != 0 or b != 0:
          non_black_count += 1

      return non_black_count

  except Exception as e:
    print(f"An error occurred while processing the image: {e}")
    return None


if __name__ == "__main__":
  # --- Argument Parsing ---
  # Sets up a command-line interface for the script.
  # You can run this script from your terminal and pass the image file as an argument.
  parser = argparse.ArgumentParser(
    description="Count the number of non-black pixels in an image.", epilog="Example: python count_pixels.py your_image.png"
  )

  # The script requires one argument: the path to the image file.
  parser.add_argument("image_file", type=str, help="The path to the image file (e.g., C:/Users/YourUser/Desktop/image.png)")

  args = parser.parse_args()

  # --- Main Execution ---
  # Call the function with the provided file path and store the result.
  count = count_non_black_pixels(args.image_file)

  # If the function executed successfully (didn't return None), print the result.
  if count is not None:
    print("--- Analysis Complete ---")
    print(f"Image File: {os.path.basename(args.image_file)}")
    print(f"Total non-black pixels found: {count}")
