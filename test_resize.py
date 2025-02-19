from PIL import Image, ImageOps

def resize_image(image_path, output_path):
    with Image.open(image_path) as img:
        original_size = img.size  # Save original size in a variable
        img = ImageOps.contain(img, (800, 800))  # Resize while keeping aspect ratio
        new_img = Image.new("RGB", (800, 800), (0, 0, 0))  # Create black background
        new_img.paste(img, ((800 - img.size[0]) // 2, (800 - img.size[1]) // 2))  # Center the image
        new_img.save(output_path)
    
    return original_size  # Return original size instead of saving to JSON

def restore_image(resized_path, original_size, restored_path):
    with Image.open(resized_path) as img:
        img = img.crop(((800 - original_size[0]) // 2, (800 - original_size[1]) // 2, 
                        (800 + original_size[0]) // 2, (800 + original_size[1]) // 2))  # Crop back
        img = img.resize(original_size, Image.Resampling.LANCZOS)  # Resize back
        img.save(restored_path)

# Example usage
original_size = resize_image("input_images/cars.png", "test_out/resized.jpg")
restore_image("test_out/resized.jpg", original_size, "test_out/restored.jpg")
