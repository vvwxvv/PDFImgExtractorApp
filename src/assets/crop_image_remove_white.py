from PIL import Image
import numpy as np
import cv2
import os
import glob


class ContourBasedCropper:
    def __init__(self, white_threshold=240, min_contour_area=1000, edge_buffer=5):
        self.white_threshold = white_threshold
        self.min_contour_area = min_contour_area
        self.edge_buffer = edge_buffer

    def preprocess_image(self, img_array):
        """
        Preprocess image for better contour detection.
        """
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        return enhanced

    def find_main_contour(self, gray_image):
        """
        Find the main artwork contour by detecting the boundary between artwork and white space.
        """
        height, width = gray_image.shape

        # Create binary mask - everything that's not white
        binary = (gray_image < self.white_threshold).astype(np.uint8) * 255

        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Fill small holes in the artwork
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Filter contours by area and position
        valid_contours = []
        min_area = max(
            self.min_contour_area, (width * height) * 0.01
        )  # At least 1% of image

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Filter out contours that are too close to edges (likely artifacts)
                margin = min(width, height) * 0.02  # 2% margin
                if (
                    x > margin
                    and y > margin
                    and x + w < width - margin
                    and y + h < height - margin
                ):
                    valid_contours.append(contour)
                elif (
                    area > (width * height) * 0.3
                ):  # Very large contour, probably main artwork
                    valid_contours.append(contour)

        if not valid_contours:
            # If no valid contours found, use the largest one
            largest_contour = max(contours, key=cv2.contourArea)
            return largest_contour

        # Return the largest valid contour
        main_contour = max(valid_contours, key=cv2.contourArea)
        return main_contour

    def get_contour_bounding_box(self, contour, img_shape):
        """
        Get tight bounding box from contour with some buffer.
        """
        height, width = img_shape[:2]

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Add buffer but stay within image bounds
        x = max(0, x - self.edge_buffer)
        y = max(0, y - self.edge_buffer)
        w = min(width - x, w + 2 * self.edge_buffer)
        h = min(height - y, h + 2 * self.edge_buffer)

        return x, y, w, h

    def refine_boundaries_by_density(self, gray_image, initial_bbox):
        """
        Refine boundaries by analyzing content density at edges.
        """
        x, y, w, h = initial_bbox

        # Extract the region
        region = gray_image[y : y + h, x : x + w]
        region_height, region_width = region.shape

        # Analyze edge densities
        edge_thickness = max(1, min(region_height // 20, region_width // 20, 10))

        # Top edge
        top_edge = region[:edge_thickness, :]
        top_density = np.mean(top_edge < self.white_threshold)

        # Bottom edge
        bottom_edge = region[-edge_thickness:, :]
        bottom_density = np.mean(bottom_edge < self.white_threshold)

        # Left edge
        left_edge = region[:, :edge_thickness]
        left_density = np.mean(left_edge < self.white_threshold)

        # Right edge
        right_edge = region[:, -edge_thickness:]
        right_density = np.mean(right_edge < self.white_threshold)

        # If edge has very low density, try to crop it
        density_threshold = 0.05  # 5% non-white pixels

        new_x, new_y = x, y
        new_w, new_h = w, h

        # Adjust top
        if top_density < density_threshold:
            for i in range(edge_thickness, region_height // 3):
                row_density = np.mean(region[i, :] < self.white_threshold)
                if row_density > density_threshold * 2:
                    new_y = y + i
                    new_h = h - i
                    break

        # Adjust bottom
        if bottom_density < density_threshold:
            for i in range(
                region_height - edge_thickness - 1, region_height * 2 // 3, -1
            ):
                row_density = np.mean(region[i, :] < self.white_threshold)
                if row_density > density_threshold * 2:
                    new_h = i + 1 - (new_y - y)
                    break

        # Adjust left
        if left_density < density_threshold:
            for i in range(edge_thickness, region_width // 3):
                col_density = np.mean(region[:, i] < self.white_threshold)
                if col_density > density_threshold * 2:
                    new_x = x + i
                    new_w = w - i
                    break

        # Adjust right
        if right_density < density_threshold:
            for i in range(
                region_width - edge_thickness - 1, region_width * 2 // 3, -1
            ):
                col_density = np.mean(region[:, i] < self.white_threshold)
                if col_density > density_threshold * 2:
                    new_w = i + 1 - (new_x - x)
                    break

        return new_x, new_y, new_w, new_h

    def crop_image(self, image_path, output_path=None, debug=False):
        """
        Main cropping function using contour detection.
        """
        # Load image
        image = Image.open(image_path)
        img_array = np.array(image)
        original_height, original_width = img_array.shape[:2]

        print(f"Processing: {os.path.basename(image_path)}")
        print(f"Original size: {original_width} x {original_height}")

        # Preprocess image
        gray = self.preprocess_image(img_array)

        # Find main contour
        print("Finding main artwork contour...")
        main_contour = self.find_main_contour(gray)

        if main_contour is None:
            print("No valid contour found. Returning original image.")
            if output_path:
                image.save(output_path)
            return image

        # Get bounding box from contour
        x, y, w, h = self.get_contour_bounding_box(main_contour, img_array.shape)
        print(f"Initial contour bbox: x={x}, y={y}, w={w}, h={h}")

        # Refine boundaries
        print("Refining boundaries...")
        final_x, final_y, final_w, final_h = self.refine_boundaries_by_density(
            gray, (x, y, w, h)
        )

        print(f"Final bbox: x={final_x}, y={final_y}, w={final_w}, h={final_h}")

        # Ensure valid boundaries
        final_x = max(0, final_x)
        final_y = max(0, final_y)
        final_w = min(original_width - final_x, final_w)
        final_h = min(original_height - final_y, final_h)

        # Crop the image
        cropped_image = image.crop(
            (final_x, final_y, final_x + final_w, final_y + final_h)
        )

        print(f"Cropped size: {cropped_image.width} x {cropped_image.height}")

        size_reduction = (
            1
            - (cropped_image.width * cropped_image.height)
            / (original_width * original_height)
        ) * 100
        print(f"Size reduction: {size_reduction:.1f}%")

        # Debug visualization
        if debug:
            self.create_debug_visualization(
                img_array,
                main_contour,
                (final_x, final_y, final_w, final_h),
                output_path,
            )

        # Save if output path provided
        if output_path:
            cropped_image.save(output_path, quality=95)
            print(f"Saved to: {output_path}")

        return cropped_image

    def create_debug_visualization(self, img_array, contour, final_bbox, output_path):
        """
        Create debug visualization showing detected contour and crop area.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Original image with contour
            ax1.imshow(img_array)
            ax1.set_title("Detected Main Contour")

            # Draw contour
            if contour is not None:
                contour_points = contour.reshape(-1, 2)
                ax1.plot(
                    contour_points[:, 0],
                    contour_points[:, 1],
                    "r-",
                    linewidth=2,
                    label="Main Contour",
                )

            # Draw final bounding box
            x, y, w, h = final_bbox
            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=3,
                edgecolor="yellow",
                facecolor="none",
                linestyle="--",
                label="Crop Area",
            )
            ax1.add_patch(rect)
            ax1.legend()
            ax1.set_xlim(0, img_array.shape[1])
            ax1.set_ylim(img_array.shape[0], 0)

            # Cropped result
            cropped = img_array[y : y + h, x : x + w]
            ax2.imshow(cropped)
            ax2.set_title("Cropped Result")

            plt.tight_layout()

            if output_path:
                debug_path = output_path.replace(".", "_debug.")
                plt.savefig(debug_path, dpi=150, bbox_inches="tight")
                print(f"Debug visualization saved to: {debug_path}")

            plt.close()

        except ImportError:
            print("Matplotlib not available. Skipping debug visualization.")
        except Exception as e:
            print(f"Could not create debug visualization: {e}")

    def batch_crop(self, input_folder, output_folder, debug=False):
        """
        Batch process multiple images.
        """
        os.makedirs(output_folder, exist_ok=True)

        formats = [
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.bmp",
            "*.tiff",
            "*.webp",
            "*.JPG",
            "*.JPEG",
            "*.PNG",
        ]

        processed = 0
        errors = 0

        for format_ext in formats:
            for image_path in glob.glob(os.path.join(input_folder, format_ext)):
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_filename = f"{name}_cropped{ext}"
                output_path = os.path.join(output_folder, output_filename)

                try:
                    print(f"\n{'-'*50}")
                    self.crop_image(image_path, output_path, debug)
                    processed += 1
                    print(f"✓ Successfully processed: {filename}")
                except Exception as e:
                    print(f"✗ Error processing {filename}: {e}")
                    errors += 1

        print(f"\n{'='*50}")
        print(f"Batch processing completed!")
        print(f"Successfully processed: {processed} images")
        print(f"Errors: {errors} images")


# Simple interface functions
def contour_crop_image(image_path, output_path=None, sensitivity="medium", debug=False):
    """
    Crop image using contour detection.

    Args:
        image_path (str): Path to input image
        output_path (str): Path to save cropped image
        sensitivity (str): 'low', 'medium', 'high' - how aggressive the cropping is
        debug (bool): Show debug visualization

    Returns:
        PIL.Image: Cropped image
    """

    # Configure based on sensitivity
    configs = {
        "low": {"white_threshold": 245, "min_contour_area": 2000, "edge_buffer": 10},
        "medium": {"white_threshold": 240, "min_contour_area": 1000, "edge_buffer": 5},
        "high": {"white_threshold": 235, "min_contour_area": 500, "edge_buffer": 2},
    }

    config = configs.get(sensitivity, configs["medium"])

    cropper = ContourBasedCropper(**config)
    return cropper.crop_image(image_path, output_path, debug)


def batch_contour_crop(input_folder, output_folder, sensitivity="medium", debug=False):
    """
    Batch process images using contour-based cropping.

    Args:
        input_folder (str): Folder containing input images
        output_folder (str): Folder to save cropped images
        sensitivity (str): 'low', 'medium', 'high'
        debug (bool): Create debug visualizations
    """
    configs = {
        "low": {"white_threshold": 245, "min_contour_area": 2000, "edge_buffer": 10},
        "medium": {"white_threshold": 240, "min_contour_area": 1000, "edge_buffer": 5},
        "high": {"white_threshold": 235, "min_contour_area": 500, "edge_buffer": 2},
    }

    config = configs.get(sensitivity, configs["medium"])

    cropper = ContourBasedCropper(**config)
    cropper.batch_crop(input_folder, output_folder, debug)


def is_mostly_white_with_text(
    image,
    white_threshold=240,
    text_area_ratio=0.01,
    min_artwork_area_ratio=0.1,
    max_text_contours=20,
    max_total_text_area_ratio=0.2,
):
    """
    Improved: Returns True if the image is mostly white with only text (even if many text blocks), not a pure artwork/image.
    Args:
        image (PIL.Image): Input image
        white_threshold (int): Threshold for white pixel
        text_area_ratio (float): Max ratio of non-white area to consider as 'only text'
        min_artwork_area_ratio (float): Minimum area ratio for a contour to be considered artwork
        max_text_contours (int): Max number of contours to consider as text page
        max_total_text_area_ratio (float): Max total area of all contours to still consider as text page
    Returns:
        bool: True if image is mostly white with only text, False otherwise
    """
    import numpy as np
    import cv2

    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    total_area = gray.shape[0] * gray.shape[1]
    # Binary mask for non-white
    non_white = (gray < white_threshold).astype(np.uint8)
    non_white_area = np.sum(non_white)
    # If non-white area is very small, it's mostly white
    if non_white_area / total_area < text_area_ratio:
        return True
    # Find contours
    contours, _ = cv2.findContours(
        non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return True
    max_contour_area = max(cv2.contourArea(c) for c in contours)
    # If all contours are small (likely text), treat as mostly white with text
    if max_contour_area / total_area < text_area_ratio:
        return True
    # If no contour is large enough to be artwork, treat as text page
    if max_contour_area / total_area < min_artwork_area_ratio:
        return True
    # Heuristic 1: Many small contours (text blocks)
    if len(contours) > max_text_contours:
        total_contour_area = sum(cv2.contourArea(c) for c in contours)
        if total_contour_area / total_area < max_total_text_area_ratio:
            return True
    # Heuristic 2: Average aspect ratio of contours (text is often long/thin)
    aspect_ratios = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 0:
            aspect_ratios.append(w / h)
    if aspect_ratios:
        avg_aspect = np.mean(aspect_ratios)
        if avg_aspect > 4 and len(contours) > 5:
            return True
    # Heuristic 3: Density of non-white pixels in largest contour (text is sparse)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    contour_pixels = np.sum(mask > 0)
    contour_nonwhite = np.sum((gray < white_threshold) & (mask > 0))
    if contour_pixels > 0 and (contour_nonwhite / contour_pixels) < 0.3:
        return True
    return False


# Example usage
if __name__ == "__main__":
    # Single image processing

    # Low sensitivity (conservative cropping)
    contour_crop_image("input.jpg", "output_low.jpg", sensitivity="low")

    # Medium sensitivity (balanced) - recommended
    contour_crop_image("input.jpg", "output_medium.jpg", sensitivity="medium")

    # High sensitivity (aggressive cropping)
    contour_crop_image("input.jpg", "output_high.jpg", sensitivity="high")

    # With debug visualization
    # contour_crop_image("input.jpg", "output_debug.jpg", sensitivity='medium', debug=True)

    # Batch processing
    # batch_contour_crop("input_folder", "output_folder", sensitivity='medium')

    print("Contour-based cropping completed!")
