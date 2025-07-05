import os
import io
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
import hashlib
import logging
from typing import Optional, Set, Union
from dataclasses import dataclass
from enum import Enum
import time


class ImageFormat(Enum):
    """Supported image formats."""

    WEBP = "WEBP"
    JPEG = "JPEG"
    PNG = "PNG"


@dataclass
class ExtractionConfig:
    """Configuration for PDF image extraction."""

    target_size_kb: int = 700
    image_format: ImageFormat = ImageFormat.WEBP
    image_quality: int = 85
    dpi: int = 150
    max_compression_attempts: int = 10
    min_quality: int = 10
    resize_factor: float = 0.9
    skip_duplicates: bool = True
    render_fallback: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.target_size_kb <= 0:
            raise ValueError("target_size_kb must be positive")
        if not (1 <= self.image_quality <= 100):
            raise ValueError("image_quality must be between 1 and 100")
        if self.dpi <= 0:
            raise ValueError("dpi must be positive")
        if not (0.1 <= self.resize_factor <= 1.0):
            raise ValueError("resize_factor must be between 0.1 and 1.0")


@dataclass
class ExtractionStats:
    """Statistics from PDF image extraction."""

    total_pages: int = 0
    images_extracted: int = 0
    pages_rendered: int = 0
    duplicates_skipped: int = 0
    errors: int = 0
    processing_time: float = 0.0
    output_folder: Optional[Path] = None


class PDFImageExtractorError(Exception):
    """Base exception for PDF image extraction errors."""

    pass


class PDFImageExtractor:
    """
    Production-level PDF image extractor with comprehensive error handling,
    logging, and configurable compression settings.
    """

    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the PDF image extractor.

        Args:
            config: Configuration object for extraction settings
            logger: Optional logger instance
        """
        self.config = config or ExtractionConfig()
        self.logger = logger or self._setup_logger()
        self._seen_hashes: Set[str] = set()
        self._stats = ExtractionStats()

    def _setup_logger(self) -> logging.Logger:
        """Set up default logger if none provided."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def extract_images(
        self, pdf_path: Union[str, Path], output_folder: Union[str, Path]
    ) -> ExtractionStats:
        """
        Extract and compress images from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            output_folder: Directory to save extracted images

        Returns:
            ExtractionStats object with processing statistics

        Raises:
            PDFImageExtractorError: If extraction fails
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)
        output_folder = Path(output_folder)

        # Reset statistics
        self._stats = ExtractionStats()
        self._stats.output_folder = output_folder
        self._seen_hashes.clear()

        try:
            self._validate_inputs(pdf_path)
            self._create_output_folder(output_folder)

            with fitz.open(str(pdf_path)) as pdf_document:
                self._stats.total_pages = len(pdf_document)
                self.logger.info(f"Processing PDF with {self._stats.total_pages} pages")

                for page_num in range(len(pdf_document)):
                    try:
                        self._process_page(pdf_document, page_num, output_folder)
                    except Exception as e:
                        self._stats.errors += 1
                        self.logger.error(f"Error processing page {page_num + 1}: {e}")

        except Exception as e:
            raise PDFImageExtractorError(f"Failed to extract images: {e}") from e
        finally:
            self._stats.processing_time = time.time() - start_time
            self._log_completion_stats()

        return self._stats

    def _validate_inputs(self, pdf_path: Path) -> None:
        """Validate input parameters."""
        if not pdf_path.exists():
            raise PDFImageExtractorError(f"PDF file not found: {pdf_path}")
        if not pdf_path.is_file():
            raise PDFImageExtractorError(f"Path is not a file: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise PDFImageExtractorError(f"File is not a PDF: {pdf_path}")

    def _create_output_folder(self, output_folder: Path) -> None:
        """Create output folder if it doesn't exist."""
        try:
            output_folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise PDFImageExtractorError(f"Cannot create output folder: {e}") from e

    def _process_page(
        self, pdf_document: fitz.Document, page_num: int, output_folder: Path
    ) -> None:
        """Process a single page of the PDF."""
        page = pdf_document.load_page(page_num)
        images = page.get_images(full=True)

        if not images and self.config.render_fallback:
            self._render_page_as_image(page, page_num, output_folder)
            return
        elif not images:
            self.logger.warning(f"No images found on page {page_num + 1}")
            return

        for img_index, img in enumerate(images):
            try:
                self._extract_embedded_image(
                    pdf_document, img, page_num, img_index, output_folder
                )
            except Exception as e:
                self._stats.errors += 1
                self.logger.error(
                    f"Failed to extract image {img_index + 1} from page {page_num + 1}: {e}"
                )

    def _extract_embedded_image(
        self,
        pdf_document: fitz.Document,
        img: tuple,
        page_num: int,
        img_index: int,
        output_folder: Path,
    ) -> None:
        """Extract a single embedded image."""
        xref = img[0]
        base_image = pdf_document.extract_image(xref)
        image_bytes = base_image["image"]

        if self.config.skip_duplicates:
            image_hash = hashlib.md5(image_bytes).hexdigest()
            if image_hash in self._seen_hashes:
                self._stats.duplicates_skipped += 1
                self.logger.debug(f"Skipped duplicate image on page {page_num + 1}")
                return
            self._seen_hashes.add(image_hash)

        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Generate filename
        filename = self._generate_filename(page_num, img_index, embedded=True)

        # Save compressed image
        self._save_compressed_image(image, output_folder, filename)
        self._stats.images_extracted += 1

    def _render_page_as_image(
        self, page: fitz.Page, page_num: int, output_folder: Path
    ) -> None:
        """Render entire page as image (fallback method)."""
        self.logger.info(f"Rendering page {page_num + 1} as image")

        zoom = self.config.dpi / 72  # PDF default is 72 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert to PIL Image
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Generate filename
        filename = self._generate_filename(page_num, 0, embedded=False)

        # Save compressed image
        self._save_compressed_image(image, output_folder, filename)
        self._stats.pages_rendered += 1
        self._stats.images_extracted += 1

    def _generate_filename(self, page_num: int, img_index: int, embedded: bool) -> str:
        """Generate filename for extracted image."""
        ext = self.config.image_format.value.lower()
        if embedded:
            return f"page_{page_num + 1}_img_{img_index + 1}.{ext}"
        else:
            return f"page_{page_num + 1}_rendered.{ext}"

    def _save_compressed_image(
        self, image: Image.Image, output_folder: Path, filename: str
    ) -> None:
        """Save image with compression to target size."""
        try:
            compressed_image = self._compress_image_to_target_size(image)
            file_path = output_folder / filename

            # Save with format-specific options
            save_kwargs = {
                "format": self.config.image_format.value,
                "quality": self.config.image_quality,
                "optimize": True,
            }

            if self.config.image_format == ImageFormat.WEBP:
                save_kwargs["method"] = 6  # Best compression
            elif self.config.image_format == ImageFormat.JPEG:
                save_kwargs["progressive"] = True

            compressed_image.save(str(file_path), **save_kwargs)

            # Log file size
            file_size_kb = file_path.stat().st_size / 1024
            self.logger.info(f"Saved {filename} ({file_size_kb:.1f} KB)")

        except Exception as e:
            raise PDFImageExtractorError(f"Failed to save image {filename}: {e}") from e

    def _compress_image_to_target_size(self, image: Image.Image) -> Image.Image:
        """
        Compress image to approach target size through iterative resizing and quality adjustment.
        """
        target_size_bytes = self.config.target_size_kb * 1024
        current_image = image.copy()
        current_quality = self.config.image_quality

        for _ in range(self.config.max_compression_attempts):
            # Test current settings
            buffer = io.BytesIO()
            current_image.save(
                buffer,
                format=self.config.image_format.value,
                quality=current_quality,
                optimize=True,
            )
            current_size = buffer.tell()

            # Check if we've reached target or minimum quality
            if (
                current_size <= target_size_bytes
                or current_quality <= self.config.min_quality
            ):
                buffer.seek(0)
                return Image.open(buffer)

            # Resize image
            width, height = current_image.size
            new_width = int(width * self.config.resize_factor)
            new_height = int(height * self.config.resize_factor)
            current_image = current_image.resize((new_width, new_height), Image.LANCZOS)

            # Reduce quality
            current_quality = max(current_quality - 5, self.config.min_quality)

        # Return best attempt
        buffer.seek(0)
        return Image.open(buffer)

    def _log_completion_stats(self) -> None:
        """Log completion statistics."""
        self.logger.info(
            f"Extraction completed in {self._stats.processing_time:.2f}s: "
            f"{self._stats.images_extracted} images extracted, "
            f"{self._stats.pages_rendered} pages rendered, "
            f"{self._stats.duplicates_skipped} duplicates skipped, "
            f"{self._stats.errors} errors"
        )

    def get_stats(self) -> ExtractionStats:
        """Get current extraction statistics."""
        return self._stats

    def reset_duplicate_tracking(self) -> None:
        """Reset duplicate image tracking."""
        self._seen_hashes.clear()


# Example usage
def main():
    """Example usage of the PDFImageExtractor."""
    # Configure extraction settings
    config = ExtractionConfig(
        target_size_kb=500, image_format=ImageFormat.WEBP, image_quality=90, dpi=200
    )

    # Create extractor
    extractor = PDFImageExtractor(config=config)

    try:
        # Extract images
        stats = extractor.extract_images(
            pdf_path="example.pdf", output_folder="extracted_images"
        )

        print(f"Extraction completed successfully!")
        print(f"Images extracted: {stats.images_extracted}")
        print(f"Processing time: {stats.processing_time:.2f}s")

    except PDFImageExtractorError as e:
        print(f"Extraction failed: {e}")


if __name__ == "__main__":
    main()
