import os
import io
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
import hashlib
import logging
from typing import Optional, Set, Union, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import tempfile
import concurrent.futures
from contextlib import contextmanager
import shutil
from src.assets.crop_image_remove_white import (
    contour_crop_image,
    is_mostly_white_with_text,
)
import psutil


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
    parallel_processing: bool = True
    max_workers: int = 4
    crop_images: bool = True
    metadata_extraction: bool = True
    save_original_images: bool = False
    original_images_folder: str = "originals"
    timeout_per_page: int = 60  # seconds
    memory_limit_mb: int = 1000  # MB

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
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.timeout_per_page <= 0:
            raise ValueError("timeout_per_page must be positive")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")


@dataclass
class ImageMetadata:
    """Metadata for an extracted image."""

    page_num: int
    image_index: int
    width: int
    height: int
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    format: str
    quality: int
    extraction_method: str
    is_cropped: bool
    md5_hash: str
    extraction_time: float


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
    memory_usage_mb: float = 0.0
    total_original_size_mb: float = 0.0
    total_compressed_size_mb: float = 0.0
    compression_ratio: float = 0.0
    image_metadata: List[ImageMetadata] = field(default_factory=list)
    error_details: Dict[int, List[str]] = field(default_factory=dict)

    def add_error(self, page_num: int, error_msg: str) -> None:
        """Add error details for a specific page."""
        if page_num not in self.error_details:
            self.error_details[page_num] = []
        self.error_details[page_num].append(error_msg)

    def export_to_json(self, filepath: Union[str, Path]) -> None:
        """Export statistics to a JSON file."""
        import json

        export_data = {
            "total_pages": self.total_pages,
            "images_extracted": self.images_extracted,
            "pages_rendered": self.pages_rendered,
            "duplicates_skipped": self.duplicates_skipped,
            "errors": self.errors,
            "processing_time_seconds": self.processing_time,
            "output_folder": str(self.output_folder) if self.output_folder else None,
            "memory_usage_mb": self.memory_usage_mb,
            "total_original_size_mb": self.total_original_size_mb,
            "total_compressed_size_mb": self.total_compressed_size_mb,
            "compression_ratio": self.compression_ratio,
            "error_details": self.error_details,
            "image_metadata": [vars(meta) for meta in self.image_metadata],
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)


class PDFImageExtractorError(Exception):
    """Base exception for PDF image extraction errors."""

    pass


class PDFFileError(PDFImageExtractorError):
    """Exception for PDF file-related errors."""

    pass


class ImageExtractionError(PDFImageExtractorError):
    """Exception for image extraction errors."""

    pass


class ImageCompressionError(PDFImageExtractorError):
    """Exception for image compression errors."""

    pass


class TimeoutError(PDFImageExtractorError):
    """Exception for timeout errors."""

    pass


class MemoryLimitError(PDFImageExtractorError):
    """Exception for memory limit exceeded errors."""

    pass


class PDFImageExtractor:
    """
    Production-level PDF image extractor with comprehensive error handling,
    logging, parallel processing, and configurable compression settings.
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
        self._temp_dir = None

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

    @contextmanager
    def _memory_usage_monitor(self):
        """Context manager to monitor memory usage."""
        try:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
            yield
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            self._stats.memory_usage_mb = final_memory - initial_memory

            if final_memory > self.config.memory_limit_mb:
                self.logger.warning(
                    f"Memory usage exceeded limit: {final_memory:.2f}MB > {self.config.memory_limit_mb}MB"
                )
        except ImportError:
            self.logger.warning("psutil not installed, memory monitoring disabled")
            yield
            self._stats.memory_usage_mb = 0

    @contextmanager
    def _create_temp_directory(self):
        """Create and manage a temporary directory."""
        self._temp_dir = tempfile.mkdtemp(prefix="pdf_image_extractor_")
        try:
            yield self._temp_dir
        finally:
            if self._temp_dir and os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir, ignore_errors=True)
                self._temp_dir = None

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

        with self._memory_usage_monitor(), self._create_temp_directory():
            try:
                self._validate_inputs(pdf_path)
                self._create_output_folder(output_folder)

                # Create originals folder if needed
                if self.config.save_original_images:
                    orig_folder = output_folder / self.config.original_images_folder
                    orig_folder.mkdir(exist_ok=True)

                with fitz.open(str(pdf_path)) as pdf_document:
                    self._stats.total_pages = len(pdf_document)
                    self.logger.info(
                        f"Processing PDF with {self._stats.total_pages} pages"
                    )

                    if self.config.parallel_processing and self._stats.total_pages > 1:
                        self._process_pages_parallel(pdf_document, output_folder)
                    else:
                        self._process_pages_sequential(pdf_document, output_folder)

            except Exception as e:
                self.logger.error(f"Failed to extract images: {e}", exc_info=True)
                raise PDFImageExtractorError(f"Failed to extract images: {e}") from e
            finally:
                self._stats.processing_time = time.time() - start_time

                # Calculate compression statistics
                if self._stats.total_original_size_mb > 0:
                    self._stats.compression_ratio = (
                        self._stats.total_original_size_mb
                        / self._stats.total_compressed_size_mb
                        if self._stats.total_compressed_size_mb > 0
                        else 0
                    )

                self._log_completion_stats()

                # Export metadata if enabled
                if self.config.metadata_extraction:
                    metadata_path = output_folder / "extraction_metadata.json"
                    try:
                        self._stats.export_to_json(metadata_path)
                    except Exception as e:
                        self.logger.error(f"Failed to export metadata: {e}")

        return self._stats

    def _process_pages_sequential(
        self, pdf_document: fitz.Document, output_folder: Path
    ) -> None:
        """Process PDF pages sequentially."""
        for page_num in range(len(pdf_document)):
            try:
                with self._timeout_context(self.config.timeout_per_page):
                    self._process_page(pdf_document, page_num, output_folder)
            except TimeoutError:
                self._stats.errors += 1
                error_msg = f"Processing timeout on page {page_num + 1}"
                self._stats.add_error(page_num + 1, error_msg)
                self.logger.error(error_msg)
            except MemoryLimitError:
                self._stats.errors += 1
                error_msg = f"Memory limit exceeded on page {page_num + 1}"
                self._stats.add_error(page_num + 1, error_msg)
                self.logger.error(error_msg)
            except Exception as e:
                self._stats.errors += 1
                error_msg = f"Error processing page {page_num + 1}: {str(e)}"
                self._stats.add_error(page_num + 1, error_msg)
                self.logger.error(error_msg, exc_info=True)

    def _process_pages_parallel(
        self, pdf_document: fitz.Document, output_folder: Path
    ) -> None:
        """Process PDF pages in parallel."""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers
        ) as executor:
            futures = {
                executor.submit(
                    self._process_page_with_timeout,
                    pdf_document,
                    page_num,
                    output_folder,
                ): page_num
                for page_num in range(len(pdf_document))
            }

            for future in concurrent.futures.as_completed(futures):
                page_num = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self._stats.errors += 1
                    error_msg = f"Error processing page {page_num + 1}: {str(e)}"
                    self._stats.add_error(page_num + 1, error_msg)
                    self.logger.error(error_msg, exc_info=True)

    def _process_page_with_timeout(
        self, pdf_document: fitz.Document, page_num: int, output_folder: Path
    ) -> None:
        """Process a page with timeout."""
        try:
            with self._timeout_context(self.config.timeout_per_page):
                self._process_page(pdf_document, page_num, output_folder)
        except TimeoutError:
            raise TimeoutError(f"Processing timeout on page {page_num + 1}")
        except MemoryLimitError:
            raise MemoryLimitError(f"Memory limit exceeded on page {page_num + 1}")
        except Exception as e:
            raise ImageExtractionError(f"Failed to process page {page_num + 1}: {e}")

    @contextmanager
    def _timeout_context(self, timeout_seconds: int):
        """No-op context manager for page processing timeout (not enforced on Windows)."""
        # On Windows, signal.SIGALRM is not available, so we skip timeout enforcement.
        yield

    def _validate_inputs(self, pdf_path: Path) -> None:
        """Validate input parameters."""
        if not pdf_path.exists():
            raise PDFFileError(f"PDF file not found: {pdf_path}")
        if not pdf_path.is_file():
            raise PDFFileError(f"Path is not a file: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise PDFFileError(f"File is not a PDF: {pdf_path}")

        # Check if file is readable
        try:
            with open(pdf_path, "rb") as f:
                f.read(100)  # Try to read the first 100 bytes
        except Exception as e:
            raise PDFFileError(f"Cannot read PDF file: {e}")

        # Check if file is not empty
        if pdf_path.stat().st_size == 0:
            raise PDFFileError(f"PDF file is empty: {pdf_path}")

    def _create_output_folder(self, output_folder: Path) -> None:
        """Create output folder if it doesn't exist."""
        try:
            output_folder.mkdir(parents=True, exist_ok=True)

            # Check if folder is writable
            test_file = output_folder / ".write_test"
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                raise PDFImageExtractorError(f"Output folder is not writable: {e}")

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
                error_msg = f"Failed to extract image {img_index + 1} from page {page_num + 1}: {e}"
                self._stats.add_error(page_num + 1, error_msg)
                self.logger.error(error_msg, exc_info=True)

    def _extract_embedded_image(
        self,
        pdf_document: fitz.Document,
        img: tuple,
        page_num: int,
        img_index: int,
        output_folder: Path,
    ) -> None:
        """Extract a single embedded image."""
        start_time = time.time()

        xref = img[0]
        base_image = pdf_document.extract_image(xref)
        image_bytes = base_image["image"]
        original_size_bytes = len(image_bytes)

        # Update total original size
        self._stats.total_original_size_mb += original_size_bytes / (1024 * 1024)

        # Calculate MD5 hash for duplicate detection
        image_hash = hashlib.md5(image_bytes).hexdigest()

        if self.config.skip_duplicates:
            if image_hash in self._seen_hashes:
                self._stats.duplicates_skipped += 1
                self.logger.debug(f"Skipped duplicate image on page {page_num + 1}")
                return
            self._seen_hashes.add(image_hash)

        # Save original image if configured
        if self.config.save_original_images:
            orig_folder = output_folder / self.config.original_images_folder
            orig_filename = (
                f"page_{page_num + 1}_img_{img_index + 1}_original.{base_image['ext']}"
            )
            with open(orig_folder / orig_filename, "wb") as f:
                f.write(image_bytes)

        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size

        # Generate filename
        filename = self._generate_filename(page_num, img_index, embedded=True)

        # Process and save compressed image
        is_cropped = False
        if self.config.crop_images:
            try:
                # Use temporary file for cropping
                temp_img_path = os.path.join(self._temp_dir, f"temp_{image_hash}.png")
                image.save(temp_img_path)
                cropped_image = contour_crop_image(temp_img_path, output_path=None)
                # Skip if image is mostly white with only small text
                if is_mostly_white_with_text(cropped_image):
                    self.logger.info(
                        f"Skipped mostly white/text image on page {page_num + 1}, img {img_index + 1}"
                    )
                    os.unlink(temp_img_path)
                    return
                is_cropped = True
                os.unlink(temp_img_path)
            except Exception as e:
                self.logger.warning(f"Failed to crop image: {e}, using original image")
                cropped_image = image.copy()
        else:
            cropped_image = image.copy()

        # Save compressed image
        compressed_image, quality, compressed_size_bytes = self._save_compressed_image(
            cropped_image, output_folder, filename
        )

        # Update total compressed size
        self._stats.total_compressed_size_mb += compressed_size_bytes / (1024 * 1024)

        # Create metadata
        extraction_time = time.time() - start_time
        metadata = ImageMetadata(
            page_num=page_num + 1,
            image_index=img_index + 1,
            width=width,
            height=height,
            original_size_bytes=original_size_bytes,
            compressed_size_bytes=compressed_size_bytes,
            compression_ratio=(
                original_size_bytes / compressed_size_bytes
                if compressed_size_bytes > 0
                else 0
            ),
            format=self.config.image_format.value,
            quality=quality,
            extraction_method="embedded",
            is_cropped=is_cropped,
            md5_hash=image_hash,
            extraction_time=extraction_time,
        )

        if self.config.metadata_extraction:
            self._stats.image_metadata.append(metadata)

        self._stats.images_extracted += 1

    def _render_page_as_image(
        self, page: fitz.Page, page_num: int, output_folder: Path
    ) -> None:
        """Render entire page as image (fallback method)."""
        start_time = time.time()
        self.logger.info(f"Rendering page {page_num + 1} as image")

        zoom = self.config.dpi / 72  # PDF default is 72 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Get original size
        original_size_bytes = len(pix.samples)
        self._stats.total_original_size_mb += original_size_bytes / (1024 * 1024)

        # Convert to PIL Image
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        width, height = image.size

        # Generate MD5 hash
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_hash = hashlib.md5(image_bytes.getvalue()).hexdigest()

        # Save original image if configured
        if self.config.save_original_images:
            orig_folder = output_folder / self.config.original_images_folder
            orig_filename = f"page_{page_num + 1}_rendered_original.png"
            image.save(orig_folder / orig_filename)

        # Generate filename
        filename = self._generate_filename(page_num, 0, embedded=False)

        # Save compressed image
        compressed_image, quality, compressed_size_bytes = self._save_compressed_image(
            image, output_folder, filename
        )

        # Update total compressed size
        self._stats.total_compressed_size_mb += compressed_size_bytes / (1024 * 1024)

        # Create metadata
        extraction_time = time.time() - start_time
        metadata = ImageMetadata(
            page_num=page_num + 1,
            image_index=0,
            width=width,
            height=height,
            original_size_bytes=original_size_bytes,
            compressed_size_bytes=compressed_size_bytes,
            compression_ratio=(
                original_size_bytes / compressed_size_bytes
                if compressed_size_bytes > 0
                else 0
            ),
            format=self.config.image_format.value,
            quality=quality,
            extraction_method="rendered",
            is_cropped=False,
            md5_hash=image_hash,
            extraction_time=extraction_time,
        )

        if self.config.metadata_extraction:
            self._stats.image_metadata.append(metadata)

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
    ) -> Tuple[Image.Image, int, int]:
        """Save image with compression to target size and return the compressed image, quality, and size."""
        try:
            # Compress image to target size
            compressed_image, quality = self._compress_image_to_target_size(image)
            file_path = output_folder / filename

            # Save with format-specific options
            save_kwargs = {
                "format": self.config.image_format.value,
                "quality": quality,
                "optimize": True,
            }

            if self.config.image_format == ImageFormat.WEBP:
                save_kwargs["method"] = 6  # Best compression
            elif self.config.image_format == ImageFormat.JPEG:
                save_kwargs["progressive"] = True

            compressed_image.save(str(file_path), **save_kwargs)

            # Get file size
            compressed_size_bytes = file_path.stat().st_size
            file_size_kb = compressed_size_bytes / 1024
            self.logger.info(
                f"Saved {filename} ({file_size_kb:.1f} KB, quality={quality})"
            )

            return compressed_image, quality, compressed_size_bytes

        except Exception as e:
            raise ImageCompressionError(f"Failed to save image {filename}: {e}") from e

    def _compress_image_to_target_size(
        self, image: Image.Image
    ) -> Tuple[Image.Image, int]:
        """
        Compress image to approach target size through iterative resizing and quality adjustment.
        Returns the compressed image and the final quality setting.
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
                return Image.open(buffer), current_quality

            # Resize image
            width, height = current_image.size
            new_width = int(width * self.config.resize_factor)
            new_height = int(height * self.config.resize_factor)

            # Ensure dimensions don't get too small
            if new_width < 100 or new_height < 100:
                # Just reduce quality instead of resizing further
                current_quality = max(current_quality - 10, self.config.min_quality)
            else:
                current_image = current_image.resize(
                    (new_width, new_height), Image.LANCZOS
                )
                # Reduce quality gradually
                current_quality = max(current_quality - 5, self.config.min_quality)

        # Return best attempt
        buffer.seek(0)
        return Image.open(buffer), current_quality

    def _log_completion_stats(self) -> None:
        """Log completion statistics."""
        self.logger.info(
            f"Extraction completed in {self._stats.processing_time:.2f}s: "
            f"{self._stats.images_extracted} images extracted, "
            f"{self._stats.pages_rendered} pages rendered, "
            f"{self._stats.duplicates_skipped} duplicates skipped, "
            f"{self._stats.errors} errors"
        )

        self.logger.info(
            f"Memory usage: {self._stats.memory_usage_mb:.2f}MB, "
            f"Original size: {self._stats.total_original_size_mb:.2f}MB, "
            f"Compressed size: {self._stats.total_compressed_size_mb:.2f}MB, "
            f"Compression ratio: {self._stats.compression_ratio:.2f}x"
        )

    def get_stats(self) -> ExtractionStats:
        """Get current extraction statistics."""
        return self._stats

    def reset_duplicate_tracking(self) -> None:
        """Reset duplicate image tracking."""
        self._seen_hashes.clear()

    def cleanup(self) -> None:
        """Clean up temporary resources."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None


# Example usage
def main():
    """Example usage of the PDFImageExtractor."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract images from PDF files")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--output", "-o", default="extracted_images", help="Output folder"
    )
    parser.add_argument(
        "--format", choices=["webp", "jpeg", "png"], default="webp", help="Image format"
    )
    parser.add_argument("--quality", type=int, default=85, help="Image quality (1-100)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for rendering")
    parser.add_argument(
        "--target-size", type=int, default=500, help="Target size in KB"
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Enable parallel processing"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker threads"
    )
    parser.add_argument("--no-crop", action="store_true", help="Disable image cropping")
    parser.add_argument(
        "--save-originals", action="store_true", help="Save original images"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configure extraction settings
    format_map = {
        "webp": ImageFormat.WEBP,
        "jpeg": ImageFormat.JPEG,
        "png": ImageFormat.PNG,
    }

    config = ExtractionConfig(
        target_size_kb=args.target_size,
        image_format=format_map[args.format],
        image_quality=args.quality,
        dpi=args.dpi,
        parallel_processing=args.parallel,
        max_workers=args.workers,
        crop_images=not args.no_crop,
        save_original_images=args.save_originals,
    )

    # Create extractor
    extractor = PDFImageExtractor(config=config)

    try:
        # Extract images
        stats = extractor.extract_images(
            pdf_path=args.pdf_path, output_folder=args.output
        )

        print(f"Extraction completed successfully!")
        print(f"Images extracted: {stats.images_extracted}")
        print(f"Pages rendered: {stats.pages_rendered}")
        print(f"Duplicates skipped: {stats.duplicates_skipped}")
        print(f"Errors: {stats.errors}")
        print(f"Processing time: {stats.processing_time:.2f}s")
        print(f"Compression ratio: {stats.compression_ratio:.2f}x")
        print(f"Output folder: {stats.output_folder}")

    except PDFImageExtractorError as e:
        print(f"Extraction failed: {e}")
    finally:
        extractor.cleanup()


if __name__ == "__main__":
    main()
