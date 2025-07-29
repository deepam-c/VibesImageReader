import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Utility class for image processing and format conversions
    """
    
    def __init__(self):
        """Initialize image processor"""
        pass
    
    def pil_to_opencv(self, pil_image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to OpenCV format (BGR)
        """
        try:
            # Convert PIL image to RGB if not already
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            np_array = np.array(pil_image)
            
            # Convert RGB to BGR for OpenCV
            cv_image = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
            
            return cv_image
            
        except Exception as e:
            logger.error(f"Error converting PIL to OpenCV: {e}")
            raise
    
    def opencv_to_pil(self, cv_image: np.ndarray) -> Image.Image:
        """
        Convert OpenCV image (BGR) to PIL Image
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            return pil_image
            
        except Exception as e:
            logger.error(f"Error converting OpenCV to PIL: {e}")
            raise
    
    def base64_to_opencv(self, base64_string: str) -> np.ndarray:
        """
        Convert base64 encoded image to OpenCV format
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(base64_string)
            
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to OpenCV
            cv_image = self.pil_to_opencv(pil_image)
            
            return cv_image
            
        except Exception as e:
            logger.error(f"Error converting base64 to OpenCV: {e}")
            raise
    
    def opencv_to_base64(self, cv_image: np.ndarray, format: str = 'JPEG') -> str:
        """
        Convert OpenCV image to base64 encoded string
        """
        try:
            # Convert to PIL Image
            pil_image = self.opencv_to_pil(cv_image)
            
            # Convert to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format=format)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f"data:image/{format.lower()};base64,{img_str}"
            
        except Exception as e:
            logger.error(f"Error converting OpenCV to base64: {e}")
            raise
    
    def resize_image(self, image: np.ndarray, max_width: int = 1024, max_height: int = 1024) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        """
        try:
            height, width = image.shape[:2]
            
            # Calculate scaling factor
            scale_x = max_width / width
            scale_y = max_height / height
            scale = min(scale_x, scale_y, 1.0)  # Don't upscale
            
            if scale < 1.0:
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                return resized_image
            
            return image
            
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Basic image enhancement for better analysis
        """
        try:
            # Convert to LAB color space for better processing
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Apply slight sharpening
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image
    
    def crop_region(self, image: np.ndarray, bbox: dict) -> np.ndarray:
        """
        Crop specific region from image using bounding box
        """
        try:
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            
            # Ensure coordinates are within image bounds
            height, width = image.shape[:2]
            x = max(0, min(x, width))
            y = max(0, min(y, height))
            w = max(0, min(w, width - x))
            h = max(0, min(h, height - y))
            
            cropped = image[y:y+h, x:x+w]
            return cropped
            
        except Exception as e:
            logger.error(f"Error cropping image region: {e}")
            return image
    
    def validate_image(self, image: np.ndarray) -> bool:
        """
        Validate if image is valid for processing
        """
        try:
            if image is None:
                return False
            
            if len(image.shape) < 2:
                return False
            
            height, width = image.shape[:2]
            if height < 10 or width < 10:
                return False
            
            if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_image_stats(self, image: np.ndarray) -> dict:
        """
        Get basic statistics about the image
        """
        try:
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1
            
            stats = {
                'width': int(width),
                'height': int(height),
                'channels': int(channels),
                'size_mb': (image.nbytes / (1024 * 1024)),
                'aspect_ratio': round(width / height, 2),
                'mean_brightness': float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))) if channels == 3 else float(np.mean(image))
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting image stats: {e}")
            return {}
    
    def preprocess_for_analysis(self, image: np.ndarray, enhance_quality: bool = True) -> np.ndarray:
        """
        Preprocess image for optimal analysis results
        """
        try:
            # Validate input
            if not self.validate_image(image):
                raise ValueError("Invalid image for processing")
            
            # Resize if too large
            processed_image = self.resize_image(image, max_width=1024, max_height=1024)
            
            # Enhance quality if requested
            if enhance_quality:
                processed_image = self.enhance_image_quality(processed_image)
            
            return processed_image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image 