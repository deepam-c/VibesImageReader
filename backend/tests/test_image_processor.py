import pytest
import numpy as np
from PIL import Image
import cv2
import base64
import io

# Import the module to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_processor import ImageProcessor

class TestImageProcessor:
    """Test cases for ImageProcessor utility class"""
    
    @pytest.fixture
    def processor(self):
        """Create ImageProcessor instance for testing"""
        return ImageProcessor()
    
    @pytest.fixture
    def sample_pil_image(self):
        """Create a sample PIL image for testing"""
        return Image.new('RGB', (100, 100), color=(255, 0, 0))  # Red image
    
    @pytest.fixture
    def sample_opencv_image(self):
        """Create a sample OpenCV image for testing"""
        # Create a blue image in BGR format (OpenCV default)
        return np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)  # Blue in BGR
    
    def test_processor_initialization(self, processor):
        """Test that ImageProcessor initializes correctly"""
        assert isinstance(processor, ImageProcessor)
    
    def test_pil_to_opencv_conversion(self, processor, sample_pil_image):
        """Test PIL to OpenCV conversion"""
        opencv_img = processor.pil_to_opencv(sample_pil_image)
        
        # Check output type
        assert isinstance(opencv_img, np.ndarray)
        
        # Check dimensions (should be same as input)
        assert opencv_img.shape == (100, 100, 3)
        
        # Check that it's BGR format (red in PIL should be blue in OpenCV BGR)
        # Red pixel in PIL RGB (255,0,0) becomes (0,0,255) in OpenCV BGR
        assert opencv_img[0, 0, 2] == 255  # Red channel (BGR)
        assert opencv_img[0, 0, 0] == 0    # Blue channel (BGR)
    
    def test_opencv_to_pil_conversion(self, processor, sample_opencv_image):
        """Test OpenCV to PIL conversion"""
        pil_img = processor.opencv_to_pil(sample_opencv_image)
        
        # Check output type
        assert isinstance(pil_img, Image.Image)
        
        # Check dimensions
        assert pil_img.size == (100, 100)
        
        # Check that it's RGB format
        assert pil_img.mode == 'RGB'
    
    def test_roundtrip_conversion(self, processor, sample_pil_image):
        """Test PIL -> OpenCV -> PIL conversion maintains image integrity"""
        # Convert PIL to OpenCV and back
        opencv_img = processor.pil_to_opencv(sample_pil_image)
        final_pil_img = processor.opencv_to_pil(opencv_img)
        
        # Check dimensions are preserved
        assert final_pil_img.size == sample_pil_image.size
        assert final_pil_img.mode == sample_pil_image.mode
        
        # Convert to arrays for pixel comparison
        original_array = np.array(sample_pil_image)
        final_array = np.array(final_pil_img)
        
        # Arrays should be identical (or very close due to conversion)
        np.testing.assert_array_equal(original_array, final_array)
    
    def test_pil_to_opencv_with_grayscale(self, processor):
        """Test PIL to OpenCV conversion with grayscale image"""
        gray_img = Image.new('L', (50, 50), color=128)  # Grayscale
        
        opencv_img = processor.pil_to_opencv(gray_img)
        
        # Should convert to RGB then BGR, so 3 channels
        assert opencv_img.shape == (50, 50, 3)
        assert isinstance(opencv_img, np.ndarray)
    
    def test_pil_to_opencv_with_rgba(self, processor):
        """Test PIL to OpenCV conversion with RGBA image"""
        rgba_img = Image.new('RGBA', (50, 50), color=(255, 0, 0, 128))  # Semi-transparent red
        
        opencv_img = processor.pil_to_opencv(rgba_img)
        
        # Should convert to RGB then BGR, so 3 channels (alpha removed)
        assert opencv_img.shape == (50, 50, 3)
        assert isinstance(opencv_img, np.ndarray)
    
    def test_error_handling_invalid_input(self, processor):
        """Test error handling with invalid input"""
        with pytest.raises(Exception):
            processor.pil_to_opencv(None)
        
        with pytest.raises(Exception):
            processor.opencv_to_pil(None)
    
    def test_empty_image_handling(self, processor):
        """Test handling of empty/zero-size images"""
        # Create minimal valid image
        tiny_img = Image.new('RGB', (1, 1), color='red')
        opencv_img = processor.pil_to_opencv(tiny_img)
        
        assert opencv_img.shape == (1, 1, 3)
        assert isinstance(opencv_img, np.ndarray)

if __name__ == '__main__':
    pytest.main([__file__]) 