import pytest
import numpy as np
import cv2
from PIL import Image
import base64
import io
import json

# Test basic functionality without heavy dependencies
class TestBasicFunctionality:
    """Test cases for basic CV functionality that works without heavy ML models"""
    
    def test_opencv_installation(self):
        """Test that OpenCV is properly installed"""
        assert cv2.__version__ is not None
        print(f"OpenCV version: {cv2.__version__}")
    
    def test_pil_installation(self):
        """Test that PIL/Pillow is properly installed"""
        img = Image.new('RGB', (100, 100), color='red')
        assert img.size == (100, 100)
        assert img.mode == 'RGB'
    
    def test_numpy_opencv_compatibility(self):
        """Test numpy and OpenCV work together"""
        # Create numpy array
        np_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Use OpenCV operations
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        
        assert gray.shape == (100, 100)
        assert len(gray.shape) == 2  # Grayscale
    
    def test_base64_image_encoding_decoding(self):
        """Test base64 image encoding and decoding pipeline"""
        # Create test image
        img = Image.new('RGB', (50, 50), color='blue')
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
        base64_string = base64.b64encode(img_bytes).decode('utf-8')
        
        # Decode base64 back to image
        decoded_bytes = base64.b64decode(base64_string)
        decoded_img = Image.open(io.BytesIO(decoded_bytes))
        
        assert decoded_img.size == (50, 50)
        assert decoded_img.mode == 'RGB'
    
    def test_color_space_conversions(self):
        """Test various color space conversions"""
        # Create test image
        bgr_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test conversions
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        
        assert rgb_img.shape == (100, 100, 3)
        assert hsv_img.shape == (100, 100, 3)
        assert gray_img.shape == (100, 100)
    
    def test_image_resizing(self):
        """Test image resizing functionality"""
        # Create test image
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Resize image
        resized = cv2.resize(img, (100, 100))
        
        assert resized.shape == (100, 100, 3)
    
    def test_basic_image_filters(self):
        """Test basic image filtering operations"""
        # Create test image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Apply basic filters
        blurred = cv2.blur(img, (5, 5))
        gaussian = cv2.GaussianBlur(img, (5, 5), 0)
        
        assert blurred.shape == img.shape
        assert gaussian.shape == img.shape
    
    def test_edge_detection(self):
        """Test basic edge detection"""
        # Create test image
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Apply Canny edge detection
        edges = cv2.Canny(img, 50, 150)
        
        assert edges.shape == img.shape
        assert edges.dtype == np.uint8
    
    def test_json_serialization(self):
        """Test JSON serialization of analysis results"""
        test_data = {
            'timestamp': '2024-01-01T12:00:00',
            'people': [
                {
                    'id': 1,
                    'age': 25,
                    'confidence': 0.85
                }
            ],
            'summary': {
                'total_people': 1
            }
        }
        
        # Should serialize and deserialize without error
        json_str = json.dumps(test_data)
        parsed_data = json.loads(json_str)
        
        assert parsed_data['summary']['total_people'] == 1
        assert parsed_data['people'][0]['age'] == 25

class TestImageProcessingUtilities:
    """Test image processing utility functions"""
    
    def test_image_shape_validation(self):
        """Test image shape validation"""
        valid_3channel = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        valid_1channel = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        invalid_shape = np.random.randint(0, 255, (100,), dtype=np.uint8)
        
        assert len(valid_3channel.shape) == 3
        assert valid_3channel.shape[2] == 3
        
        assert len(valid_1channel.shape) == 2
        
        assert len(invalid_shape.shape) == 1  # Invalid for image processing
    
    def test_image_type_validation(self):
        """Test image data type validation"""
        uint8_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        float_img = np.random.random((100, 100, 3)).astype(np.float32)
        
        assert uint8_img.dtype == np.uint8
        assert float_img.dtype == np.float32
        
        # Convert float to uint8
        converted = (float_img * 255).astype(np.uint8)
        assert converted.dtype == np.uint8

if __name__ == '__main__':
    pytest.main([__file__]) 