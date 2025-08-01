import pytest
import json
import base64
import io
from PIL import Image
from unittest.mock import patch, Mock

# Import the Flask app
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock heavy dependencies before importing
with patch('mediapipe.solutions.face_detection'), \
     patch('mediapipe.solutions.pose'), \
     patch('mediapipe.solutions.hands'), \
     patch('deepface.DeepFace'), \
     patch('ultralytics.YOLO'):
    from app import app

@pytest.fixture
def client():
    """Create test client for Flask app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_image_base64():
    """Create a sample base64 encoded image for testing"""
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    base64_string = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_string}"

class TestHealthEndpoint:
    """Test cases for the health check endpoint"""
    
    def test_health_check_status_code(self, client):
        """Test that health endpoint returns 200"""
        response = client.get('/health')
        assert response.status_code == 200
    
    def test_health_check_content(self, client):
        """Test that health endpoint returns correct content"""
        response = client.get('/health')
        data = json.loads(response.data)
        
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert data['service'] == 'Human Analysis API'
    
    def test_health_check_content_type(self, client):
        """Test that health endpoint returns JSON"""
        response = client.get('/health')
        assert response.content_type == 'application/json'

class TestBasicEndpoints:
    """Test cases for basic endpoint functionality"""
    
    def test_analyze_image_no_data(self, client):
        """Test analyze endpoint with no data"""
        response = client.post('/analyze-image')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_analyze_image_no_image_key(self, client):
        """Test analyze endpoint with missing image key"""
        response = client.post('/analyze-image', 
                             json={'other_key': 'value'})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] == 'No image data provided'
    
    def test_analyze_image_invalid_base64(self, client):
        """Test analyze endpoint with invalid base64 data"""
        response = client.post('/analyze-image', 
                             json={'image': 'invalid_base64_data'})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] == 'Invalid image data'

if __name__ == '__main__':
    pytest.main([__file__]) 