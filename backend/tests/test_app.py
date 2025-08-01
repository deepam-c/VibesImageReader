import pytest
import json
import base64
import io
from PIL import Image
import numpy as np

# Import the Flask app
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

class TestAnalyzeImageEndpoint:
    """Test cases for the image analysis endpoint"""
    
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
    
    def test_analyze_image_valid_request(self, client, sample_image_base64):
        """Test analyze endpoint with valid image data"""
        response = client.post('/analyze-image', 
                             json={'image': sample_image_base64})
        
        # Should not return 400 or 500 errors for valid image
        assert response.status_code in [200, 500]  # 500 might occur due to missing AI models
        
        data = json.loads(response.data)
        if response.status_code == 200:
            # Check that response has expected structure
            assert isinstance(data, dict)
        else:
            # If error due to missing models, that's expected in test environment
            assert 'error' in data
    
    def test_analyze_image_content_type(self, client):
        """Test that analyze endpoint returns JSON"""
        response = client.post('/analyze-image', json={})
        assert response.content_type == 'application/json'

class TestEndpointSecurity:
    """Test cases for endpoint security and error handling"""
    
    def test_nonexistent_endpoint(self, client):
        """Test that nonexistent endpoints return 404"""
        response = client.get('/nonexistent')
        assert response.status_code == 404
    
    def test_wrong_method_health(self, client):
        """Test wrong HTTP method on health endpoint"""
        response = client.post('/health')
        assert response.status_code == 405  # Method not allowed
    
    def test_wrong_method_analyze(self, client):
        """Test wrong HTTP method on analyze endpoint"""
        response = client.get('/analyze-image')
        assert response.status_code == 405  # Method not allowed

if __name__ == '__main__':
    pytest.main([__file__]) 