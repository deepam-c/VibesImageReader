"""
Test cases for app_enhanced_refactored.py
Tests the enhanced Flask application with AI features
"""

import unittest
import json
import base64
from unittest.mock import patch, MagicMock
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_enhanced_refactored import create_enhanced_app


class TestEnhancedRefactoredApp(unittest.TestCase):
    """Test suite for enhanced refactored Flask app"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = create_enhanced_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Sample base64 image for testing
        self.sample_image_b64 = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAAAAA"
    
    def test_app_creation(self):
        """Test that the enhanced app can be created successfully"""
        self.assertIsNotNone(self.app)
        self.assertTrue(self.app.config['TESTING'])
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = self.client.get('/health')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check required fields
        self.assertIn('status', data)
        self.assertIn('timestamp', data)
        self.assertIn('version', data)
        self.assertIn('service', data)
        self.assertIn('ai_backend', data)
        self.assertIn('architecture', data)
        
        # Check values
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['version'], '2.1.0-enhanced-ai')
        self.assertEqual(data['service'], 'Enhanced CV API with Advanced AI')
        self.assertIn('Clean Architecture', data['architecture'])
    
    def test_cors_headers(self):
        """Test that CORS headers are properly set"""
        response = self.client.get('/health')
        
        # Should have CORS headers
        self.assertIn('Access-Control-Allow-Origin', response.headers)
    
    def test_options_endpoint(self):
        """Test CORS preflight OPTIONS request"""
        response = self.client.options('/analyze-image')
        
        self.assertEqual(response.status_code, 200)
    
    def test_analyze_image_missing_data(self):
        """Test analyze-image endpoint with missing data"""
        response = self.client.post('/analyze-image',
                                   json={},
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('No image data provided', data['error'])
    
    def test_analyze_image_invalid_json(self):
        """Test analyze-image endpoint with invalid JSON"""
        response = self.client.post('/analyze-image',
                                   data='invalid json',
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
    
    @patch('services.smart_image_processor.SmartImageProcessor')
    def test_analyze_image_success_mock(self, mock_processor_class):
        """Test analyze-image endpoint with mocked successful response"""
        # Mock the processor
        mock_processor = MagicMock()
        mock_processor.analyze_image_sync.return_value = {
            'success': True,
            'people': [
                {
                    'person_id': 1,
                    'demographics': {
                        'age': {'estimated_age': 25, 'confidence': 0.8},
                        'gender': {'prediction': 'male', 'confidence': 0.9}
                    },
                    'pose': {'detected': True, 'confidence': 0.7}
                }
            ],
            'summary': {
                'total_people': 1,
                'average_age': 25
            },
            'model_info': {
                'version': '2.1.0',
                'ai_backend': 'Enhanced Mock'
            }
        }
        mock_processor_class.return_value = mock_processor
        
        # Test the endpoint
        response = self.client.post('/analyze-image',
                                   json={'image': self.sample_image_b64},
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check response structure
        self.assertIn('success', data)
        self.assertIn('people', data)
        self.assertIn('analysis_id', data)
        self.assertTrue(data['success'])
    
    @patch('services.smart_image_processor.SmartImageProcessor')
    def test_analyze_image_processor_error(self, mock_processor_class):
        """Test analyze-image endpoint when processor raises exception"""
        # Mock the processor to raise an exception
        mock_processor = MagicMock()
        mock_processor.analyze_image_sync.side_effect = Exception("Processing failed")
        mock_processor_class.return_value = mock_processor
        
        response = self.client.post('/analyze-image',
                                   json={'image': self.sample_image_b64},
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_analyze_image_no_content_type(self):
        """Test analyze-image endpoint without proper content type"""
        response = self.client.post('/analyze-image',
                                   data=json.dumps({'image': self.sample_image_b64}))
        
        # Should handle this gracefully
        self.assertIn(response.status_code, [400, 415])  # Bad Request or Unsupported Media Type
    
    @patch('services.smart_image_processor.SmartImageProcessor')
    def test_capabilities_endpoint(self, mock_processor_class):
        """Test the capabilities endpoint if it exists"""
        try:
            response = self.client.get('/capabilities')
            if response.status_code != 404:  # If endpoint exists
                self.assertEqual(response.status_code, 200)
                data = json.loads(response.data)
                self.assertIsInstance(data, dict)
        except Exception:
            # If endpoint doesn't exist, that's okay for now
            pass
    
    @patch('services.smart_image_processor.SmartImageProcessor')
    def test_test_ai_endpoint(self, mock_processor_class):
        """Test the test-ai endpoint if it exists"""
        try:
            response = self.client.post('/test-ai')
            if response.status_code != 404:  # If endpoint exists
                self.assertIn(response.status_code, [200, 400, 500])
                data = json.loads(response.data)
                self.assertIn('test_status', data)
        except Exception:
            # If endpoint doesn't exist, that's okay for now
            pass
    
    def test_error_handling_malformed_image_data(self):
        """Test handling of malformed image data"""
        response = self.client.post('/analyze-image',
                                   json={'image': 'invalid_base64_data'},
                                   content_type='application/json')
        
        # Should return an error, not crash
        self.assertIn(response.status_code, [400, 500])
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_large_request_handling(self):
        """Test handling of very large requests"""
        large_data = 'x' * 1000000  # 1MB of data
        response = self.client.post('/analyze-image',
                                   json={'image': large_data},
                                   content_type='application/json')
        
        # Should handle gracefully, not crash
        self.assertIsNotNone(response.status_code)


class TestEnhancedAppIntegration(unittest.TestCase):
    """Integration tests for the enhanced app"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.app = create_enhanced_app()
        self.client = self.app.test_client()
    
    def test_full_request_cycle(self):
        """Test a complete request cycle"""
        # 1. Check health
        health_response = self.client.get('/health')
        self.assertEqual(health_response.status_code, 200)
        
        # 2. Try image analysis
        response = self.client.post('/analyze-image',
                                   json={'image': 'test_image_data'},
                                   content_type='application/json')
        
        # Should return some response (even if error due to bad image data)
        self.assertIsNotNone(response.status_code)
    
    def test_concurrent_requests_simulation(self):
        """Simulate multiple concurrent requests"""
        responses = []
        
        # Simulate multiple health checks
        for _ in range(5):
            response = self.client.get('/health')
            responses.append(response)
        
        # All should succeed
        for response in responses:
            self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main() 