import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.response_formatter import ResponseFormatter

class TestResponseFormatter:
    """Test cases for ResponseFormatter utility class"""
    
    @pytest.fixture
    def formatter(self):
        """Create ResponseFormatter instance for testing"""
        return ResponseFormatter()
    
    @pytest.fixture
    def sample_analysis_data(self):
        """Create sample analysis data for testing"""
        return {
            'timestamp': '2024-01-01T12:00:00',
            'image_info': {
                'height': 480,
                'width': 640,
                'channels': 3
            },
            'people': [
                {
                    'person_id': 1,
                    'demographics': {
                        'estimatedAge': 25,
                        'gender': 'Male',
                        'confidence': 0.85
                    },
                    'emotions': {
                        'primary': 'happy',
                        'confidence': 0.9
                    }
                }
            ],
            'summary': {
                'total_people': 1,
                'average_age': 25
            }
        }
    
    def test_formatter_initialization(self, formatter):
        """Test that ResponseFormatter initializes correctly"""
        assert isinstance(formatter, ResponseFormatter)
    
    def test_format_analysis_basic(self, formatter, sample_analysis_data):
        """Test basic analysis formatting"""
        formatted = formatter.format_analysis(sample_analysis_data)
        
        assert isinstance(formatted, dict)
        assert 'timestamp' in formatted
        assert 'people' in formatted
        assert 'summary' in formatted
    
    def test_format_analysis_preserves_structure(self, formatter, sample_analysis_data):
        """Test that formatting preserves data structure"""
        formatted = formatter.format_analysis(sample_analysis_data)
        
        # Check that key information is preserved
        assert formatted['summary']['total_people'] == 1
        assert len(formatted['people']) == 1
        assert formatted['people'][0]['demographics']['gender'] == 'Male'
    
    def test_format_analysis_empty_data(self, formatter):
        """Test formatting with empty data"""
        empty_data = {
            'timestamp': '2024-01-01T12:00:00',
            'image_info': {},
            'people': [],
            'summary': {}
        }
        
        formatted = formatter.format_analysis(empty_data)
        assert isinstance(formatted, dict)
        assert formatted['people'] == []
    
    def test_format_analysis_missing_fields(self, formatter):
        """Test formatting with missing fields"""
        incomplete_data = {
            'timestamp': '2024-01-01T12:00:00',
            'people': []
        }
        
        # Should handle missing fields gracefully
        formatted = formatter.format_analysis(incomplete_data)
        assert isinstance(formatted, dict)
        assert 'timestamp' in formatted

if __name__ == '__main__':
    pytest.main([__file__]) 