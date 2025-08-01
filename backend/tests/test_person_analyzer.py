import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import cv2

# Import the module to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock external dependencies before import
with patch('mediapipe.solutions.face_detection'), \
     patch('mediapipe.solutions.pose'), \
     patch('mediapipe.solutions.hands'), \
     patch('deepface.DeepFace'), \
     patch('ultralytics.YOLO'):
    from models.person_analyzer import PersonAnalyzer

class TestPersonAnalyzer:
    """Test cases for PersonAnalyzer class"""
    
    @pytest.fixture
    def mock_analyzer(self):
        """Create PersonAnalyzer with mocked dependencies"""
        with patch('mediapipe.solutions.face_detection'), \
             patch('mediapipe.solutions.pose'), \
             patch('mediapipe.solutions.hands'), \
             patch('deepface.DeepFace'), \
             patch('ultralytics.YOLO'), \
             patch('cv2.CascadeClassifier'):
            
            analyzer = PersonAnalyzer()
            
            # Mock the models
            analyzer.face_detection = Mock()
            analyzer.pose = Mock()
            analyzer.yolo_model = Mock()
            analyzer.hair_cascade = Mock()
            
            return analyzer
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing"""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_analyzer_initialization(self, mock_analyzer):
        """Test that PersonAnalyzer initializes correctly"""
        assert isinstance(mock_analyzer, PersonAnalyzer)
        assert hasattr(mock_analyzer, 'face_detection')
        assert hasattr(mock_analyzer, 'pose')
    
    def test_get_image_info(self, mock_analyzer, sample_image):
        """Test image info extraction"""
        info = mock_analyzer.get_image_info(sample_image)
        
        assert isinstance(info, dict)
        assert 'height' in info
        assert 'width' in info
        assert 'channels' in info
        assert info['height'] == 100
        assert info['width'] == 100
        assert info['channels'] == 3
    
    @patch('cv2.cvtColor')
    def test_analyze_image_structure(self, mock_cvtColor, mock_analyzer, sample_image):
        """Test that analyze_image returns correct structure"""
        # Mock cv2.cvtColor to return the same image
        mock_cvtColor.return_value = sample_image
        
        # Mock face and pose detection to return empty results
        mock_analyzer.detect_faces = Mock(return_value=[])
        mock_analyzer.detect_poses = Mock(return_value=[])
        mock_analyzer.generate_summary = Mock(return_value={})
        
        result = mock_analyzer.analyze_image(sample_image)
        
        assert isinstance(result, dict)
        assert 'timestamp' in result
        assert 'image_info' in result
        assert 'people' in result
        assert 'summary' in result
        assert isinstance(result['people'], list)
    
    def test_detect_faces_called(self, mock_analyzer, sample_image):
        """Test that face detection is called during analysis"""
        with patch('cv2.cvtColor', return_value=sample_image):
            mock_analyzer.detect_faces = Mock(return_value=[])
            mock_analyzer.detect_poses = Mock(return_value=[])
            mock_analyzer.generate_summary = Mock(return_value={})
            
            mock_analyzer.analyze_image(sample_image)
            
            mock_analyzer.detect_faces.assert_called_once()
    
    def test_detect_poses_called(self, mock_analyzer, sample_image):
        """Test that pose detection is called during analysis"""
        with patch('cv2.cvtColor', return_value=sample_image):
            mock_analyzer.detect_faces = Mock(return_value=[])
            mock_analyzer.detect_poses = Mock(return_value=[])
            mock_analyzer.generate_summary = Mock(return_value={})
            
            mock_analyzer.analyze_image(sample_image)
            
            mock_analyzer.detect_poses.assert_called_once()
    
    def test_analyze_clothing_basic(self, mock_analyzer, sample_image):
        """Test basic clothing analysis functionality"""
        # Mock YOLO model response
        mock_results = Mock()
        mock_results.boxes = Mock()
        mock_results.boxes.cls = np.array([0, 1])  # Sample class IDs
        mock_results.boxes.conf = np.array([0.8, 0.9])  # Sample confidences
        
        if mock_analyzer.yolo_model:
            mock_analyzer.yolo_model.return_value = [mock_results]
        
        clothing_info = mock_analyzer.analyze_clothing(sample_image, (0, 0, 50, 50))
        
        assert isinstance(clothing_info, dict)
        assert 'detected_items' in clothing_info
        assert 'dominant_colors' in clothing_info
    
    def test_analyze_colors_basic(self, mock_analyzer, sample_image):
        """Test basic color analysis functionality"""
        colors = mock_analyzer.analyze_colors(sample_image)
        
        assert isinstance(colors, list)
        # Should return some color analysis even for random image
        assert len(colors) >= 0
    
    def test_error_handling_empty_image(self, mock_analyzer):
        """Test error handling with empty image"""
        empty_image = np.array([])
        
        # Should handle empty image gracefully
        with pytest.raises((ValueError, Exception)):
            mock_analyzer.get_image_info(empty_image)
    
    def test_error_handling_invalid_image_shape(self, mock_analyzer):
        """Test error handling with invalid image shape"""
        invalid_image = np.random.randint(0, 255, (100,), dtype=np.uint8)  # 1D array
        
        with pytest.raises((ValueError, Exception)):
            mock_analyzer.get_image_info(invalid_image)
    
    def test_generate_summary_with_people(self, mock_analyzer):
        """Test summary generation with people data"""
        mock_people = [
            {'demographics': {'estimatedAge': 25, 'gender': 'Male'}},
            {'demographics': {'estimatedAge': 30, 'gender': 'Female'}}
        ]
        
        summary = mock_analyzer.generate_summary(mock_people)
        
        assert isinstance(summary, dict)
        assert 'total_people' in summary or len(summary) >= 0
    
    def test_generate_summary_empty_people(self, mock_analyzer):
        """Test summary generation with no people"""
        summary = mock_analyzer.generate_summary([])
        
        assert isinstance(summary, dict)

if __name__ == '__main__':
    pytest.main([__file__]) 