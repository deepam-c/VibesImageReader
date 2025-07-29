import json
from typing import Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ResponseFormatter:
    """
    Format analysis results into structured responses for the frontend
    """
    
    def __init__(self):
        """Initialize response formatter"""
        pass
    
    def format_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format comprehensive analysis results
        """
        try:
            formatted_response = {
                'success': True,
                'timestamp': analysis_results.get('timestamp', datetime.now().isoformat()),
                'processing_info': self._format_processing_info(analysis_results),
                'detection_summary': self._format_detection_summary(analysis_results),
                'people': self._format_people_analysis(analysis_results.get('people', [])),
                'scene_analysis': self._format_scene_analysis(analysis_results),
                'metadata': self._format_metadata(analysis_results)
            }
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting analysis results: {e}")
            return self._format_error_response(str(e))
    
    def _format_processing_info(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format processing information"""
        image_info = results.get('image_info', {})
        
        return {
            'image_dimensions': {
                'width': image_info.get('width', 0),
                'height': image_info.get('height', 0),
                'aspect_ratio': image_info.get('aspect_ratio', 0)
            },
            'processing_status': 'completed' if not results.get('error') else 'error',
            'analysis_version': '1.0.0'
        }
    
    def _format_detection_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format detection summary"""
        people = results.get('people', [])
        summary = results.get('summary', {})
        
        return {
            'total_people_detected': len(people),
            'faces_detected': len([p for p in people if p.get('face_analysis', {}).get('age')]),
            'poses_detected': len([p for p in people if p.get('body_analysis', {}).get('pose_detected')]),
            'gender_distribution': summary.get('gender_distribution', {}),
            'age_distribution': summary.get('age_distribution', {}),
            'confidence_scores': self._calculate_confidence_scores(people)
        }
    
    def _format_people_analysis(self, people_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format individual people analysis"""
        formatted_people = []
        
        for person in people_data:
            formatted_person = {
                'person_id': person.get('person_id', 0),
                'demographics': self._format_demographics(person),
                'physical_attributes': self._format_physical_attributes(person),
                'appearance': self._format_appearance(person),
                'pose_analysis': self._format_pose_analysis(person),
                'confidence_metrics': self._format_confidence_metrics(person)
            }
            
            formatted_people.append(formatted_person)
        
        return formatted_people
    
    def _format_demographics(self, person: Dict[str, Any]) -> Dict[str, Any]:
        """Format demographic information"""
        face_analysis = person.get('face_analysis', {})
        
        age_info = face_analysis.get('age', {})
        gender_info = face_analysis.get('gender', {})
        
        return {
            'age': {
                'estimated_age': age_info.get('estimated_age', 'unknown'),
                'age_range': age_info.get('age_range', 'unknown'),
                'confidence': 'high' if age_info.get('estimated_age', 0) > 0 else 'low'
            },
            'gender': {
                'prediction': gender_info.get('prediction', 'unknown'),
                'confidence': self._confidence_level(gender_info.get('confidence', 0))
            }
        }
    
    def _format_physical_attributes(self, person: Dict[str, Any]) -> Dict[str, Any]:
        """Format physical attributes"""
        face_analysis = person.get('face_analysis', {})
        hair_analysis = person.get('hair_analysis', {})
        body_analysis = person.get('body_analysis', {})
        
        return {
            'facial_features': {
                'face_shape': face_analysis.get('face_features', {}).get('face_shape', 'unknown'),
                'skin_tone': face_analysis.get('face_features', {}).get('skin_tone', 'unknown'),
                'facial_hair': face_analysis.get('face_features', {}).get('facial_hair', 'none')
            },
            'hair': {
                'detected': hair_analysis.get('hair_detected', False),
                'style': hair_analysis.get('hair_style', 'unknown'),
                'color': hair_analysis.get('hair_color', 'unknown'),
                'length': hair_analysis.get('hair_length', 'unknown'),
                'texture': hair_analysis.get('hair_texture', 'unknown')
            },
            'body': {
                'build': body_analysis.get('body_measurements', {}).get('build', 'unknown'),
                'height_estimate': body_analysis.get('body_measurements', {}).get('height_estimate', 'unknown'),
                'visible_parts': body_analysis.get('visible_body_parts', [])
            }
        }
    
    def _format_appearance(self, person: Dict[str, Any]) -> Dict[str, Any]:
        """Format appearance analysis"""
        appearance = person.get('appearance_analysis', {})
        
        clothing = appearance.get('clothing', {})
        
        return {
            'clothing': {
                'detected_items': clothing.get('detected_items', []),
                'dominant_colors': clothing.get('colors', []),
                'style_category': clothing.get('style', 'unknown')
            },
            'accessories': appearance.get('accessories', []),
            'overall_style': appearance.get('overall_style', 'unknown'),
            'color_palette': self._extract_color_palette(appearance)
        }
    
    def _format_pose_analysis(self, person: Dict[str, Any]) -> Dict[str, Any]:
        """Format pose and body analysis"""
        body_analysis = person.get('body_analysis', {})
        
        return {
            'pose_detected': body_analysis.get('pose_detected', False),
            'body_position': body_analysis.get('body_position', 'unknown'),
            'pose_confidence': body_analysis.get('pose_confidence', 0),
            'activity': self._infer_activity(body_analysis),
            'orientation': self._determine_orientation(body_analysis)
        }
    
    def _format_confidence_metrics(self, person: Dict[str, Any]) -> Dict[str, Any]:
        """Format confidence metrics for all analyses"""
        face_analysis = person.get('face_analysis', {})
        body_analysis = person.get('body_analysis', {})
        
        return {
            'overall_confidence': self._calculate_overall_confidence(person),
            'face_detection_confidence': face_analysis.get('confidence', 0) if 'bbox' in str(face_analysis) else 0,
            'age_confidence': 'high' if face_analysis.get('age', {}).get('estimated_age', 0) > 0 else 'low',
            'gender_confidence': self._confidence_level(face_analysis.get('gender', {}).get('confidence', 0)),
            'pose_confidence': body_analysis.get('pose_confidence', 0)
        }
    
    def _format_scene_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format overall scene analysis"""
        return {
            'scene_type': self._determine_scene_type(results),
            'lighting_conditions': self._analyze_lighting(results),
            'image_quality': self._assess_image_quality(results),
            'analysis_challenges': self._identify_challenges(results)
        }
    
    def _format_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format metadata information"""
        return {
            'analysis_timestamp': results.get('timestamp'),
            'processing_time_ms': 'N/A',  # Would need timing implementation
            'model_versions': {
                'face_detection': 'MediaPipe 0.10.x',
                'age_gender': 'DeepFace',
                'pose_estimation': 'MediaPipe Pose',
                'object_detection': 'YOLOv8'
            },
            'capabilities_used': self._list_capabilities_used(results)
        }
    
    # Helper methods
    def _confidence_level(self, confidence: float) -> str:
        """Convert numerical confidence to descriptive level"""
        if confidence >= 0.8:
            return 'high'
        elif confidence >= 0.6:
            return 'medium'
        elif confidence >= 0.4:
            return 'low'
        else:
            return 'very_low'
    
    def _calculate_confidence_scores(self, people: List[Dict]) -> Dict[str, float]:
        """Calculate average confidence scores"""
        if not people:
            return {'overall': 0.0, 'face_detection': 0.0, 'pose_detection': 0.0}
        
        face_confidences = []
        pose_confidences = []
        
        for person in people:
            face_analysis = person.get('face_analysis', {})
            body_analysis = person.get('body_analysis', {})
            
            if 'confidence' in str(face_analysis):
                face_confidences.append(0.8)  # Placeholder
            
            pose_conf = body_analysis.get('pose_confidence', 0)
            if pose_conf > 0:
                pose_confidences.append(pose_conf)
        
        return {
            'overall': (sum(face_confidences) + sum(pose_confidences)) / (len(face_confidences) + len(pose_confidences)) if (face_confidences or pose_confidences) else 0.0,
            'face_detection': sum(face_confidences) / len(face_confidences) if face_confidences else 0.0,
            'pose_detection': sum(pose_confidences) / len(pose_confidences) if pose_confidences else 0.0
        }
    
    def _calculate_overall_confidence(self, person: Dict[str, Any]) -> float:
        """Calculate overall confidence for a person"""
        confidences = []
        
        # Face detection confidence
        face_analysis = person.get('face_analysis', {})
        if face_analysis.get('age', {}).get('estimated_age', 0) > 0:
            confidences.append(0.8)
        
        # Pose confidence
        body_analysis = person.get('body_analysis', {})
        pose_conf = body_analysis.get('pose_confidence', 0)
        if pose_conf > 0:
            confidences.append(pose_conf)
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _extract_color_palette(self, appearance: Dict[str, Any]) -> List[str]:
        """Extract color palette from appearance analysis"""
        colors = appearance.get('clothing', {}).get('colors', [])
        return colors[:5]  # Return top 5 colors
    
    def _infer_activity(self, body_analysis: Dict[str, Any]) -> str:
        """Infer activity based on pose analysis"""
        position = body_analysis.get('body_position', 'unknown')
        
        activity_map = {
            'standing': 'standing',
            'sitting': 'sitting',
            'walking': 'walking',
            'running': 'running'
        }
        
        return activity_map.get(position, 'stationary')
    
    def _determine_orientation(self, body_analysis: Dict[str, Any]) -> str:
        """Determine person orientation"""
        # This would need more sophisticated analysis
        return 'frontal'  # Placeholder
    
    def _determine_scene_type(self, results: Dict[str, Any]) -> str:
        """Determine the type of scene"""
        people_count = len(results.get('people', []))
        
        if people_count == 0:
            return 'no_people'
        elif people_count == 1:
            return 'portrait'
        elif people_count <= 3:
            return 'small_group'
        else:
            return 'large_group'
    
    def _analyze_lighting(self, results: Dict[str, Any]) -> str:
        """Analyze lighting conditions"""
        # This would need image analysis
        return 'adequate'  # Placeholder
    
    def _assess_image_quality(self, results: Dict[str, Any]) -> str:
        """Assess overall image quality"""
        image_info = results.get('image_info', {})
        width = image_info.get('width', 0)
        height = image_info.get('height', 0)
        
        if width >= 1024 and height >= 768:
            return 'high'
        elif width >= 640 and height >= 480:
            return 'medium'
        else:
            return 'low'
    
    def _identify_challenges(self, results: Dict[str, Any]) -> List[str]:
        """Identify analysis challenges"""
        challenges = []
        
        people = results.get('people', [])
        if not people:
            challenges.append('no_people_detected')
        
        for person in people:
            face_analysis = person.get('face_analysis', {})
            if not face_analysis.get('age', {}).get('estimated_age'):
                challenges.append('face_detection_issues')
                break
        
        return challenges
    
    def _list_capabilities_used(self, results: Dict[str, Any]) -> List[str]:
        """List which capabilities were used in analysis"""
        capabilities = []
        
        people = results.get('people', [])
        if people:
            capabilities.append('person_detection')
            
            for person in people:
                if person.get('face_analysis', {}).get('age'):
                    capabilities.extend(['face_detection', 'age_estimation', 'gender_detection'])
                
                if person.get('body_analysis', {}).get('pose_detected'):
                    capabilities.append('pose_estimation')
                
                if person.get('hair_analysis', {}).get('hair_detected'):
                    capabilities.append('hair_analysis')
                
                if person.get('appearance_analysis', {}).get('clothing'):
                    capabilities.append('clothing_detection')
        
        return list(set(capabilities))  # Remove duplicates
    
    def _format_error_response(self, error_message: str) -> Dict[str, Any]:
        """Format error response"""
        return {
            'success': False,
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'processing_info': {},
            'detection_summary': {},
            'people': [],
            'scene_analysis': {},
            'metadata': {}
        }
    
    def format_capabilities_response(self, capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Format capabilities endpoint response"""
        return {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'capabilities': capabilities,
            'status': 'operational'
        } 