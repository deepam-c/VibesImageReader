import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import logging
from typing import List, Dict, Any, Tuple
import pickle
import os
from ultralytics import YOLO
from datetime import datetime

logger = logging.getLogger(__name__)

class PersonAnalyzer:
    """
    Comprehensive person analysis using multiple computer vision models
    """
    
    def __init__(self):
        """Initialize all the models and components"""
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize all computer vision models"""
        try:
            # MediaPipe for pose and face detection
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_pose = mp.solutions.pose
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5
            )
            
            # YOLO for object detection (clothing, accessories)
            try:
                self.yolo_model = YOLO('yolov8n.pt')  # Will download if not present
            except Exception as e:
                logger.warning(f"YOLO model not available: {e}")
                self.yolo_model = None
            
            # Hair cascade classifier
            self.load_hair_classifier()
            
            logger.info("Person analyzer models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def load_hair_classifier(self):
        """Load or create hair detection classifier"""
        try:
            # Try to load pre-trained hair classifier
            hair_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
            if os.path.exists(hair_cascade_path):
                self.hair_cascade = cv2.CascadeClassifier(hair_cascade_path)
            else:
                self.hair_cascade = None
                logger.warning("Hair cascade classifier not found")
        except Exception as e:
            logger.warning(f"Could not load hair classifier: {e}")
            self.hair_cascade = None
    
    def analyze_image(self, image: np.ndarray, is_video_frame: bool = False) -> Dict[str, Any]:
        """
        Main analysis function that coordinates all analysis tasks
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'image_info': self.get_image_info(image),
            'people': [],
            'summary': {}
        }
        
        try:
            # Convert BGR to RGB for most models
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces and people
            face_results = self.detect_faces(rgb_image)
            pose_results = self.detect_poses(rgb_image)
            
            # Analyze each detected person
            for i, face in enumerate(face_results):
                person_analysis = self.analyze_person(rgb_image, image, face, pose_results, i)
                results['people'].append(person_analysis)
            
            # Generate summary
            results['summary'] = self.generate_summary(results['people'])
            
            # Additional scene analysis if no faces detected but poses found
            if not face_results and pose_results:
                for i, pose in enumerate(pose_results):
                    person_analysis = self.analyze_person_from_pose(rgb_image, image, pose, i)
                    results['people'].append(person_analysis)
            
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            results['error'] = str(e)
        
        return results
    
    def get_image_info(self, image: np.ndarray) -> Dict[str, Any]:
        """Get basic image information"""
        height, width = image.shape[:2]
        return {
            'width': int(width),
            'height': int(height),
            'channels': int(image.shape[2]) if len(image.shape) == 3 else 1,
            'aspect_ratio': round(width / height, 2)
        }
    
    def detect_faces(self, rgb_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in the image"""
        faces = []
        try:
            results = self.face_detection.process(rgb_image)
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = rgb_image.shape
                    
                    face_info = {
                        'bbox': {
                            'x': int(bbox.xmin * w),
                            'y': int(bbox.ymin * h),
                            'width': int(bbox.width * w),
                            'height': int(bbox.height * h)
                        },
                        'confidence': detection.score[0] if detection.score else 0.0
                    }
                    faces.append(face_info)
        
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
        
        return faces
    
    def detect_poses(self, rgb_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect human poses in the image"""
        poses = []
        try:
            results = self.pose.process(rgb_image)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                poses.append({
                    'landmarks': landmarks,
                    'pose_confidence': np.mean([lm['visibility'] for lm in landmarks])
                })
        
        except Exception as e:
            logger.error(f"Error in pose detection: {e}")
        
        return poses
    
    def analyze_person(self, rgb_image: np.ndarray, bgr_image: np.ndarray, 
                      face_info: Dict, pose_results: List, person_id: int) -> Dict[str, Any]:
        """Comprehensive analysis of a detected person"""
        person_data = {
            'person_id': person_id,
            'face_analysis': {},
            'body_analysis': {},
            'appearance_analysis': {},
            'hair_analysis': {}
        }
        
        try:
            # Extract face region
            bbox = face_info['bbox']
            face_region = rgb_image[
                bbox['y']:bbox['y'] + bbox['height'],
                bbox['x']:bbox['x'] + bbox['width']
            ]
            
            if face_region.size > 0:
                # Face analysis using DeepFace
                person_data['face_analysis'] = self.analyze_face(face_region, bgr_image, bbox)
                
                # Hair analysis
                person_data['hair_analysis'] = self.analyze_hair(rgb_image, bbox)
                
                # Body and pose analysis
                if pose_results:
                    person_data['body_analysis'] = self.analyze_body(pose_results[0] if pose_results else None)
                
                # Appearance analysis
                person_data['appearance_analysis'] = self.analyze_appearance(rgb_image, bbox, pose_results)
        
        except Exception as e:
            logger.error(f"Error analyzing person {person_id}: {e}")
            person_data['error'] = str(e)
        
        return person_data
    
    def analyze_face(self, face_region: np.ndarray, full_image: np.ndarray, bbox: Dict) -> Dict[str, Any]:
        """Detailed face analysis including age, gender, emotion"""
        face_analysis = {}
        
        try:
            # Convert face region back to BGR for DeepFace
            face_bgr = cv2.cvtColor(face_region, cv2.COLOR_RGB2BGR)
            
            # Age and gender estimation
            try:
                age_gender = DeepFace.analyze(
                    face_bgr, 
                    actions=['age', 'gender'], 
                    enforce_detection=False
                )
                
                if isinstance(age_gender, list):
                    age_gender = age_gender[0]
                
                face_analysis['age'] = {
                    'estimated_age': int(age_gender.get('age', 0)),
                    'age_range': self.get_age_range(age_gender.get('age', 0))
                }
                
                face_analysis['gender'] = {
                    'prediction': age_gender.get('dominant_gender', 'unknown'),
                    'confidence': max(age_gender.get('gender', {}).values()) if age_gender.get('gender') else 0
                }
                
            except Exception as e:
                logger.warning(f"DeepFace analysis failed: {e}")
                face_analysis['age'] = {'estimated_age': 0, 'age_range': 'unknown'}
                face_analysis['gender'] = {'prediction': 'unknown', 'confidence': 0}
            
            # Emotion analysis
            try:
                emotion_result = DeepFace.analyze(
                    face_bgr, 
                    actions=['emotion'], 
                    enforce_detection=False
                )
                
                if isinstance(emotion_result, list):
                    emotion_result = emotion_result[0]
                
                emotions = emotion_result.get('emotion', {})
                face_analysis['emotion'] = {
                    'dominant_emotion': emotion_result.get('dominant_emotion', 'unknown'),
                    'all_emotions': emotions
                }
                
            except Exception as e:
                logger.warning(f"Emotion analysis failed: {e}")
                face_analysis['emotion'] = {'dominant_emotion': 'unknown', 'all_emotions': {}}
            
            # Face shape and features
            face_analysis['face_features'] = self.analyze_face_features(face_region)
            
        except Exception as e:
            logger.error(f"Error in face analysis: {e}")
        
        return face_analysis
    
    def analyze_hair(self, image: np.ndarray, face_bbox: Dict) -> Dict[str, Any]:
        """Analyze hair style, color, and characteristics"""
        hair_analysis = {
            'hair_detected': False,
            'hair_style': 'unknown',
            'hair_color': 'unknown',
            'hair_length': 'unknown',
            'hair_texture': 'unknown'
        }
        
        try:
            # Extend bbox upward to capture hair region
            h, w = image.shape[:2]
            hair_bbox = {
                'x': max(0, face_bbox['x'] - face_bbox['width'] // 4),
                'y': max(0, face_bbox['y'] - face_bbox['height']),
                'width': min(w, face_bbox['width'] + face_bbox['width'] // 2),
                'height': min(h, face_bbox['height'] + face_bbox['height'] // 2)
            }
            
            # Extract hair region
            hair_region = image[
                hair_bbox['y']:hair_bbox['y'] + hair_bbox['height'],
                hair_bbox['x']:hair_bbox['x'] + hair_bbox['width']
            ]
            
            if hair_region.size > 0:
                # Basic hair detection and analysis
                hair_analysis.update(self.detect_hair_characteristics(hair_region))
        
        except Exception as e:
            logger.error(f"Error in hair analysis: {e}")
        
        return hair_analysis
    
    def analyze_body(self, pose_data: Dict) -> Dict[str, Any]:
        """Analyze body pose and characteristics"""
        body_analysis = {
            'pose_detected': pose_data is not None,
            'body_position': 'unknown',
            'visible_body_parts': [],
            'pose_confidence': 0
        }
        
        if pose_data:
            try:
                landmarks = pose_data['landmarks']
                body_analysis['pose_confidence'] = pose_data.get('pose_confidence', 0)
                
                # Determine body position
                body_analysis['body_position'] = self.determine_body_position(landmarks)
                
                # Identify visible body parts
                body_analysis['visible_body_parts'] = self.identify_visible_body_parts(landmarks)
                
                # Body measurements estimation
                body_analysis['body_measurements'] = self.estimate_body_measurements(landmarks)
                
            except Exception as e:
                logger.error(f"Error in body analysis: {e}")
        
        return body_analysis
    
    def analyze_appearance(self, image: np.ndarray, face_bbox: Dict, pose_results: List) -> Dict[str, Any]:
        """Analyze clothing, accessories, and overall appearance"""
        appearance_analysis = {
            'clothing': {
                'detected_items': [],
                'colors': [],
                'style': 'unknown'
            },
            'accessories': [],
            'overall_style': 'unknown'
        }
        
        try:
            # Use YOLO for object detection if available
            if self.yolo_model:
                clothing_items = self.detect_clothing_items(image)
                appearance_analysis['clothing']['detected_items'] = clothing_items
            
            # Color analysis
            appearance_analysis['clothing']['colors'] = self.analyze_clothing_colors(image, face_bbox)
            
            # Style classification
            appearance_analysis['overall_style'] = self.classify_style(image, appearance_analysis['clothing'])
            
        except Exception as e:
            logger.error(f"Error in appearance analysis: {e}")
        
        return appearance_analysis
    
    # Helper methods
    def get_age_range(self, age: int) -> str:
        """Convert age to age range"""
        if age < 13:
            return 'child'
        elif age < 20:
            return 'teenager'
        elif age < 30:
            return 'young_adult'
        elif age < 50:
            return 'adult'
        elif age < 70:
            return 'middle_aged'
        else:
            return 'senior'
    
    def analyze_face_features(self, face_region: np.ndarray) -> Dict[str, Any]:
        """Analyze facial features and characteristics"""
        return {
            'face_shape': 'oval',  # Placeholder - would need specialized model
            'skin_tone': 'medium',  # Placeholder - would need color analysis
            'facial_hair': 'none'   # Placeholder - would need specialized detection
        }
    
    def detect_hair_characteristics(self, hair_region: np.ndarray) -> Dict[str, Any]:
        """Detect hair characteristics from hair region"""
        # Basic color analysis
        average_color = np.mean(hair_region, axis=(0, 1))
        hair_color = self.classify_hair_color(average_color)
        
        return {
            'hair_detected': True,
            'hair_color': hair_color,
            'hair_style': 'medium',  # Placeholder
            'hair_length': 'medium', # Placeholder
            'hair_texture': 'straight' # Placeholder
        }
    
    def classify_hair_color(self, color: np.ndarray) -> str:
        """Classify hair color based on RGB values"""
        r, g, b = color
        
        if r < 50 and g < 50 and b < 50:
            return 'black'
        elif r > 200 and g > 200 and b > 200:
            return 'white'
        elif r > 150 and g > 100 and b < 100:
            return 'blonde'
        elif r > 100 and g < 80 and b < 80:
            return 'brown'
        elif r > 120 and g < 70 and b < 70:
            return 'red'
        else:
            return 'brown'  # Default
    
    def determine_body_position(self, landmarks: List[Dict]) -> str:
        """Determine if person is standing, sitting, etc."""
        # Simplified logic based on pose landmarks
        return 'standing'  # Placeholder
    
    def identify_visible_body_parts(self, landmarks: List[Dict]) -> List[str]:
        """Identify which body parts are visible"""
        visible_parts = []
        
        # Check visibility of major body parts
        body_parts = {
            'head': [0, 1, 2, 3, 4],
            'shoulders': [11, 12],
            'arms': [13, 14, 15, 16],
            'torso': [11, 12, 23, 24],
            'legs': [23, 24, 25, 26, 27, 28]
        }
        
        for part_name, indices in body_parts.items():
            if any(landmarks[i]['visibility'] > 0.5 for i in indices if i < len(landmarks)):
                visible_parts.append(part_name)
        
        return visible_parts
    
    def estimate_body_measurements(self, landmarks: List[Dict]) -> Dict[str, str]:
        """Estimate body measurements and build"""
        return {
            'build': 'average',  # Placeholder
            'height_estimate': 'average'  # Placeholder
        }
    
    def detect_clothing_items(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect clothing items using YOLO"""
        clothing_items = []
        
        if self.yolo_model:
            try:
                results = self.yolo_model(image)
                
                # Filter for clothing-related classes
                clothing_classes = ['person', 'tie', 'handbag', 'backpack', 'umbrella', 'shoe']
                
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = self.yolo_model.names[class_id]
                        
                        if class_name in clothing_classes:
                            clothing_items.append({
                                'item': class_name,
                                'confidence': float(box.conf[0]),
                                'bbox': box.xyxy[0].tolist()
                            })
                            
            except Exception as e:
                logger.warning(f"YOLO detection failed: {e}")
        
        return clothing_items
    
    def analyze_clothing_colors(self, image: np.ndarray, face_bbox: Dict) -> List[str]:
        """Analyze dominant colors in clothing areas"""
        h, w = image.shape[:2]
        
        # Define clothing region (below face)
        clothing_region = image[
            face_bbox['y'] + face_bbox['height']:h,
            max(0, face_bbox['x'] - face_bbox['width']//2):
            min(w, face_bbox['x'] + face_bbox['width'] + face_bbox['width']//2)
        ]
        
        if clothing_region.size > 0:
            # Simple color analysis
            colors = []
            average_color = np.mean(clothing_region, axis=(0, 1))
            color_name = self.classify_color(average_color)
            colors.append(color_name)
            return colors
        
        return ['unknown']
    
    def classify_color(self, color: np.ndarray) -> str:
        """Classify RGB color into basic color names"""
        r, g, b = color
        
        if r > 200 and g > 200 and b > 200:
            return 'white'
        elif r < 50 and g < 50 and b < 50:
            return 'black'
        elif r > 150 and g < 100 and b < 100:
            return 'red'
        elif r < 100 and g > 150 and b < 100:
            return 'green'
        elif r < 100 and g < 100 and b > 150:
            return 'blue'
        elif r > 150 and g > 150 and b < 100:
            return 'yellow'
        elif r > 150 and g < 100 and b > 150:
            return 'purple'
        elif r > 200 and g > 100 and b < 100:
            return 'orange'
        else:
            return 'mixed'
    
    def classify_style(self, image: np.ndarray, clothing_info: Dict) -> str:
        """Classify overall style based on clothing and appearance"""
        # This would need a more sophisticated model
        # For now, return a placeholder
        return 'casual'
    
    def analyze_person_from_pose(self, rgb_image: np.ndarray, bgr_image: np.ndarray, 
                                pose_data: Dict, person_id: int) -> Dict[str, Any]:
        """Analyze person when only pose is detected (no face)"""
        return {
            'person_id': person_id,
            'face_analysis': {'detected': False},
            'body_analysis': self.analyze_body(pose_data),
            'appearance_analysis': {'estimated': True},
            'hair_analysis': {'detected': False}
        }
    
    def generate_summary(self, people_data: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics from all detected people"""
        summary = {
            'total_people': len(people_data),
            'gender_distribution': {'male': 0, 'female': 0, 'unknown': 0},
            'age_distribution': {'child': 0, 'teenager': 0, 'young_adult': 0, 'adult': 0, 'middle_aged': 0, 'senior': 0},
            'dominant_colors': [],
            'common_styles': []
        }
        
        for person in people_data:
            # Gender distribution
            gender = person.get('face_analysis', {}).get('gender', {}).get('prediction', 'unknown')
            if gender in summary['gender_distribution']:
                summary['gender_distribution'][gender] += 1
            
            # Age distribution
            age_range = person.get('face_analysis', {}).get('age', {}).get('age_range', 'unknown')
            if age_range in summary['age_distribution']:
                summary['age_distribution'][age_range] += 1
        
        return summary 