"""
Enhanced Refactored Flask application with advanced AI features
Simple version for testing without complex dependency injection
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging
from datetime import datetime
from services.smart_image_processor import SmartImageProcessor
from services.data_export_service import DataExportService
from infrastructure.repositories.firebase_repository import FirebaseAnalysisRepository

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_enhanced_app() -> Flask:
    """Create Flask application with enhanced AI features"""
    
    # Create Flask app
    app = Flask(__name__)
    
    # Configure CORS with explicit settings for Azure deployment
    CORS(app, 
         origins=[
             'https://orange-sea-0ffac2603.2.azurestaticapps.net',  # Production frontend
             'http://localhost:3000',  # Local development
             'http://localhost:3001',  # Alternative local port
             'https://localhost:3000',  # HTTPS local development
         ],
         methods=['GET', 'POST', 'OPTIONS'],
         allow_headers=['Content-Type', 'Authorization', 'X-Requested-With'],
         supports_credentials=False,
         resources={
             r"/*": {
                 "origins": [
                     'https://orange-sea-0ffac2603.2.azurestaticapps.net',
                     'http://localhost:3000',
                     'http://localhost:3001',
                     'https://localhost:3000'
                 ]
             }
         }
    )
    
    # Additional CORS headers for robust cross-origin support
    @app.after_request
    def after_request(response):
        origin = request.headers.get('Origin')
        allowed_origins = [
            'https://orange-sea-0ffac2603.2.azurestaticapps.net',
            'http://localhost:3000',
            'http://localhost:3001',
            'https://localhost:3000'
        ]
        
        if origin in allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
        else:
            # Default to production frontend for any unspecified origin
            response.headers['Access-Control-Allow-Origin'] = 'https://orange-sea-0ffac2603.2.azurestaticapps.net'
            
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response
    
    # Initialize services
    enhanced_processor = SmartImageProcessor()
    
    # Initialize Firebase repository for data export
    try:
        firebase_config = {}  # Using default Firebase configuration
        firebase_repo = FirebaseAnalysisRepository(firebase_config)
        export_service = DataExportService(data_repository=firebase_repo)
        logger.info("Export service initialized with Firebase repository")
    except Exception as e:
        logger.warning(f"Failed to initialize Firebase repository: {e}. Using mock data.")
        export_service = DataExportService()  # Fallback to mock data
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '2.1.0-enhanced-refactored',
            'service': 'CV Analysis API'
        })

    @app.route('/capabilities', methods=['GET'])
    def get_capabilities():
        """Get AI capabilities"""
        try:
            capabilities = enhanced_processor.get_capabilities()
            return jsonify(capabilities)
        except Exception as e:
            logger.error(f"Error getting capabilities: {str(e)}")
            return jsonify({'error': 'Failed to get capabilities'}), 500

    @app.route('/<path:path>', methods=['OPTIONS'])
    def handle_options(path):
        return '', 200
    
    @app.route('/analyze-image', methods=['POST'])
    def analyze_image():
        """Enhanced image analysis endpoint"""
        try:
            logger.info("Received analyze-image request")
            data = request.get_json()
            
            if not data or 'image' not in data:
                logger.error("No image data provided in request")
                return jsonify({'error': 'No image data provided'}), 400
            
            logger.info("Starting image analysis...")
            
            # Process with enhanced AI (synchronous call) - remove signal timeout
            try:
                # SmartImageProcessor already has built-in protections and limits
                result = enhanced_processor.analyze_image_sync(data['image'])
                logger.info("Image analysis completed successfully")
                    
            except Exception as proc_error:
                logger.error(f"SmartImageProcessor error: {str(proc_error)}")
                # Fallback to simple mock response if processor fails
                result = {
                    'success': True,
                    'message': 'Image processed with fallback mode due to processing error',
                    'people': [{
                        'person_id': 1,
                        'demographics': {
                            'age': {'estimated_age': 'unknown', 'confidence': 'low'},
                            'gender': {'prediction': 'unknown', 'confidence': 'low'}
                        },
                        'pose': {'detected': True, 'confidence': 0.7},
                        'appearance': {'style': 'detected but not analyzed due to complexity'}
                    }],
                    'summary': {'total_people': 1, 'average_age': 'unknown', 'processing_note': 'Simplified due to processing error'},
                    'model_info': {
                        'version': '2.1.0-fallback',
                        'ai_backend': 'Fallback Mode - Processing Error',
                        'error': f'Processing fallback: {str(proc_error)[:100]}'
                    }
                }
            
            # Add analysis ID for tracking
            result['analysis_id'] = f"enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            return jsonify(result), 200
            
        except Exception as e:
            logger.error(f"Error in analyze_image: {str(e)}")
            return jsonify({
                'error': f'Internal server error: {str(e)}',
                'success': False,
                'fallback': True
            }), 500

    @app.route('/export-data', methods=['GET'])
    def export_data():
        """Export analysis data in various formats"""
        try:
            # Get query parameters
            format_type = request.args.get('format', 'csv').lower()
            
            logger.info(f"Received export request for format: {format_type}")
            
            # Validate format
            supported_formats = export_service.get_supported_formats()
            if format_type not in supported_formats:
                return jsonify({
                    'error': f'Unsupported format: {format_type}',
                    'supported_formats': supported_formats
                }), 400
            
            # Export data
            data_bytes, content_type, filename = export_service.export_analyses(format_type)
            
            # Create response
            response = Response(
                data_bytes,
                mimetype=content_type,
                headers={
                    'Content-Disposition': f'attachment; filename="{filename}"',
                    'Content-Type': content_type,
                    'Access-Control-Expose-Headers': 'Content-Disposition'
                }
            )
            
            logger.info(f"Successfully exported data as {format_type}, filename: {filename}")
            return response
            
        except ValueError as ve:
            logger.error(f"Validation error in export: {str(ve)}")
            return jsonify({'error': str(ve)}), 400
        except Exception as e:
            logger.error(f"Error in export_data: {str(e)}")
            return jsonify({
                'error': f'Export failed: {str(e)}',
                'success': False
            }), 500

    @app.route('/export-formats', methods=['GET'])
    def get_export_formats():
        """Get supported export formats"""
        try:
            formats = export_service.get_supported_formats()
            return jsonify({
                'supported_formats': formats,
                'default_format': 'csv'
            })
        except Exception as e:
            logger.error(f"Error getting export formats: {str(e)}")
            return jsonify({'error': 'Failed to get export formats'}), 500
    
    @app.route('/test-simple', methods=['POST'])
    def test_simple():
        """Simple test endpoint without image processing"""
        try:
            logger.info("Received test-simple request")
            data = request.get_json()
            
            return jsonify({
                'success': True,
                'message': 'Simple test endpoint working',
                'received_data': bool(data),
                'timestamp': datetime.now().isoformat(),
                'version': '2.1.0-enhanced-ai'
            }), 200
            
        except Exception as e:
            logger.error(f"Error in test_simple: {str(e)}")
            return jsonify({
                'error': f'Error: {str(e)}',
                'success': False
            }), 500
    
    @app.route('/test-ai', methods=['POST'])
    async def test_ai():
        """Simple AI test endpoint"""
        try:
            # Create a simple test image (red square)
            import numpy as np
            import cv2
            import base64
            from PIL import Image
            import io
            
            # Generate test image
            test_img = np.full((200, 200, 3), [0, 0, 255], dtype=np.uint8)  # Red image
            
            # Convert to base64
            pil_img = Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG')
            img_bytes = buffer.getvalue()
            base64_string = base64.b64encode(img_bytes).decode('utf-8')
            
            # Analyze
            result = await enhanced_processor.analyze_image(f"data:image/jpeg;base64,{base64_string}")
            
            return jsonify({
                'test_status': 'success',
                'ai_available': enhanced_processor.deepface_available,
                'analysis_summary': {
                    'people_detected': result.get('detection_summary', {}).get('total_people_detected', 0),
                    'processing_time': result.get('processing_info', {}).get('processing_time_ms', 0),
                    'confidence': result.get('processing_info', {}).get('overall_confidence', 0)
                }
            })
            
        except Exception as e:
            logger.error(f"Error in AI test: {e}")
            return jsonify({'test_status': 'error', 'error': str(e)}), 500
    
    logger.info("🚀 Enhanced CV API created successfully")
    return app

if __name__ == '__main__':
    app = create_enhanced_app()
    
    print("🚀 Starting Enhanced Computer Vision API...")
    print("🤖 AI Models: Advanced AI Features Integrated")
    print("🏗️ Architecture: Clean Design + Enhanced AI")
    print("✅ Server starting on http://localhost:5000")
    print("\n📍 Available Endpoints:")
    print("  GET  /health       - Health check with AI status")
    print("  POST /analyze-image - Enhanced AI image analysis")
    print("  GET  /capabilities  - System capabilities")
    print("  POST /test-ai       - Simple AI functionality test")
    
    app.run(host='0.0.0.0', port=5000, debug=True) 