"""
Enhanced Flask application with WebSocket support for real-time dashboard updates
Replaces polling with efficient WebSocket communication
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import logging
import json
import time
import threading
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import queue
from functools import wraps

from services.smart_image_processor import SmartImageProcessor
from services.data_export_service import DataExportService
from services.dashboard_service import GetDashboardReadings
from infrastructure.repositories.firebase_repository import FirebaseAnalysisRepository

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

def async_route(f):
    """Decorator to handle async functions in Flask routes"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        def run_async():
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(f(*args, **kwargs))
            finally:
                loop.close()
        
        future = executor.submit(run_async)
        return future.result()  # This blocks until the async function completes
    
    return wrapper

def create_enhanced_flask_async_app() -> tuple[Flask, SocketIO]:
    """Create Flask application with WebSocket support for real-time dashboard updates"""
    
    # Create Flask app
    app = Flask(__name__)
    
    # Configure CORS with explicit settings for Azure deployment
    allowed_origins = [
             'https://orange-sea-0ffac2603.2.azurestaticapps.net',  # Production frontend
             'http://localhost:3000',  # Local development
             'http://localhost:3001',  # Alternative local port
             'https://localhost:3000',  # HTTPS local development
    ]
    
    CORS(app, 
         origins=allowed_origins,
         methods=['GET', 'POST', 'OPTIONS'],
         allow_headers=['Content-Type', 'Authorization', 'X-Requested-With'],
         supports_credentials=False,
         resources={
             r"/*": {
                 "origins": allowed_origins
             }
         }
    )
    
    # Initialize SocketIO with CORS settings
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",  # Allow all origins for development - more permissive
        async_mode='threading',
        logger=True,
        engineio_logger=True
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
    
    # Initialize Firebase repository for data export and dashboard
    try:
        firebase_config = {
            'service_account_path': 'firebase-service-account.json'
        }
        
        print("\n" + "="*80)
        print("🔥 ATTEMPTING FIREBASE INITIALIZATION")
        print("="*80)
        print(f"📁 Service account path: {firebase_config['service_account_path']}")
        
        # Check if file exists
        import os
        file_path = firebase_config['service_account_path']
        if os.path.exists(file_path):
            print(f"✅ Service account file found: {file_path}")
            print(f"📊 File size: {os.path.getsize(file_path)} bytes")
        else:
            print(f"❌ Service account file NOT found: {file_path}")
            print(f"📁 Current working directory: {os.getcwd()}")
            print(f"📁 Files in current directory: {os.listdir('.')}")
        
        firebase_repo = FirebaseAnalysisRepository(firebase_config)
        export_service = DataExportService(data_repository=firebase_repo)
        dashboard_service = GetDashboardReadings(analysis_repository=firebase_repo)
        
        print("✅ Firebase repository initialized successfully!")
        print("✅ Dashboard service using REAL Firebase data!")
        print("="*80 + "\n")
        logger.info("Export and dashboard services initialized with Firebase repository")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ FIREBASE INITIALIZATION FAILED")
        print("="*80)
        print(f"🚨 Error details: {str(e)}")
        print(f"🚨 Error type: {type(e).__name__}")
        import traceback
        print(f"🚨 Full traceback:")
        traceback.print_exc()
        print("🔄 Falling back to MOCK DATA mode")
        print("⚠️  Dashboard will show ZERO values until Firebase is fixed!")
        print("="*80 + "\n")
        
        logger.warning(f"Failed to initialize Firebase repository: {e}. Using mock data.")
        export_service = DataExportService()
        dashboard_service = GetDashboardReadings(analysis_repository=None)
    
    # WebSocket Dashboard Update Manager with Real-time Broadcasting
    class WebSocketDashboardManager:
        def __init__(self, socketio_instance):
            self.socketio = socketio_instance
            self.connected_clients = set()
            self.update_thread = None
            self.stop_event = threading.Event()
            self.last_dashboard_data = None
            
        def start_updates(self):
            """Start background thread for real-time WebSocket updates"""
            if self.update_thread is None or not self.update_thread.is_alive():
                self.stop_event.clear()
                self.update_thread = threading.Thread(target=self._websocket_update_loop, daemon=True)
                self.update_thread.start()
                logger.info("WebSocket dashboard update manager started")
        
        def stop_updates(self):
            """Stop background updates"""
            self.stop_event.set()
            if self.update_thread:
                self.update_thread.join(timeout=2)
            logger.info("WebSocket dashboard update manager stopped")
        
        def _websocket_update_loop(self):
            """Background loop for WebSocket dashboard updates"""
            def run_async_update():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(dashboard_service.get_dashboard_readings())
                except Exception as e:
                    logger.error(f"Error getting dashboard data: {e}")
                    return None
                finally:
                    loop.close()
            
            while not self.stop_event.is_set():
                try:
                    if self.connected_clients:  # Only update if clients are connected
                        dashboard_data = run_async_update()
                        if dashboard_data:
                            # Convert to dict for JSON serialization
                            dashboard_dict = {
                        'stats': dashboard_data.stats,
                        'recent_activity': dashboard_data.recent_activity,
                        'system_status': dashboard_data.system_status,
                        'performance_metrics': dashboard_data.performance_metrics,
                        'timestamp': dashboard_data.timestamp.isoformat(),
                        'success': True,
                                'realtime': True,
                                'websocket': True
                            }
                            
                            # Only broadcast if data has changed
                            if dashboard_dict != self.last_dashboard_data:
                                self.socketio.emit('dashboard_update', dashboard_dict, room='dashboard')
                                self.last_dashboard_data = dashboard_dict
                                logger.info(f"📊 WebSocket: Broadcasted dashboard update to {len(self.connected_clients)} clients")
                            
                except Exception as e:
                    logger.error(f"Error in WebSocket dashboard update loop: {e}")
                
                # Wait 2 seconds between updates (much more efficient than polling)
                self.stop_event.wait(2)
        
        def add_client(self, client_id):
            """Add a client to the connected set"""
            self.connected_clients.add(client_id)
            logger.info(f"Dashboard client connected: {client_id} (Total: {len(self.connected_clients)})")
            
        def remove_client(self, client_id):
            """Remove a client from the connected set"""
            self.connected_clients.discard(client_id)
            logger.info(f"Dashboard client disconnected: {client_id} (Total: {len(self.connected_clients)})")
    
    # Initialize WebSocket Dashboard Manager
    websocket_dashboard_manager = WebSocketDashboardManager(socketio)
    
    # WebSocket Event Handlers
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        client_id = request.sid
        logger.info(f"🔌 WebSocket client connected: {client_id}")
        print(f"🔌 WebSocket client connected: {client_id}")  # Also print to console
        emit('connection_confirmed', {'status': 'connected', 'client_id': client_id})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        client_id = request.sid
        websocket_dashboard_manager.remove_client(client_id)
        leave_room('dashboard')
        logger.info(f"🔌 WebSocket client disconnected: {client_id}")
        print(f"🔌 WebSocket client disconnected: {client_id}")  # Also print to console
    
    @socketio.on('connect_error')
    def handle_connect_error(error):
        """Handle connection errors"""
        logger.error(f"🔌 WebSocket connection error: {error}")
        print(f"🔌 WebSocket connection error: {error}")  # Also print to console
    
    @socketio.on('subscribe_dashboard')
    def handle_subscribe_dashboard():
        """Handle dashboard subscription"""
        client_id = request.sid
        logger.info(f"📊 Client {client_id} subscribing to dashboard")
        print(f"📊 Client {client_id} subscribing to dashboard")  # Also print to console
        
        join_room('dashboard')
        websocket_dashboard_manager.add_client(client_id)
        
        # Send immediate confirmation
        emit('dashboard_subscribed', {'status': 'subscribed', 'client_id': client_id})
        
        # Send current dashboard data immediately
        def get_current_data():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(dashboard_service.get_dashboard_readings())
            except Exception as e:
                logger.error(f"Error getting current dashboard data: {e}")
                return None
            finally:
                loop.close()
        
        try:
            current_data = get_current_data()
            if current_data:
                dashboard_dict = {
                    'stats': current_data.stats,
                    'recent_activity': current_data.recent_activity,
                    'system_status': current_data.system_status,
                    'performance_metrics': current_data.performance_metrics,
                    'timestamp': current_data.timestamp.isoformat(),
                    'success': True,
                    'realtime': True,
                    'websocket': True,
                    'initial_data': True
                }
                emit('dashboard_update', dashboard_dict)
                logger.info(f"📊 Sent initial dashboard data to client: {client_id}")
                print(f"📊 Sent initial dashboard data to client: {client_id}")  # Also print to console
            
            # Start the update manager if this is the first client
            if len(websocket_dashboard_manager.connected_clients) == 1:
                websocket_dashboard_manager.start_updates()
                
        except Exception as e:
            logger.error(f"Error sending initial dashboard data: {e}")
            print(f"Error sending initial dashboard data: {e}")  # Also print to console
            emit('dashboard_error', {'error': str(e)})
    
    @socketio.on('unsubscribe_dashboard')
    def handle_unsubscribe_dashboard():
        """Handle dashboard unsubscription"""
        client_id = request.sid
        leave_room('dashboard')
        websocket_dashboard_manager.remove_client(client_id)
        
        # Stop updates if no clients are connected
        if len(websocket_dashboard_manager.connected_clients) == 0:
            websocket_dashboard_manager.stop_updates()
    
    @socketio.on('request_dashboard_refresh')
    def handle_dashboard_refresh():
        """Handle manual dashboard refresh request"""
        def refresh_data():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(dashboard_service.refresh_dashboard_cache())
            except Exception as e:
                logger.error(f"Error refreshing dashboard: {e}")
                return None
            finally:
                loop.close()
        
        try:
            refreshed_data = refresh_data()
            if refreshed_data:
                dashboard_dict = {
                    'stats': refreshed_data.stats,
                    'recent_activity': refreshed_data.recent_activity,
                    'system_status': refreshed_data.system_status,
                    'performance_metrics': refreshed_data.performance_metrics,
                    'timestamp': refreshed_data.timestamp.isoformat(),
                    'success': True,
                    'realtime': True,
                    'websocket': True,
                    'manually_refreshed': True
                }
                # Broadcast to all dashboard subscribers
                socketio.emit('dashboard_update', dashboard_dict, room='dashboard')
                logger.info("📊 Manual dashboard refresh broadcasted to all clients")
            
        except Exception as e:
            logger.error(f"Error in manual refresh: {e}")
            emit('dashboard_error', {'error': str(e)})
    
    @socketio.on('new_analysis_notification')
    def handle_new_analysis():
        """Handle notification of new analysis - triggers dashboard update"""
        try:
            # Force a dashboard update when new analysis is added
            def get_updated_data():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(dashboard_service.refresh_dashboard_cache())
                finally:
                    loop.close()
            
            updated_data = get_updated_data()
            if updated_data:
                dashboard_dict = {
                    'stats': updated_data.stats,
                    'recent_activity': updated_data.recent_activity,
                    'system_status': updated_data.system_status,
                    'performance_metrics': updated_data.performance_metrics,
                    'timestamp': updated_data.timestamp.isoformat(),
                    'success': True,
                    'realtime': True,
                    'websocket': True,
                    'triggered_by_new_analysis': True
                }
                # Broadcast to all dashboard subscribers
                socketio.emit('dashboard_update', dashboard_dict, room='dashboard')
                socketio.emit('new_analysis', {'message': 'New analysis completed'}, room='dashboard')
                logger.info("📊 Dashboard updated due to new analysis")
                
        except Exception as e:
            logger.error(f"Error handling new analysis notification: {e}")
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '2.1.0-flask-async',
            'service': 'CV Analysis API with Async Dashboard (Flask)',
            'async_support': 'Threading + Background Tasks'
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
        """Handle OPTIONS requests for CORS"""
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
            
            # Process with enhanced AI (synchronous call) - same as working version
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
            
            # 🚀 WebSocket Notification: Trigger dashboard update for new analysis
            try:
                if websocket_dashboard_manager.connected_clients:
                    # Notify about new analysis completion
                    socketio.emit('new_analysis', {
                        'message': 'New CV analysis completed',
                        'analysis_id': result.get('analysis_id'),
                        'people_detected': result.get('detection_summary', {}).get('total_people_detected', 0),
                        'timestamp': datetime.now().isoformat()
                    }, room='dashboard')
                    
                    # Trigger dashboard data refresh
                    def get_updated_dashboard():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            return loop.run_until_complete(dashboard_service.refresh_dashboard_cache())
                        except Exception as e:
                            logger.error(f"Error refreshing dashboard after analysis: {e}")
                            return None
                        finally:
                            loop.close()
                    
                    updated_dashboard = get_updated_dashboard()
                    if updated_dashboard:
                        dashboard_dict = {
                            'stats': updated_dashboard.stats,
                            'recent_activity': updated_dashboard.recent_activity,
                            'system_status': updated_dashboard.system_status,
                            'performance_metrics': updated_dashboard.performance_metrics,
                            'timestamp': updated_dashboard.timestamp.isoformat(),
                            'success': True,
                            'realtime': True,
                            'websocket': True,
                            'triggered_by_new_analysis': True
                        }
                        socketio.emit('dashboard_update', dashboard_dict, room='dashboard')
                        logger.info(f"📊 Dashboard updated via WebSocket after new analysis - {len(websocket_dashboard_manager.connected_clients)} clients notified")
                    
            except Exception as ws_error:
                logger.error(f"Error sending WebSocket notification: {ws_error}")
                # Don't fail the analysis if WebSocket fails
            
            return jsonify(result), 200
            
        except Exception as e:
            logger.error(f"Error in analyze_image: {str(e)}")
            return jsonify({
                'error': f'Internal server error: {str(e)}',
                'success': False,
                'fallback': True
            }), 500

    @app.route('/dashboard-readings', methods=['GET'])
    def get_dashboard_readings():
        """Get real-time dashboard readings"""
        try:
            logger.info("Received dashboard readings request")
            
            # FORCE REFRESH CACHE to apply the new calculation logic (handle async properly)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            dashboard_data = loop.run_until_complete(dashboard_service.refresh_dashboard_cache())
            
            # Convert to dict for JSON response
            response_data = {
                'stats': dashboard_data.stats,
                'recent_activity': dashboard_data.recent_activity,
                'system_status': dashboard_data.system_status,
                'performance_metrics': dashboard_data.performance_metrics,
                'timestamp': dashboard_data.timestamp.isoformat(),
                'success': True,
                'cache_refreshed': True  # Indicate cache was refreshed
            }
            
            logger.info("Successfully retrieved dashboard readings with refreshed cache")
            return jsonify(response_data), 200
            
        except Exception as e:
            logger.error(f"Error getting dashboard readings: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return jsonify({
                'error': f'Failed to get dashboard readings: {str(e)}',
                'success': False,
                'fallback': True
            }), 500

    @app.route('/dashboard-refresh', methods=['POST'])
    def refresh_dashboard():
        """Force refresh dashboard cache"""
        try:
            logger.info("Received dashboard refresh request")
            
            # Force refresh using async (handle async properly)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            dashboard_data = loop.run_until_complete(dashboard_service.refresh_dashboard_cache())
            
            response_data = {
                'stats': dashboard_data.stats,
                'recent_activity': dashboard_data.recent_activity,
                'system_status': dashboard_data.system_status,
                'performance_metrics': dashboard_data.performance_metrics,
                'timestamp': dashboard_data.timestamp.isoformat(),
                'success': True,
                'message': 'Dashboard cache refreshed successfully',
                'force_refreshed': True
            }
            
            logger.info("Dashboard cache refreshed successfully")
            return jsonify(response_data), 200
            
        except Exception as e:
            logger.error(f"Error refreshing dashboard: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return jsonify({
                'error': f'Failed to refresh dashboard: {str(e)}',
                'success': False
            }), 500

    @app.route('/dashboard-stream', methods=['GET'])
    def dashboard_stream():
        """Real-time Server-Sent Events stream with proper async handling"""
        def generate():
            import uuid
            sub_id = str(uuid.uuid4())
            
            logger.info(f"Client connected to dashboard stream: {sub_id}")
            
            # Add subscriber to manager
            sub_queue = dashboard_manager.add_subscriber(sub_id)
            
            try:
                # Send initial data
                def get_initial_data():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(dashboard_service.get_dashboard_readings())
                    finally:
                        loop.close()
                
                try:
                    dashboard_data = get_initial_data()
                    initial_data = {
                        'stats': dashboard_data.stats,
                        'recent_activity': dashboard_data.recent_activity,
                        'system_status': dashboard_data.system_status,
                        'performance_metrics': dashboard_data.performance_metrics,
                        'timestamp': dashboard_data.timestamp.isoformat(),
                        'success': True,
                        'realtime': True
                    }
                    yield f"data: {json.dumps(initial_data)}\n\n"
                except Exception as e:
                    logger.error(f"Error sending initial data: {e}")
                    yield f"data: {json.dumps({'error': str(e), 'success': False})}\n\n"
                
                # Stream real-time updates
                while True:
                    try:
                        # Get data from queue (blocking with timeout)
                        data = sub_queue.get(timeout=30)  # 30 second timeout
                        yield f"data: {json.dumps(data)}\n\n"
                        
                    except queue.Empty:
                        # Send keepalive
                        yield f"data: {json.dumps({'keepalive': True, 'timestamp': datetime.now().isoformat()})}\n\n"
                        continue
                    except GeneratorExit:
                        logger.info(f"Client disconnected from dashboard stream: {sub_id}")
                        break
                    except Exception as e:
                        logger.error(f"Stream error for {sub_id}: {e}")
                        yield f"data: {json.dumps({'error': str(e), 'success': False})}\n\n"
                        break
                        
            finally:
                # Clean up subscriber
                dashboard_manager.remove_subscriber(sub_id)
                logger.info(f"Cleaned up subscriber: {sub_id}")
        
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Cache-Control'
            }
        )

    @app.route('/export-data', methods=['GET'])
    def export_data():
        """Export analysis data in various formats"""
        try:
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
                'version': '2.1.0-flask-async',
                'async_support': True
            }), 200
            
        except Exception as e:
            logger.error(f"Error in test_simple: {str(e)}")
            return jsonify({
                'error': f'Error: {str(e)}',
                'success': False
            }), 500

    @app.route('/debug/firebase-count', methods=['GET'])
    def debug_firebase_count():
        """Debug endpoint to check Firebase connection and record count"""
        try:
            if hasattr(export_service, 'data_repository') and export_service.data_repository:
                import asyncio
                
                # Handle async call properly (same as working version)
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Get all analyses
                analyses = loop.run_until_complete(
                    export_service.data_repository.get_all_analyses()
                )
                
                # Create summary
                record_summaries = []
                for i, analysis in enumerate(analyses):
                    summary = {
                        'index': i,
                        'id': getattr(analysis, 'id', 'no_id'),
                        'timestamp': str(analysis.timestamp) if analysis.timestamp else 'no_timestamp',
                        'has_detection_summary': bool(analysis.detection_summary),
                        'people_count': len(analysis.people_analysis) if analysis.people_analysis else 0
                    }
                    record_summaries.append(summary)
                
                return jsonify({
                    'success': True,
                    'firebase_connected': True,
                    'total_records': len(analyses),
                    'records': record_summaries
                })
            else:
                return jsonify({
                    'success': False,
                    'firebase_connected': False,
                    'error': 'No Firebase repository available'
                })
                
        except Exception as e:
            logger.error(f"Debug Firebase count error: {e}")
            return jsonify({
                'success': False,
                'firebase_connected': False,
                'error': str(e)
            }), 500

    @app.route('/test-ai', methods=['POST'])
    def test_ai():
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
            
            # Analyze (use sync version for consistency)
            result = enhanced_processor.analyze_image_sync(f"data:image/jpeg;base64,{base64_string}")
            
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

    # Cleanup on app shutdown
    @app.teardown_appcontext
    def cleanup(error):
        if error:
            logger.error(f"App context error: {error}")
        websocket_dashboard_manager.stop_updates()
    
    logger.info("🚀 Enhanced Flask App with Async Support created successfully")
    return app, socketio

if __name__ == '__main__':
    app, socketio = create_enhanced_flask_async_app()
    
    print("🚀 Starting Enhanced Flask CV Analysis API (Fixed)...")
    print("🤖 AI Models: Advanced AI Features Integrated")
    print("⚡ Real-time Support: Live dashboard updates")
    print("🏗️ Architecture: Flask + Enhanced Features (Fixed)")
    print("✅ Server starting on http://localhost:5000")
    print("\n📍 Available Endpoints:")
    print("  GET  /health              - Health check")
    print("  GET  /capabilities        - AI system capabilities")
    print("  POST /analyze-image       - Enhanced AI image analysis")
    print("  GET  /dashboard-readings  - Real-time dashboard readings")
    print("  GET  /dashboard-stream    - Real-time SSE stream")
    print("  POST /dashboard-refresh   - Dashboard cache refresh")
    print("  GET  /export-data         - Export analysis data")
    print("  GET  /export-formats      - Get supported export formats")
    print("  POST /test-simple         - Simple connectivity test")
    print("  POST /test-ai             - AI functionality test")
    print("  GET  /debug/firebase-count- Debug Firebase connection")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
    finally:
        print("✅ Server shutdown complete") 