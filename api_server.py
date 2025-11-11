from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow requests from any domain

# Import face verification libraries
try:
    from deepface import DeepFace
    USE_DEEPFACE = True
except ImportError:
    USE_DEEPFACE = False

try:
    import face_recognition
    USE_FACE_RECOGNITION = True
except ImportError:
    USE_FACE_RECOGNITION = False


def decode_image(base64_string):
    """Convert base64 string to numpy array"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)
    except Exception as e:
        return None

def verify_faces(id_image, selfie_image):
    """Main verification function"""
    if USE_DEEPFACE:
        try:
            result = DeepFace.verify(
                id_image, 
                selfie_image,
                model_name='VGG-Face',
                enforce_detection=False
            )
            
            distance = result['distance']
            verified = result['verified']
            similarity = max(0, min(100, (1 - distance) * 100))
            
            return {
                'success': True,
                'verified': verified,
                'similarity': round(similarity, 2),
                'distance': round(distance, 4),
                'method': 'DeepFace'
            }
        except Exception as e:
            pass
    
    if USE_FACE_RECOGNITION:
        try:
            id_encodings = face_recognition.face_encodings(id_image)
            selfie_encodings = face_recognition.face_encodings(selfie_image)
            
            if len(id_encodings) == 0 or len(selfie_encodings) == 0:
                return {'success': False, 'error': 'No face detected'}
            
            distance = np.linalg.norm(id_encodings[0] - selfie_encodings[0])
            verified = distance < 0.5
            similarity = max(0, min(100, (1 - distance) * 100))
            
            return {
                'success': True,
                'verified': verified,
                'similarity': round(similarity, 2),
                'distance': round(distance, 4),
                'method': 'face_recognition'
            }
        except Exception as e:
            pass
    
    return {'success': False, 'error': 'No verification method available'}


@app.route('/api/verify', methods=['POST'])
def verify():
    """
    API Endpoint for face verification
    
    Expected JSON body:
    {
        "id_photo": "base64_encoded_image",
        "selfie": "base64_encoded_image"
    }
    
    Returns:
    {
        "success": true,
        "verified": true/false,
        "similarity": 85.5,
        "distance": 0.145,
        "method": "DeepFace"
    }
    """
    try:
        data = request.json
        
        if not data or 'id_photo' not in data or 'selfie' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing id_photo or selfie in request body'
            }), 400
        
        # Decode images
        id_image = decode_image(data['id_photo'])
        selfie_image = decode_image(data['selfie'])
        
        if id_image is None or selfie_image is None:
            return jsonify({
                'success': False,
                'error': 'Failed to decode images'
            }), 400
        
        # Verify
        result = verify_faces(id_image, selfie_image)
        
        if not result.get('success'):
            return jsonify(result), 400
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'deepface_available': USE_DEEPFACE,
        'face_recognition_available': USE_FACE_RECOGNITION
    })


@app.route('/', methods=['GET'])
def home():
    """API documentation"""
    return jsonify({
        'name': 'KYC Face Verification API',
        'version': '1.0',
        'endpoints': {
            'POST /api/verify': 'Verify face match between ID and selfie',
            'GET /api/health': 'Check API health',
            'GET /': 'API documentation'
        },
        'usage_example': {
            'url': 'POST http://your-api-url.com/api/verify',
            'body': {
                'id_photo': 'base64_encoded_string',
                'selfie': 'base64_encoded_string'
            }
        }
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 KYC FACE VERIFICATION API")
    print("="*60)
    print(f"✓ DeepFace: {'Available' if USE_DEEPFACE else 'Not Available'}")
    print(f"✓ face_recognition: {'Available' if USE_FACE_RECOGNITION else 'Not Available'}")
    print("="*60)
    print("📡 Server running on: http://localhost:5000")
    print("📖 Documentation: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
