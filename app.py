from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np
import cv2

pp = Flask(__name__)
CORS(app)

# We'll use DeepFace for face verification
try:
    from deepface import DeepFace
    USE_DEEPFACE = True
    print("✓ DeepFace loaded successfully")
except ImportError:
    print("⚠ DeepFace not available, using basic face detection")
    USE_DEEPFACE = False

# Alternative: face_recognition library
try:
    import face_recognition
    USE_FACE_RECOGNITION = True
    print("✓ face_recognition loaded successfully")
except ImportError:
    USE_FACE_RECOGNITION = False


def decode_image(base64_string):
    """Convert base64 string to numpy array (image)"""
    try:
        # Remove header if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def verify_faces_deepface(id_image, selfie_image):
    """Verify faces using DeepFace library"""
    try:
        result = DeepFace.verify(
            id_image, 
            selfie_image,
            model_name='VGG-Face',
            enforce_detection=False
        )
        
        distance = result['distance']
        verified = result['verified']
        
        # Convert distance to similarity percentage
        # Lower distance = higher similarity
        similarity = max(0, min(100, (1 - distance) * 100))
        
        return {
            'verified': verified,
            'similarity': round(similarity, 2),
            'distance': round(distance, 4),
            'method': 'DeepFace (VGG-Face)'
        }
    except Exception as e:
        print(f"DeepFace error: {e}")
        return None


def verify_faces_face_recognition(id_image, selfie_image):
    """Verify faces using face_recognition library"""
    try:
        # Get face encodings (128-dimensional vectors)
        id_encodings = face_recognition.face_encodings(id_image)
        selfie_encodings = face_recognition.face_encodings(selfie_image)
        
        if len(id_encodings) == 0:
            return {'error': 'No face detected in ID photo'}
        if len(selfie_encodings) == 0:
            return {'error': 'No face detected in selfie'}
        
        # Get first face from each image
        id_encoding = id_encodings[0]
        selfie_encoding = selfie_encodings[0]
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(id_encoding - selfie_encoding)
        
        # Threshold calibration (typical values)
        # distance < 0.45 = same person (high confidence)
        # distance 0.45-0.6 = uncertain
        # distance > 0.6 = different person
        
        verified = distance < 0.5
        
        # Convert distance to similarity percentage
        # distance of 0.0 = 100% similar
        # distance of 1.0 = 0% similar
        similarity = max(0, min(100, (1 - distance) * 100))
        
        return {
            'verified': verified,
            'similarity': round(similarity, 2),
            'distance': round(distance, 4),
            'method': 'face_recognition (dlib)',
            'threshold_used': 0.5
        }
    except Exception as e:
        print(f"face_recognition error: {e}")
        return None


def verify_faces_opencv(id_image, selfie_image):
    """Basic face detection using OpenCV (fallback method)"""
    try:
        # Load Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Convert to grayscale
        id_gray = cv2.cvtColor(id_image, cv2.COLOR_RGB2GRAY)
        selfie_gray = cv2.cvtColor(selfie_image, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        id_faces = face_cascade.detectMultiScale(id_gray, 1.1, 4)
        selfie_faces = face_cascade.detectMultiScale(selfie_gray, 1.1, 4)
        
        if len(id_faces) == 0:
            return {'error': 'No face detected in ID photo'}
        if len(selfie_faces) == 0:
            return {'error': 'No face detected in selfie'}
        
        # Basic verification (just checks if faces exist)
        return {
            'verified': True,
            'similarity': 75.0,  # Placeholder
            'method': 'OpenCV (Basic Detection)',
            'note': 'Basic face detection only - install DeepFace or face_recognition for accurate verification'
        }
    except Exception as e:
        print(f"OpenCV error: {e}")
        return None


@app.route('/api/verify-face', methods=['POST'])
def verify_face():
    """Main endpoint for face verification"""
    try:
        data = request.json
        
        if not data or 'idPhoto' not in data or 'selfie' not in data:
            return jsonify({'error': 'Missing idPhoto or selfie'}), 400
        
        # Decode images
        id_image = decode_image(data['idPhoto'])
        selfie_image = decode_image(data['selfie'])
        
        if id_image is None or selfie_image is None:
            return jsonify({'error': 'Failed to decode images'}), 400
        
        # Try verification methods in order of preference
        result = None
        
        if USE_DEEPFACE:
            result = verify_faces_deepface(id_image, selfie_image)
        
        if result is None and USE_FACE_RECOGNITION:
            result = verify_faces_face_recognition(id_image, selfie_image)
        
        if result is None:
            result = verify_faces_opencv(id_image, selfie_image)
        
        if result is None:
            return jsonify({'error': 'Face verification failed'}), 500
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'deepface_available': USE_DEEPFACE,
        'face_recognition_available': USE_FACE_RECOGNITION
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 KYC Automation Server Starting...")
    print("="*50)
    print(f"DeepFace Available: {USE_DEEPFACE}")
    print(f"face_recognition Available: {USE_FACE_RECOGNITION}")
    print("="*50 + "\n")
    
    app.run(debug=True, port=5000)
