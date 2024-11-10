import torch
from torchvision import models, transforms
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64  # Ensure base64 is imported

app = Flask(__name__)

# Load the pre-trained ResNet50 model and modify the final layer for 4 classes (glass, paper, plastic, waste)
model = models.resnet50(weights=None)  # Use pre-trained weights if needed, but we load custom model weights later
model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 4 classes for waste classification
# Load model weights with `weights_only=True` if supported
model.load_state_dict(torch.load('resNet50_Final.pth', map_location=torch.device('cpu'), weights_only=True))
 # Load your model weights
model.eval()  # Set the model to evaluation mode

# Image transformation for preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the class labels (these should match the output of your model)
class_names = ['glass', 'paper', 'plastic', 'waste']
@app.route('/')
def home():
    return render_template('Index.html')  # Ensure 'index.html' is in the templates folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the base64 image from the frontend
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Debugging: Log received data
        print("Received image data from frontend.")

        # Decode the base64-encoded image
        img_data = base64.b64decode(data['image'].split(',')[1])  # Decode the image data
        img = Image.open(io.BytesIO(img_data))  # Open the image

        # Debugging: Log image mode and size
        print(f"Image mode: {img.mode}, Image size: {img.size}")

        # Check if image is in a valid format
        if img.mode != 'RGB':
            img = img.convert('RGB')  # Convert image to RGB if not already

        # Apply the transformations and add batch dimension
        img = transform(img).unsqueeze(0)

        # Debugging: Log tensor shape
        print(f"Image tensor shape: {img.shape}")

        # Perform the prediction
        with torch.no_grad():  # We don't need gradients for inference
            outputs = model(img)  # Pass the image through the model
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get the class index with the highest probability
            top_p, top_class = probs.topk(1, dim=1)
            
            # Get the class name and confidence
            prediction = class_names[top_class.item()]
            confidence = top_p.item() * 100  # Convert to percentage

            if confidence < 90:
                prediction = 'unknown'
                confidence_message = "Low confidence"
            else:
                confidence_message = f"{confidence:.2f}%"
        
        # Debugging line
        print(f"Prediction: {prediction}, Confidence: {confidence_message}")

        # Return prediction and confidence as JSON response
        return jsonify({
            'prediction': prediction,
            'confidence_message': confidence_message,
            'image': data['image']  # Return the image data to display it on the frontend
        })
    
    except Exception as e:
        # Log the error message for debugging purposes
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': 'Error processing the image.'}), 500

if __name__ == "__main__":
   app.run(debug=True, host="0.0.0.0", port=5001)

