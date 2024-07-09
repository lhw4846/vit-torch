import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2

from vit_model import VisionTransformer


def preprocess_image(image, transform):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


def predict(frame, transform, model, device):
    image = preprocess_image(frame, transform).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()


if __name__ == '__main__':
    # Initialize model
    model = VisionTransformer(img_size=224, patch_size=16, num_classes=10)
    model.load_state_dict(torch.load('vit_model.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"device: {device}")

    # Set transform for preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Open camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Predict
        predicted_class = predict(frame, model, device)

        # Display
        cv2.putText(frame, f'Predicted class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Camera', frame)

        # Wait key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release
    cap.release()
    cv2.destroyAllWindows()
