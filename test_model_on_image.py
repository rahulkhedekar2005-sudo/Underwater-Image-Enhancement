import cv2
import torch
from models.cnn_model import SimpleCNN

# -----------------------------
# Load trained CNN model
# -----------------------------
model = SimpleCNN()
model.load_state_dict(torch.load("underwater_cnn.pth", map_location="cpu"))
model.eval()

# -----------------------------
# Load input image
# -----------------------------
img_path = "personal_input\pic_21745_68331.jpg"  # change if needed

img = cv2.imread(img_path)

if img is None:
    raise ValueError("Image not found. Check file path.")

# Keep original image size
original_h, original_w = img.shape[:2]

# Convert BGR → RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize for CNN input
img_resized = cv2.resize(img_rgb, (128, 128))

# -----------------------------
# Convert image to tensor
# -----------------------------
tensor = torch.from_numpy(img_resized).float()
tensor = tensor.permute(2, 0, 1)  # HWC → CHW
tensor = tensor / 255.0
tensor = tensor.unsqueeze(0)      # Add batch dimension

# -----------------------------
# Run CNN inference
# -----------------------------
with torch.no_grad():
    output = model(tensor)

# -----------------------------
# Convert output tensor → image
# -----------------------------
output_img = output.squeeze(0).permute(1, 2, 0).numpy()
output_img = (output_img * 255).clip(0, 255).astype("uint8")

# RGB → BGR for OpenCV
output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

# Resize enhanced image back to original size
output_img = cv2.resize(output_img, (original_w, original_h))

# -----------------------------
# Display images (CORRECT)
# -----------------------------
cv2.imshow(" Enhanced  Image (CNN)", img)
cv2.imshow("Original Image ", output_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
