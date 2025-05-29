# 🧠 Fundus Disease Prediction using Deep Learning

This project uses an **ensemble of deep learning models (ResNet50, EfficientNetB0, DenseNet121)** to detect various **retinal diseases** from fundus images of the left and right eye. It also considers patient **age** and **gender** to improve accuracy.

---

## 🚀 Features

- ✅ Detects 8 eye conditions:
  - Normal (N)
  - Diabetic Retinopathy (D)
  - Glaucoma (G)
  - Cataract (C)
  - Age-related Macular Degeneration (A)
  - Hypertensive Retinopathy (H)
  - Myopia (M)
  - Other (O)
- ✅ Uses ensemble of 3 CNN models with attention modules.
- ✅ Supports image input + demographic features.
- ✅ Provides detailed probability scores and diagnosis.

---
## ⚙️ Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

##🧪 How to Run
Use the test_model.py script with command-line arguments:
python test_model.py \
  --model_dir "path/to/models" \
  --left_img_path "path/to/left.jpg" \
  --right_img_path "path/to/right.jpg" \
  --age 60 \
  --gender Male

##Output
![image](https://github.com/user-attachments/assets/667e5c5f-a2b5-4d95-ba4f-9a0bc4930d9f)
