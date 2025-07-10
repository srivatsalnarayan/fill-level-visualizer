# 🛢️ Fill Level Visualizer — Oil Tank Shadow Estimation via Satellite 🛰️
> Estimate oil tank fill levels using YOLOv8 object detection and shadow geometry. Fully Dockerized and ready to demo.

![Docker](https://img.shields.io/badge/deploy-docker-blue?logo=docker)
![Streamlit](https://img.shields.io/badge/built%20with-streamlit-orange?logo=streamlit)
![Status](https://img.shields.io/badge/status-production-green)

---

## 📸 What It Does

This app uses satellite imagery to:
- Detect oil tanks (via a custom-trained YOLOv8 model)
- Segment them from large images using tile-based inference
- Estimate tank fill levels based on **shadow proportion**
- Visualize everything in an interactive **Streamlit UI**
- Export results as a CSV 

---

##  Setup Options

###  OPTION 1: Run **locally with Docker**
> Recommended for full reproducibility

```bash
git clone https://github.com/srivatsal/fill-level-visualizer.git
cd fill-level-visualizer

# Build Docker image
docker build -t fill-visualizer .

# Run the container
docker run -p 8501:8501 fill-visualizer
```

🔗 Open [http://localhost:8501](http://localhost:8501)

---

###  OPTION 2: Use prebuilt Docker image

```bash
docker pull srivatsal/fill-visualizer
docker run -p 8501:8501 srivatsal/fill-visualizer
```

---

###  OPTION 3: Run directly with Streamlit (no Docker)

```bash
git clone https://github.com/srivatsal/fill-level-visualizer.git
cd fill-level-visualizer

# (Optional) Activate your virtual environment
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

##  Tech Stack

| Layer       | Tools                                 |
|-------------|----------------------------------------|
| Detection   | YOLOv8 (Ultralytics)                  |
| Interface   | Streamlit                             |
| Packaging   | Docker                                |
| Image Tiling| OpenCV + Custom Stitch Logic          |
| Fill Estimation | Shadow heuristics + geometry      |

---

##  Project Structure

```
 fill-level-visualizer/
├── app.py               ← Streamlit frontend
├── pipeline.py          ← Core logic (tiling, detect, merge, fill)
├── Dockerfile           ← For container builds
├── requirements.txt     ← Python dependencies
├── weights/best.pt      ← YOLOv8 trained weights
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Notes

- Trained on oil storage tank patch dataset from Kaggle
- Handles large image inference via tile-wise detection
- Fill level is computed from **shadow-to-tank ratio**

---


##  Author

**Srivatsal Narayan**  
Machine Learning Engineer | Alt-data enthusiast  | IOS developer | 

---

## ⭐ Star this repo if you like it!
