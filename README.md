# Face Recognition Streamlit App

A real-time face recognition application that detects facial landmarks including eyes, nose, mouth, and ear regions using MediaPipe and Streamlit.

## Features

- 🎯 Detects 478 facial landmarks
- 👁️ Identifies eyes with iris tracking
- 👃 Locates nose and bridge
- 👄 Outlines lips and mouth
- 👂 Approximates ear regions
- 📸 Supports image upload and webcam input
- 🎨 Color-coded feature visualization
- ⚙️ Adjustable detection confidence

## Local Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/face-recognition-app.git
cd face-recognition-app
```

2. **Create virtual environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
streamlit run app.py
```

5. **Open your browser:**
Navigate to `http://localhost:8501`

## Project Structure

```
face-recognition-app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore file
└── packages.txt          # System dependencies (for deployment)
```

## GitHub Deployment

### Deploy to Streamlit Cloud

1. **Push code to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/face-recognition-app.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `yourusername/face-recognition-app`
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Optional - Create packages.txt** (for system dependencies):
```txt
libgl1-mesa-glx
libglib2.0-0
```

### Deploy to Heroku

1. **Create Procfile:**
```
web: sh setup.sh && streamlit run app.py
```

2. **Create setup.sh:**
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. **Deploy:**
```bash
heroku create your-app-name
git push heroku main
```

## Usage

### Upload Image Mode
1. Select "Upload Image"
2. Click "Browse files" and choose an image
3. View detected facial features on the right panel

### Webcam Mode
1. Select "Use Webcam"
2. Click "Start" to capture
3. Allow camera permissions
4. Take a photo to analyze

### Adjust Settings
- Use sidebar sliders to adjust detection/tracking confidence
- Toggle "Show All Landmarks" for full mesh visualization
- Toggle "Show Face Contours" for face outline

## Color Legend

- 🟢 **Green** - Eyes
- 🔵 **Blue** - Nose
- 🔴 **Red** - Lips
- 🟡 **Yellow** - Ear Regions

## Technologies Used

- **Streamlit** - Web application framework
- **MediaPipe** - Face mesh detection (478 landmarks)
- **OpenCV** - Image processing
- **Python 3.8+** - Programming language

## Troubleshooting

### Issue: "No module named 'cv2'"
**Solution:** Install opencv-python-headless
```bash
pip install opencv-python-headless
```

### Issue: Camera not working
**Solution:** 
- Check browser permissions
- Ensure HTTPS connection (required for webcam)
- Try using uploaded images instead

### Issue: Slow performance
**Solution:**
- Reduce detection confidence
- Disable "Show All Landmarks"
- Use smaller images

## Requirements

- Python 3.8 or higher
- Webcam (optional, for camera input)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- MediaPipe by Google for face mesh detection
- Streamlit for the amazing framework
- OpenCV community

## Contact

For issues and questions, please open an issue on GitHub.

---

**Note:** This application processes all images locally in your browser. No images are stored or sent to external servers.