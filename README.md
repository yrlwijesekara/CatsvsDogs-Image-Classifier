# 🐾 Cats vs Dogs Image Classifier

A deep learning project that uses a Convolutional Neural Network (CNN) to classify images of cats and dogs with high accuracy. The project includes both a command-line interface and an interactive web application built with Streamlit.

## 🎯 Project Overview

This project implements a binary image classification system that can distinguish between cats and dogs in uploaded images. The model is trained using TensorFlow/Keras and deployed with an intuitive web interface for easy interaction.

### Key Features

- **High-accuracy CNN model** trained for binary classification
- **Interactive web application** built with Streamlit
- **Command-line interface** for batch processing
- **Real-time predictions** with confidence scores
- **Professional UI** with image preview and results visualization
- **Multiple image format support** (JPG, JPEG, PNG, BMP, GIF)

## 🏗️ Project Structure

```
CatsvsDogs-Image-Classifier/
├── best_model_cats_vs_dogs.keras    # Trained CNN model (Keras format)
├── deployment_info.joblib           # Deployment configuration
├── preprocessing_info.joblib        # Model preprocessing parameters
├── training_history.joblib          # Training metrics and history
├── predict.py                       # Command-line prediction script
├── streamlit_app.py                 # Web application (Streamlit)
├── utils.py                         # Utility functions
├── requirements.txt                 # Python dependencies
├── venv/                           # Virtual environment
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yrlwijesekara/CatsvsDogs-Image-Classifier.git
   cd CatsvsDogs-Image-Classifier
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage

### Web Application (Recommended)

Launch the interactive Streamlit web application:

```bash
streamlit run streamlit_app.py
```

Then open your browser and navigate to `http://localhost:8501`

**Features:**
- Drag & drop image upload
- Real-time prediction with confidence scores
- Beautiful visualization with progress bars
- Mobile-friendly responsive design

### Command Line Interface

For programmatic use or batch processing:

```bash
python predict.py path/to/your/image.jpg
```

**Example:**
```bash
python predict.py test_images/cat.jpg
# Output: 
# Predicted class: cats
# Confidence: 0.87
```

## 🤖 Model Specifications

- **Architecture:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow/Keras
- **Input Size:** 150x150 pixels
- **Classes:** 2 (cats, dogs)
- **Image Preprocessing:** RGB conversion, resizing, normalization (1/255)
- **Output:** Binary classification with confidence score

## 📋 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.50.0 | Web application framework |
| tensorflow | 2.20.0 | Deep learning model |
| joblib | 1.5.2 | Model serialization |
| numpy | 2.2.6 | Numerical computations |
| Pillow | 11.3.0 | Image processing |
| scikit-learn | 1.6.1 | Machine learning utilities |
| matplotlib | 3.10.7 | Plotting and visualization |
| seaborn | 0.13.2 | Statistical data visualization |

## 🎨 Web Application Features

### Main Interface
- **Clean, intuitive design** with cat 🐱 and dog 🐶 emojis
- **File upload widget** supporting multiple formats
- **Side-by-side layout** showing original image and prediction
- **Responsive design** that works on desktop and mobile

### Prediction Display
- **Confidence visualization** with progress bars
- **Color-coded results** (blue for cats, orange for dogs)
- **Confidence levels** with helpful interpretation:
  - 🎯 High confidence (>80%)
  - ⚠️ Moderate confidence (60-80%)
  - 🤔 Low confidence (<60%)

### Sidebar Information
- Model specifications and details
- Usage tips for better results
- Technical information about the architecture

## 🛠️ Development

### Model Information
The preprocessing pipeline includes:
- Image resizing to 150×150 pixels
- RGB color space conversion
- Pixel value normalization (0-1 scale)
- Batch dimension expansion for model input

### Key Functions

**`predict.py`:**
- `load_deployed_model()` - Loads the trained model and preprocessing info
- `preprocess_image()` - Prepares images for model input
- `predict_image()` - Makes predictions and calculates confidence

**`streamlit_app.py`:**
- `load_model_and_preprocessing()` - Cached model loading for efficiency
- `main()` - Main application interface and logic

## 🚀 Deployment

### Local Deployment
1. Follow the installation steps above
2. Run `streamlit run streamlit_app.py`
3. Access at `http://localhost:8501`

### Cloud Deployment Options
- **Streamlit Cloud:** Push to GitHub and deploy directly
- **Heroku:** Use the provided requirements.txt
- **AWS/GCP/Azure:** Deploy using container services

## 📊 Performance Tips

### For Better Predictions:
- Use clear, high-quality images
- Ensure the animal is the main subject
- Avoid images with multiple animals
- Good lighting and contrast help accuracy

### System Requirements:
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 500MB for model and dependencies
- **CPU:** Multi-core processor recommended for faster inference

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Yasiru Lakshitha Wijesekara**
- GitHub: [@yrlwijesekara](https://github.com/yrlwijesekara)

## 🙏 Acknowledgments

- TensorFlow team for the excellent deep learning framework
- Streamlit team for the intuitive web app framework
- The open-source community for various tools and libraries

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yrlwijesekara/CatsvsDogs-Image-Classifier/issues) page
2. Create a new issue with detailed information
3. Include error messages and system information

---

**Made with ❤️ and 🐾 for animal lovers and AI enthusiasts!**