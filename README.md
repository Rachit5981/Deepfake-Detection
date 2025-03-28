# Deepfake Detection

This repository contains a **Deepfake Detection** tool using machine learning. The tool is built with **Streamlit** for the user interface and uses **RandomForestClassifier** for classification.

## Features
- Trains a **Random Forest Classifier** on mock features.
- Allows users to test predictions on sample inputs.
- Provides a UI for easy interaction.

## Installation & Setup
### 1. Clone the repository
```bash
git clone https://github.com/your-username/deepfake-detection.git
cd deepfake-detection
```

### 2. Install dependencies
Make sure you have **Python 3.7+** installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```

## Usage
- The application will launch in your browser.
- You can adjust feature values using sliders and make predictions.
- The model outputs whether the sample is a **Real** image or a **Deepfake**, along with a probability score.

## File Structure
```
.
├── app.py              # Main Streamlit app
├── requirements.txt    # Required Python packages
├── README.md           # Project documentation
```

## Contributing
Feel free to contribute by:
- Improving the feature set.
- Enhancing the model with deep learning techniques (e.g., CNNs).
- Adding more interactive visualizations.

## License
This project is licensed under the MIT License.

