# Plant Leaf Disease Detection System

This project is a Plant Leaf Disease Detection System that uses a Convolutional Neural Network (CNN) to identify diseases in plant leaves. The system features a user-friendly interface built with Streamlit, allowing users to upload images of leaves and receive disease predictions in real-time.

## Features

- **Leaf Disease Detection:** Utilizes a Convolutional Neural Network (CNN) model to classify leaf diseases based on uploaded images.
- **User-Friendly Interface:** Built with Streamlit to provide an intuitive GUI for uploading leaf images and viewing predictions.
- **Real-Time Predictions:** Get instant disease predictions and insights directly through the Streamlit interface.

## Project Structure

- `trained_plant_disease_model.h5`: The trained CNN model for predicting leaf diseases.
- `leaf_detection.py`: The main Streamlit application that serves as the GUI.
- `requirements.txt`: List of required Python packages for running the project.
- `README.md`: This file, containing an overview of the project and instructions for setup.

## Dataset

The dataset used for training and testing the model was obtained from [Kaggle]( https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset). It includes images of various plant leaves with labels indicating the type of disease.

## Installation

1. Clone the repository:
   ```bash
   https://github.com/yash-jagtap1706/Plant_leaf_disease_detection_system.git ```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
 ```

3. Run the Streamlit app:
```bash
streamlit run leaf_detection.py 
```


## License

This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions or issues, please contact yvj0007@gmail.com 

## Demo

https://leafdetectionpy-k34rsdeyl5kfrdkoatpbfx.streamlit.app/


