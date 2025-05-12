# Protein Structure Prediction

## Overview
The Protein Structure Prediction project leverages the MC-NEST algorithm integrated with OpenAI's GPT-4o model to predict and refine protein structures based on amino acid sequences. The project includes advanced hypothesis generation and evaluation mechanisms, enabling detailed insights into protein sequences. A user-friendly Streamlit interface is provided for interaction, and the predicted structures are visualized using the py3Dmol library.

## Project Structure
```
protein-structure-prediction
├── src
│   ├── app.py                # Main application script for Streamlit interface
│   ├── predict_proteins.py    # Implementation of the MC-NEST algorithm with GPT-4o integration
│   ├── utils
│   │   ├── __init__.py       # Initialization file for utils module
│   │   └── st_utils.py       # Utility functions for Streamlit integration
├── requirements.txt           # List of required Python packages
├── README.md                  # Project documentation
└── .gitignore                 # Files and directories to ignore by Git
```

## Step-by-Step Guide to Set Up the Local Environment

1. **Clone the Repository**: 
   Clone the project repository from GitHub to your local machine using the following command:
   ```
   git clone <repository-url>
   ```

2. **Navigate to the Project Directory**: 
   Change into the project directory:
   ```
   cd protein-structure-prediction
   ```

3. **Set Up a Virtual Environment**: 
   It is recommended to create a virtual environment to manage dependencies. You can create one using:
   ```
   python -m venv venv
   ```

4. **Activate the Virtual Environment**: 
   Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

5. **Install Dependencies**: 
   Install the required packages listed in `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```

6. **Run the Application**: 
   Start the Streamlit application by running:
   ```
   streamlit run src/app.py
   ```

7. **Access the Application**: 
   Open your web browser and go to the URL provided in the terminal (usually `http://localhost:8501`) to access the protein structure prediction interface.

8. **Input Protein Sequence**: 
   Enter a protein sequence in the provided text area and click the "Predict & Visualize" button to see the predicted structure.

9. **Explore the Results**: 
   The application will display the predicted protein structure and additional information based on the input sequence.

## Additional Notes
- Ensure you have Python 3.7 or higher installed on your machine.
- If you encounter any issues, refer to the `README.md` for troubleshooting tips and further documentation.