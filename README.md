# SpaceX Falcon 9 Landing Prediction

## Project Overview
This project aims to predict the landing outcome of **SpaceX Falcon 9 rockets** using historical launch data. By leveraging **machine learning techniques**, we analyze various factors affecting the rocket's landing success and build predictive models to enhance decision-making in rocket reusability.

## Dataset
The dataset is obtained from SpaceX launch records and includes features such as:
- **Flight Number**: Unique identifier for each launch
- **Launch Site**: Location from where the rocket was launched
- **Payload Mass**: Weight of the payload in kilograms
- **Orbit Type**: The orbit in which the payload is placed
- **Booster Version**: Details of the Falcon 9 booster used
- **Weather Conditions**: Environmental factors at the time of launch
- **Landing Outcome**: Whether the rocket successfully landed or not (binary classification target)

## Tools & Technologies Used
- **Python**
- **Pandas, NumPy** for data preprocessing and analysis
- **Matplotlib, Seaborn** for data visualization
- **Scikit-Learn** for machine learning models
- **Jupyter Notebook** for development and documentation

## Project Workflow
1. **Data Collection**: Import and clean the SpaceX launch dataset
2. **Exploratory Data Analysis (EDA)**: Identify patterns and correlations
3. **Feature Engineering**: Select and transform relevant features
4. **Model Selection & Training**: Train classification models like Logistic Regression, Decision Trees, and Random Forest
5. **Model Evaluation**: Assess accuracy, precision, recall, and other performance metrics
6. **Conclusion & Future Scope**: Discuss insights and potential improvements

## How to Run the Project
1. Clone the repository:
   ```sh
   git clone https://github.com/Saubhagya49/SpaceX-Falcon-9-Landing-Prediction.git
   ```
2. Navigate to the project directory:
   ```sh
   cd SpaceX-Falcon-9-Landing-Prediction
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Open and run the Jupyter Notebook:
   ```sh
   jupyter notebook
   ```
5. Follow the notebook instructions to execute data analysis and model training.

## Results & Insights
- The model achieved **94.4% accuracy** 
- Key factors influencing landing success include **payload mass, launch site, and orbit type**
- Linear Regression performed best among the tested models. 

## Future Improvements
- Incorporate real-time launch data for continuous learning
- Use deep learning techniques for better prediction accuracy
- Optimize hyperparameters for model improvement

## Contributors
- **Saubhagya49** ([GitHub](https://github.com/Saubhagya49))

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



