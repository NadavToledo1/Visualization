# Speed Dating Data Analysis

This repository contains a Streamlit application for analyzing a speed dating dataset. The app includes various filters and visualizations to explore the data and understand patterns and trends in speed dating.

## Features

- **Data Cleaning and Imputation**: Handles missing values in the dataset using different strategies such as mode, mean, and median imputation.
- **Interactive Filters**: Allows users to filter the data based on gender, age, attractiveness rating, income, and goals.
- **Visualizations**:
  - Bubble chart showing the average success rate by attractiveness score.
  - Bar charts displaying decision outcomes by goals in absolute numbers and relative proportions.
  - Line charts depicting success rates by age and gender.
  - Feature importance bar plot for predicting date success using a Random Forest classifier.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/NadavToledo1/Visualization.git
    cd Visualization
    ```

2. **Install the required packages**:
    Ensure you have Python 3.7 or higher installed. You can install the required packages using `pip`:
    ```sh
    pip install -r requirements.txt
    ```

3. **Add your data file**:
    Place your `speed_data_data.csv` file in the root directory of the repository.

## Running the App

To run the Streamlit app locally, use the following command:
```sh
streamlit run app.py
```

## Deploying on Streamlit Cloud

1. **Push the code to GitHub**: Ensure all your code and the `speed_data_data.csv` file are pushed to your GitHub repository.
2. **Deploy on Streamlit Cloud**:
    - Go to [Streamlit Cloud](https://share.streamlit.io).
    - Click on "New app".
    - Connect your GitHub account and select the repository.
    - Choose the branch and the `app.py` file.
    - Click "Deploy".

## File Structure

```
Visualization/
│
├── app.py                # Main Streamlit app script
├── requirements.txt      # List of dependencies
├── speed_data_data.csv   # Dataset file
└── README.md             # This README file
```

## Dependencies

The project uses the following Python libraries:
- streamlit
- pandas
- numpy
- matplotlib
- plotly
- seaborn
- scikit-learn

These are listed in the `requirements.txt` file.

## Usage

### Data Cleaning and Imputation

- The dataset is cleaned by dropping rows with missing values in critical columns (`attr`, `dec`, `age`).
- Categorical columns such as `career` and `goal` are imputed using the mode within groups.
- Numeric columns are imputed using the mean or median within groups.

### Interactive Filters

Users can filter the data by:
- Gender
- Age range
- Attractiveness rating range
- Income range
- Goals

### Visualizations

1. **Bubble Chart**:
   - Shows the average success rate by attractiveness score.
   - The size of the bubbles represents the count of records for each score.

2. **Bar Charts**:
   - Display decision outcomes by goals in both absolute numbers and relative proportions.
   - Users can sort the data alphabetically or by the total number of decisions.

3. **Line Charts**:
   - Depict success rates by age and gender.

4. **Feature Importance Bar Plot**:
   - Visualizes the importance of various features in predicting date success using a Random Forest classifier.

## Contributing

Contributions are welcome! If you have any ideas for improvements or new features, please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.

## Acknowledgements

- The dataset used in this project is publicly available and sourced from [Speed Dating Experiment Data](https://www.kaggle.com/annavictoria/speed-dating-experiment).
