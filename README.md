# Optimal EV Retrofitting Age Predictor

## Project Overview

This project delivers a data-driven solution to identify the optimal time for converting traditional internal combustion engine (ICE) vehicles into Electric Vehicles (EVs) through retrofitting. By analyzing real-world driving data and customizable economic parameters, the application predicts the "break-even age" – the point at which the financial and environmental savings from EV ownership outweigh the initial retrofitting investment.

Our goal is to equip individuals and fleet managers with precise, data-backed insights, transforming guesswork into informed decisions for sustainable vehicle conversion.

-----

## Features

  * **Intelligent Data Preprocessing:** Automatically loads, cleans, and transforms raw driving telematics data, calculating essential metrics like distance and acceleration.
  * **Dynamic Retrofitting Metrics:** Computes key financial (e.g., annual savings, break-even age) and environmental indicators based on user-configurable parameters.
  * **Machine Learning Prediction:** Employs **Random Forest** and **Gradient Boosting Regressors** to accurately predict the optimal retrofitting age from driving patterns.
  * **Interactive Streamlit Dashboard:** Provides a user-friendly web interface enabling dynamic adjustment of **7 key economic parameters** (e.g., battery cost, fuel prices, CO2 factors) and instant visualization of predictions.
  * **Optimal Policy Identification:** Identifies specific driving patterns or timeframes that present the most financially advantageous opportunities for EV retrofitting.
  * **Performance Visualization:** Displays model performance metrics (e.g., R² score), **feature importance**, and correlations between driving behaviors and optimal retrofitting age.

-----

## Technical Stack

  * **Python:** Core programming language.
  * **Pandas & NumPy:** For efficient data manipulation and numerical operations.
  * **Scikit-learn:** For machine learning model development (Random Forest, Gradient Boosting, data splitting, scaling, and evaluation).
  * **Streamlit:** For building the interactive web application.
  * **Matplotlib:** For creating compelling data visualizations.

-----

## Installation & Setup

To get this project running on your local machine, follow these steps:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/YourGitHubUsername/optimal-ev-retrofitting-predictor.git
    cd optimal-ev-retrofitting-predictor
    ```

    *(**Note:** Remember to replace `https://github.com/YourGitHubUsername/optimal-ev-retrofitting-predictor.git` with the actual URL of your GitHub repository.)*

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**

    ```bash
    pip install pandas numpy scikit-learn matplotlib streamlit
    ```

4.  **Data Requirement:**

      * This project expects a `driving_data.csv` file. Currently, the `load_data` function points to a local path (`C:\Users\Kritin\Downloads\driving_data.csv`).
      * **To run the project:** Place your `driving_data.csv` file in the same directory as your main Python script, and update the `file_path` in the `load_data` function to `driving_data.csv` (or the correct relative path).
      * The CSV should contain relevant telematics data columns such as 'UTC Time', 'Speed', 'Longitudinal Acc' (or 'Acceleration'), etc.

-----

## Usage

Once you've completed the setup, run the Streamlit application from your terminal:

```bash
streamlit run your_main_script_name.py
```

*(**Note:** Replace `your_main_script_name.py` with the actual name of your Python file containing the `main()` function, e.g., `app.py` or `main.py`).*

The application will launch in your default web browser (typically at `http://localhost:8501`). You can then interact with the sliders in the sidebar to adjust various parameters and observe the real-time predictions and visualizations.

-----

## Future Improvements

  * **User Data Upload:** Implement a feature allowing users to upload their own `driving_data.csv` files directly through the Streamlit interface for personalized analysis.
  * **API Integrations:** Integrate with external APIs for real-time fuel prices, electricity rates, or regional CO2 emission factors to enhance prediction accuracy.
  * **Advanced ML Models:** Explore and implement more sophisticated time-series models or deep learning approaches for even greater predictive power.
  * **Comparative Analysis:** Develop functionality for users to compare different retrofitting scenarios or vehicle types side-by-side.
  * **Detailed ROI Breakdown:** Provide more granular financial reports and interactive visualizations of payback periods and long-term Return on Investment (ROI).

-----

## Contact

For any questions or collaborations, feel free to reach out:

  * **[Your Name/GitHub Username]**
  * **[Your Email Address (Optional)]**
  * **[Link to your LinkedIn Profile (Optional)]**

-----
