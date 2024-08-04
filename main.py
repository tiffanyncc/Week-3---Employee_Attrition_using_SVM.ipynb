import os

# Set the environment variable to avoid the memory leak warning
os.environ["OMP_NUM_THREADS"] = "1"

import logging
import matplotlib.pyplot as plt
from src.data.make_dataset import load_data
from src.features.build_features import preprocess_data, prepare_data
from src.models.logistic_regression import train_logistic_regression
from src.models.svm import train_svm
from src.models.predict_model import predict_model, evaluate_model
from src.visualization.visualize import plot_distributions, plot_categorical_distribution, plot_categorical_attrition, plot_correlation_heatmap, plot_metrics_score, plot_bivariate_attrition
from sklearn.model_selection import train_test_split

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def main():
    try:
        df = load_data('src/data/HR_Employee_Attrition.xlsx')
        logging.info('Data loaded successfully.')
    except Exception as e:
        logging.error(f'Error loading data: {e}')
        return

    try:
        num_cols = ['DailyRate', 'Age', 'DistanceFromHome', 'MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike', 'TotalWorkingYears',
                    'YearsAtCompany', 'NumCompaniesWorked', 'HourlyRate', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'TrainingTimesLastYear']
        plot_distributions(df, num_cols)
        logging.info('Distributions plot displayed and saved.')
    except Exception as e:
        logging.error(f'Error displaying distributions plot: {e}')

    try:
        cat_cols = ['Attrition', 'OverTime', 'BusinessTravel', 'Department', 'Education', 'EducationField', 'JobSatisfaction',
                    'EnvironmentSatisfaction', 'WorkLifeBalance', 'StockOptionLevel', 'Gender', 'PerformanceRating', 'JobInvolvement',
                    'JobLevel', 'JobRole', 'MaritalStatus', 'RelationshipSatisfaction']
        plot_categorical_distribution(df, cat_cols)
        plot_categorical_attrition(df, cat_cols)
        plot_bivariate_attrition(df, cat_cols)
        logging.info('Categorical and bivariate plots displayed and saved.')
    except Exception as e:
        logging.error(f'Error displaying categorical and bivariate plots: {e}')

    try:
        plot_correlation_heatmap(df, num_cols)
        logging.info('Correlation heatmap displayed and saved.')
    except Exception as e:
        logging.error(f'Error displaying correlation heatmap: {e}')

    try:
        df = preprocess_data(df)
        logging.info('Data preprocessed successfully.')
        logging.info(f'Columns after preprocessing: {df.columns.tolist()}')
    except Exception as e:
        logging.error(f'Error preprocessing data: {e}')
        return

    try:
        # Split and scale the data
        X_scaled, Y = prepare_data(df)
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=1, stratify=Y)
        logging.info('Data split and scaled successfully.')
    except Exception as e:
        logging.error(f'Error splitting and scaling data: {e}')
        return
    
    try:
        # Train logistic regression model
        lg_model = train_logistic_regression(x_train, y_train)
        y_pred_train = predict_model(lg_model, x_train)
        cm_train, acc_train = evaluate_model(y_train, y_pred_train)
        plot_metrics_score(cm_train)
        logging.info(f'Logistic Regression Train Accuracy: {acc_train}')
        y_pred_test = predict_model(lg_model, x_test)
        cm_test, acc_test = evaluate_model(y_test, y_pred_test)
        plot_metrics_score(cm_test)
        logging.info(f'Logistic Regression Test Accuracy: {acc_test}')
    except Exception as e:
        logging.error(f'Error training or evaluating logistic regression model: {e}')

    try:
        # Train SVM model with different kernels
        for kernel in ['linear', 'rbf', 'poly']:
            svm_model = train_svm(x_train, y_train, kernel=kernel)
            y_pred_train_svm = predict_model(svm_model, x_train)
            cm_train_svm, acc_train_svm = evaluate_model(y_train, y_pred_train_svm)
            plot_metrics_score(cm_train_svm)
            logging.info(f'SVM ({kernel} kernel) Train Accuracy: {acc_train_svm}')
            y_pred_test_svm = predict_model(svm_model, x_test)
            cm_test_svm, acc_test_svm = evaluate_model(y_test, y_pred_test_svm)
            plot_metrics_score(cm_test_svm)
            logging.info(f'SVM ({kernel} kernel) Test Accuracy: {acc_test_svm}')
    except Exception as e:
        logging.error(f'Error training or evaluating SVM model: {e}')

if __name__ == '__main__':
    main()
