# Employee Attrition Prediction

## Project Overview

An MNC has thousands of employees spread out across the globe. The company believes in hiring the best talent available and retaining them for as long as possible. A huge amount of resources is spent on retaining existing employees through various initiatives. The Head of People Operations wants to bring down the cost of retaining employees. For this, he proposes limiting the incentives to only those employees who are at risk of attrition. As a recently hired Data Scientist, you have been asked to identify patterns in characteristics of employees who leave the organization. Also, you have to use this information to predict if an employee is at risk of attrition. This information will be used to target them with incentives.

## Objectives

- Identify the different factors that drive attrition
- Build a model to predict if an employee will attrite or not

## Dataset

The data contains demographic details, work-related metrics, and attrition flag. The main features include:

- **EmployeeNumber** - Employee Identifier
- **Attrition** - Did the employee attrite?
- **Age** - Age of the employee
- **BusinessTravel** - Travel commitments for the job
- **DailyRate** - Daily rate
- **Department** - Employee Department
- **DistanceFromHome** - Distance from work to home (in km)
- **Education** - Level of education
- **EducationField** - Field of education
- **EnvironmentSatisfaction** - Satisfaction with the work environment
- **Gender** - Employee's gender
- **HourlyRate** - Hourly rate
- **JobInvolvement** - Level of job involvement
- **JobLevel** - Job level
- **JobRole** - Job roles
- **JobSatisfaction** - Satisfaction with the job
- **MaritalStatus** - Marital status
- **MonthlyIncome** - Monthly income
- **MonthlyRate** - Monthly rate
- **NumCompaniesWorked** - Number of companies worked at
- **Over18** - Over 18 years of age?
- **OverTime** - Overtime?
- **PercentSalaryHike** - Percentage increase in salary last year
- **PerformanceRating** - Performance rating
- **RelationshipSatisfaction** - Satisfaction with relationships at work
- **StandardHours** - Standard hours
- **StockOptionLevel** - Stock option level
- **TotalWorkingYears** - Total years worked
- **TrainingTimesLastYear** - Number of training attended last year
- **WorkLifeBalance** - Work-life balance
- **YearsAtCompany** - Years at company
- **YearsInCurrentRole** - Years in the current role
- **YearsSinceLastPromotion** - Years since the last promotion
- **YearsWithCurrManager** - Years with the current manager

## Instructions

1. **Install Dependencies**: Use the `requirements.txt` file to install the necessary dependencies.
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the Project**: Execute the `main.py` file to run the entire pipeline.
    ```bash
    python main.py
    ```

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- openpyxl
