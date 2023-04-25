import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('fico_national_avg.csv')
df = pd.DataFrame(data)
data2 = pd.read_csv('gsc_sample.csv')
df2 = pd.DataFrame(data2)
print("# of columns in gsc_sample ", df2.size)

# print(df)

def national_credit_averages():
    # Creates graph of national average vs Year

    fig = sns.lineplot(x='year', y="score", data=df)
    plt.xticks(df.year.values.tolist(), rotation=75)
    return plt.gcf()

# national_credit_averages()

def test_function():

    # load data from CSV file

    # select relevant features
    features = ['SeriousDlqin2yrs', 'age', 'NumberOfTime3059DaysPastDueNotWorse', 'MonthlyIncome',
                'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
                'NumberOfTime6089DaysPastDueNotWorse', 'NumberOfDependents', 'RevolvingUtilizationOfUnsecuredLines', 'DebtRatio','credit_score']
    X = data2[features].values
    y = data2['credit_score'].values

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # scale features to have zero mean and unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # make predictions on test set
    y_pred = model.predict(X_test)

    # evaluate model accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Model accuracy:", accuracy)

def test_function_2():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    X_train, X_test, y_train, y_test = train_test_split(df2.drop('credit_score', axis=1), df2['credit_score'], test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the credit scores on the testing set
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = r2_score(y_test, y_pred)

    print("Model accuracy:", accuracy)

# test_function()
test_function_2()


def get_percentile(score, df2):
    # Calculates percentile of user based on our data
    
    percentile = (df2['credit_score'] < score).sum() / len(df2) * 100
    return percentile

# Example: Calculate the percentile of a score
score = 720
percentile = get_percentile(score, df2)
print(f"The {score} credit score is in the {percentile:.2f} percentile.")

def make_graph_age_score(user_score):
    plt.figure(figsize=(10, 6))
    sns.histplot(df2['credit_score'], kde=False, bins=50)

    # Add a red line at the x-axis for the specific credit score
    plt.axvline(x=user_score, color='red', linestyle='--')

    # Annotate the red line with the percentile information
    plt.annotate(f"{user_score} ({percentile:.2f} percentile)", xy=(score, 0), xycoords=('data', 'axes fraction'), 
                xytext=(5, 10), textcoords='offset points', fontsize=12, color='red')

    # Set title and labels for the plot
    plt.title('Distribution of Credit Scores')
    plt.xlabel('Credit Score')
    plt.ylabel('Age')

    # Show the plot
    plt.show()
    return plt.gcf()

# make_graph_age_score()


def credit_score_education(df2):
    L = []
    # Calculate average credit score for each age group
    age_groups = pd.cut(df2['age'], [18, 30, 40, 50, 60, 70])
    avg_score = df2.groupby(age_groups)['credit_score'].mean()
    print('Average Credit Score by Age Group:')
    print(avg_score)
    L.append(avg_score)
    
    # Calculate debt-to-income ratio for each income group
    income_groups = pd.cut(df2['MonthlyIncome'], [0, 5000, 10000, 15000, 20000, 25000, 30000])
    dti = df2['DebtRatio'] / (df2['MonthlyIncome'] / 12)
    dti_groups = pd.cut(dti, [0, 0.2, 0.4, 0.6, 0.8, 1])
    dti_table = pd.pivot_table(df2, index=income_groups, columns=dti_groups, values='credit_score', aggfunc='mean')
    print('\nDebt-to-Income Ratio by Income Group:')
    print(dti_table)
    L.append(dti_table)
    
    # Calculate credit utilization ratio for each credit line group
    credit_groups = pd.cut(df2['NumberOfOpenCreditLinesAndLoans'], [0, 5, 10, 15, 20, 25, 30])
    cur = df2['RevolvingUtilizationOfUnsecuredLines']
    cur_groups = pd.cut(cur, [0, 0.2, 0.4, 0.6, 0.8, 1])
    cur_table = pd.pivot_table(df2, index=credit_groups, columns=cur_groups, values='credit_score', aggfunc='mean')
    print('\nCredit Utilization Ratio by Credit Line Group:')
    print(cur_table)
    L.append(cur_table)
    return L
    
# Apply the function to the credit report data and display the results
credit_score_education(df2)

def credit_score_goal_setting(df2, goal):
    # Calculate current credit score and factors affecting it
    score = df2['credit_score'].mean()
    factors = ['NumberOfTimes90DaysLate', 'NumberOfTime6089DaysPastDueNotWorse', 'NumberOfTime3059DaysPastDueNotWorse', 'RevolvingUtilizationOfUnsecuredLines']
    factor_counts = df2[factors].sum()
    print('Current Credit Score:', score)
    print('Factors Affecting Credit Score:')
    for factor, count in factor_counts.items():
        print(f"{factor}: {count} occurrences")
        
    # Calculate target credit score and recommended actions to achieve it
    target = goal
    actions = []
    if score < target:
        actions.append('Reduce credit utilization ratio')
        actions.append('Make payments on time')
        if factor_counts['NumberOfTimes90DaysLate'] > 0:
            actions.append('Resolve overdue accounts')
        if factor_counts['NumberOfTime6089DaysPastDueNotWorse'] > 0:
            actions.append('Resolve accounts in default')
        if factor_counts['NumberOfTime3059DaysPastDueNotWorse'] > 0:
            actions.append('Prioritize payments on accounts with highest delinquency')
    else:
        actions.append('Maintain current credit utilization ratio')
        actions.append('Continue making payments on time')
        
    # Display credit score goal, current score, and recommended actions
    print('\nCredit Score Goal:', target)
    print('Recommended Actions:')
    for action in actions:
        print('- ' + action)
        
# Apply the function to the credit report data and display the results
credit_score_goal_setting(df2, 750)