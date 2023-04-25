import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv_reader as csv

st.set_page_config(page_title='Credit Card Explorer',page_icon="creditscore.png")
#######################################################################################
df2 = csv.df2
def get_percentile(score, df2):
    # Calculates percentile of user based on our data
    
    percentile = (df2['credit_score'] < score).sum() / len(df2) * 100
    return percentile



def credit_score_goal_setting(df2, goal,score):
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
    return actions
        
#############################################################################
#title of the app
#st.image(Image.open("creditscore.png"),width = 150)
st.title("Credit Card Explorer")
st.title("")

Sex = ""
show = False
button_disable = True
with st.sidebar:
    selected = option_menu(
        menu_title = None,
        options = ["Input Analysis", "Data Analysis"]
        )
    


if selected == "Data Analysis":
        Data_container = st.container()
        with Data_container:
            L = csv.credit_score_education(csv.df2)
            print('Average Credit Score by Age Group:')
            print('\nDebt-to-Income Ratio by Income Group:')
            print('\nCredit Utilization Ratio by Credit Line Group:')
            expander3 = st.expander('Average Credit Score by Age Group:')
            expander4 = st.expander('Debt-to-Income Ratio by Income Group:')
            expander5 = st.expander('Credit Utilization Ratio by Credit Line Group:')
            expander = st.expander("fico_national_avg.csv")
            expander6 = st.expander("National Credit Average")
            expander2 = st.expander("gsc_sample.csv")
            with expander:
                st.write(csv.df)
            with expander2:
                st.write(csv.df2)
                st.write("# of columns in gsc_sample: ", csv.df2.size)
            with expander3:
                st.write(pd.DataFrame(L[0]))
                print(L[0])
                print(pd.DataFrame(L[0]))
                #print(type(L[0]))
            with expander4:
                st.write(L[1])
            with expander5:
                st.write(L[2])
            with expander6:
                st.pyplot(csv.national_credit_averages())





elif selected == "Input Analysis":
        column1, column2 = st.columns(2)
        with column1:
            Score = st.number_input("Credit Score 👇",step = 1)
        with column1:
            Age = st.number_input("Age 👇",step = 1)
        with column2:
            Sex = st.selectbox("Sex 👇",("Male", "Female"),disabled=False)
        with column2:
            Goal = st.number_input("Credit Goal👇",step = 1)
        with column2:
            if Sex and Age and Score and Goal:
                button_disable = False
            enter = st.button("Enter",disabled = button_disable)
        if enter:
            show = True
            # st.write("whatever we need to do will be in this if statment",Score,Age,Sex)
            
        User_container = st.container()
        with User_container:
            expander = st.expander("Score Percentile")
            expander2 = st.expander("Age vs Score")
            
            expander4 = st.expander("credit report data and display the results")
            with expander2:
                if show:
                    st.write(csv.make_graph_age_score(Score))
            with expander:
                if show:
                    st.write(get_percentile(Score, csv.df2))
            with expander4:
                if show:
                    Advices = credit_score_goal_setting(df2, Goal,Score)
                    st.title("Advices to meet your goal")
                    for advice in Advices:
                        st.write(advice)



