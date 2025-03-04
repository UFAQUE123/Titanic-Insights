# ----------------- Dashboard for Survival Prediction of Titanic Passengers -----------------------------#
# import libraries
# -----------------------------
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# -----------------------------
# Load the saved trained model
# -----------------------------
model = joblib.load('./notebook/titanic_trained_model.pkl')

# -----------------------------
# Dashboard Title and Description
# -----------------------------
st.title("🚢 Titanic Survival Prediction Dashboard")
st.write("Enter the details of the passenger to predict survival.")

# -----------------------------
# User Input Section
# -----------------------------
st.subheader("Passenger Details")

# Passenger Class (pclass)
pclass = st.selectbox("Passenger Class", options=[1, 2, 3])


# Cabin Class (as provided in 'class' column)
travel_class = st.selectbox("Travel Class", options=["First", "Second", "Third"])


# Who (e.g., man, woman, child)
who = st.selectbox("Who", options=["man", "woman", "child"])

# Sex
sex = st.selectbox("Sex", options=["male", "female"])

# Age
age_ranges = np.sort(np.unique([22.        , 38.        , 26.        , 35.        , 29.69911765,
       54.        ,  2.        , 27.        , 14.        ,  4.        ,
       58.        , 20.        , 39.        , 55.        , 31.        ,
       34.        , 15.        , 28.        ,  8.        , 19.        ,
       40.        , 66.        , 42.        , 21.        , 18.        ,
        3.        ,  7.        , 49.        , 29.        , 65.        ,
       28.5       ,  5.        , 11.        , 45.        , 17.        ,
       32.        , 16.        , 25.        ,  0.83      , 30.        ,
       33.        , 23.        , 24.        , 46.        , 59.        ,
       71.        , 37.        , 47.        , 14.5       , 70.5       ,
       32.5       , 12.        ,  9.        , 36.5       , 51.        ,
       55.5       , 40.5       , 44.        ,  1.        , 61.        ,
       56.        , 50.        , 36.        , 45.5       , 20.5       ,
       62.        , 41.        , 52.        , 63.        , 23.5       ,
        0.92      , 43.        , 60.        , 10.        , 64.        ,
       13.        , 48.        ,  0.75      , 53.        , 57.        ,
       80.        , 70.        , 24.5       ,  6.        ,  0.67      ,
       30.5       ,  0.42      , 34.5       , 74.        ]))  
# Select box for age selection
age = st.selectbox("Age", options=age_ranges)

# alone 
alone = st.selectbox("alone", options= ["Yes", "No"])


# Number of Siblings/Spouses aboard
sibsp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=10, value=0)


# Number of Parents/Children aboard
parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=10, value=0)

# Fare
# List of unique fare values from the Titanic dataset
fare_options = np.sort(np.unique([
    7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625, 21.075, 11.1333, 30.0708, 
    16.7, 26.55, 31.275, 7.8542, 16.0, 29.125, 13.0, 18.0, 7.225, 26.0, 8.0292, 
    35.5, 31.3875, 263.0, 7.8792, 7.8958, 27.7208, 146.5208, 7.75, 10.5, 82.1708, 
    52.0, 7.2292, 11.2417, 9.475, 21.0, 41.5792, 15.5, 21.6792, 17.8, 39.6875, 
    7.8, 76.7292, 61.9792, 27.75, 46.9, 80.0, 83.475, 27.9, 15.2458, 8.1583, 
    8.6625, 73.5, 14.4542, 56.4958, 7.65, 29.0, 12.475, 9.0, 9.5, 7.7875, 47.1, 
    15.85, 34.375, 61.175, 20.575, 34.6542, 63.3583, 23.0, 77.2875, 8.6542, 
    7.775, 24.15, 9.825, 14.4583, 247.5208, 7.1417, 22.3583, 6.975, 7.05, 14.5, 
    15.0458, 26.2833, 9.2167, 79.2, 6.75, 11.5, 36.75, 7.7958, 12.525, 66.6, 
    7.3125, 61.3792, 7.7333, 69.55, 16.1, 15.75, 20.525, 55.0, 25.925, 33.5, 
    30.6958, 25.4667, 28.7125, 0.0, 15.05, 39.0, 22.025, 50.0, 8.4042, 6.4958, 
    10.4625, 18.7875, 31.0, 113.275, 27.0, 76.2917, 90.0, 9.35, 13.5, 7.55, 
    26.25, 12.275, 7.125, 52.5542, 20.2125, 86.5, 79.65, 153.4625, 
    135.6333, 19.5, 29.7, 77.9583, 20.25, 78.85, 91.0792, 12.875, 8.85, 151.55, 
    30.5, 23.25, 12.35, 110.8833, 108.9, 24.0, 56.9292, 83.1583, 262.375, 14.0, 
    164.8667, 134.5, 6.2375, 57.9792, 28.5, 133.65, 15.9, 9.225, 35.0, 75.25, 
    69.3, 55.4417, 211.5, 4.0125, 227.525, 15.7417, 7.7292, 12.0, 120.0, 12.65, 
    18.75, 6.8583, 32.5, 7.875, 14.4, 55.9, 8.1125, 81.8583, 19.2583, 19.9667, 
    89.1042, 38.5, 7.725, 13.7917, 9.8375, 7.0458, 7.5208, 12.2875, 9.5875, 
    49.5042, 78.2667, 15.1, 7.6292, 22.525, 26.2875, 59.4, 7.4958, 34.0208, 
    93.5, 221.7792, 106.425, 49.5, 71.0, 13.8625, 7.8292, 39.6, 17.4, 51.4792, 
    26.3875, 30.0, 40.125, 8.7125, 15.0, 33.0, 42.4, 15.55, 65.0, 32.3208, 
    7.0542, 8.4333, 25.5875, 9.8417, 8.1375, 10.1708, 211.3375, 57.0, 13.4167, 
    7.7417, 9.4833, 7.7375, 8.3625, 23.45, 25.9292, 8.6833, 8.5167, 7.8875, 
    37.0042, 6.45, 6.95, 8.3, 6.4375, 39.4, 14.1083, 13.8583, 50.4958, 5.0, 
    9.8458, 10.5167
]))
# Select box for fare selection
fare = st.selectbox("Fare", options=fare_options)

# Embark Town
embark_town = st.selectbox("Embark Town", options=["Southampton", "Cherbourg", "Queenstown"])

# Embarked Ports
embarked = st.selectbox("Port of Embarkation", options=["S", "C", "Q"])

# Create a DataFrame from user inputs. Ensure the column names match what the model expects.
input_data = pd.DataFrame({
    'pclass': [pclass],
    'class': [travel_class],
    'who': [who],
    'sex': [sex],
    'age': [age],
    "alone": [alone],
    'sibsp': [sibsp],
    'parch': [parch],
    'fare': [fare],
    'embark_town': [embark_town],
    'embarked': [embarked]
})

st.write("### Input Data")
st.dataframe(input_data)

# --------------------------------------------------------------------------------------------------------------------------------------------------------#
# Prediction
# --------------------------------------------------------------------------------------------------------------------------------------------------------#
if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    # Convert prediction (0 or 1) to a meaningful output
    result = "**The passenger survived!**" if prediction[0] == 1 else "The passenger did not survive."
    st.write("### Prediction")
    st.write(f"**{result}**")
       
#########################################################################################################################################################

##  load the cleaned dataset

cl_data = pd.read_csv("./dataset/cleaned_data.csv")

########################## Tables and Plots ###########################################################################

#--------------------------------------table 1----------------------------------------------------------#
# survival rate data prep

st.header("Survival Rate of Titanic Passengers")
total_passengers = cl_data.shape[0]
st.write(f"Total passengers: {total_passengers}")
# survival count table
survive  = cl_data["survived"].value_counts().to_frame()
survive["survival %"]= cl_data["survived"].value_counts(normalize=True) * 100
# Reset index to make 'survived' a column instead of an index
survive.reset_index(inplace=True)
survive.columns = ['survived', 'count', 'survival %']

# Map survival values to labels
survive['survived'] = survive['survived'].map({0: 'Not Survived', 1: 'Survived'})

data_survive = pd.DataFrame(survive)
st.write(data_survive)

#---------------------------------------PLOT 1----------------------------------------------------------#

# Convert data to long format for Plotly
data_long = survive.melt(id_vars=['survived'], value_vars=['count', 'survival %'], 
                         var_name='Metric', value_name='Value')
# Create a grouped bar chart
fig = px.bar(data_long, x='survived', y='Value', color='Metric', 
             barmode='group',
             labels={'survived': 'Survival Status', 'Value': 'Count / Percentage'},
             title="Visualization of Survival Count & Percentage in Titanic Dataset")

# Display the chart
st.plotly_chart(fig)

#--------------------------------------table 2----------------------------------------------------------#

# Display the Survival Count and Percentage by Gender 
st.header("Survival Rates by Gender")
female_passengers = cl_data[cl_data["sex"] == "female"].shape[0]
st.write(f"Total Female Passengers: {female_passengers}")
male_passengers = cl_data[cl_data["sex"]== "male"].shape[0]
st.write(f"Total Male Passengers: {male_passengers}")
survival = cl_data.groupby("sex")["survived"].value_counts().to_frame()
survival["survival %"] = cl_data.groupby("sex")["survived"].value_counts(normalize=True) * 100

# Reset index to make 'survived' a column instead of an index
survival.reset_index(inplace=True)
survival.columns = ["sex",'survived', 'count', 'survival %']

# Map survival values to labels
survival['survived'] = survival['survived'].map({0: 'Not Survived', 1: 'Survived'})

data_survival = pd.DataFrame(survival)
st.write(data_survival)


#--------------------------------------plot 2----------------------------------------------------------#
# Convert data to long format for grouped bar plot
data_long = data_survival.melt(id_vars=["sex", "survived"], value_vars=["count", "survival %"], 
                          var_name="Metric", value_name= "Value")
# Create a grouped bar chart
fig = px.bar(data_long, x="sex", y="Value", color="Metric", 
             barmode="group", facet_col="survived",
             title="Visualization of Survival Count & Percentage by Gender",
            labels={"sex": "Gender", "Value": "Count / Percentage"})

# Adjust facet titles and spacing
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))  # Clean facet titles
# Display the chart
st.plotly_chart(fig)
#---------------------------------------table 3----------------------------------------------------------#
# Survival Rate by Class and Who
# -----------------------------
st.header("Survival Rate by Passenger Class and Passenger Type")
child_passengers = cl_data[cl_data["who"]== "child"].shape[0]
st.write(f"Total Child Passengers: {child_passengers}")
woman_passengers = cl_data[cl_data["who"]== "woman"].shape[0]
st.write(f"Total Woman Passengers: {woman_passengers}")
man_passengers = cl_data[cl_data["who"]== "man"].shape[0]
st.write(f"Total Man Passengers: {man_passengers}")
survival = cl_data.groupby(['pclass', 'who'])["survived"].value_counts().to_frame()
survival["survival %"] = cl_data.groupby(['pclass', 'who'])["survived"].value_counts(normalize=True) * 100
# Reset index to make 'survived' a column instead of an index
survival.reset_index(inplace=True)
survival.columns = ["pclass","who",'survived', 'count', 'survival %']
# Map survival values to labels
survival['survived'] = survival['survived'].map({0: 'Not Survived', 1: 'Survived'})

data_survival = pd.DataFrame(survival)
st.write(data_survival)

#--------------------------------------plot 3----------------------------------------------------------#

# Show more insights on survival by Class and Who
fig = px.bar(cl_data.groupby(['pclass', 'who'])['survived'].mean().reset_index(),
             x='pclass', y='survived', color='who', barmode='group', 
             title="Visualization of Survival Rate by Passenger Class and Passenger Type")
st.plotly_chart(fig)
#-------------------------------------PLOT 4----------------------------------------------------------#
st.subheader("Age Distribution of Titanic Passengers")
# Create a histogram using Plotly
fig = px.histogram(cl_data, x="age", nbins=30, 
                #  title="Age Distribution of Titanic Passengers",
                   labels={"age": "Age"},
                   color_discrete_sequence=px.colors.qualitative.Set2)  # Equivalent to Seaborn 'Set2'
# Update layout for better readability
fig.update_layout(
    xaxis_title="Age",
    yaxis_title="Count",
    bargap=0.1) # Adjust spacing between bars
# Display the plot in Streamlit
st.plotly_chart(fig)
#-------------------------------------PLOT 5----------------------------------------------------------#
# Streamlit App
st.subheader("Survival Rate by Age Group and  Passenger Class")

# Convert 'pclass' to a categorical variable for proper color mapping
cl_data["pclass"] = cl_data["pclass"].astype(str)

# Define a distinct color mapping for Pclass
custom_colors = {"1": "teal", "2": "skyblue", "3": "salmon"}  # Distinct colors for each class

# Create a 2D scatter plot using Plotly
fig = px.scatter(cl_data, x="age", y="survived", 
                 color="pclass",
                 labels={"age": "Age", "survived": "Survival", "pclass": "Passenger Class"},
                 color_discrete_map=custom_colors)  # Assign custom colors

# Update layout for better readability
fig.update_layout(
    xaxis_title="Age",
    yaxis_title="Survival Status"
)
# Display the plot in Streamlit
st.plotly_chart(fig)
#-------------------------------------PLOT 6----------------------------------------------------------#
# Streamlit App
st.subheader("Survival Rate by Age Group and Passenger Type")
# Define a distinct color mapping for Pclass
custom_colors = {"man": "teal", "woman": "skyblue", "child": "salmon"}  # Distinct colors for each class

# Create a 2D scatter plot using Plotly
fig = px.scatter(cl_data, x="age", y="survived", 
                 color="who",
                 labels={"age": "Age", "survived": "Survival", "who": "who"},
                 color_discrete_map=custom_colors)  # Assign custom colors

# Update layout for better readability
fig.update_layout(
    xaxis_title="Age",
    yaxis_title="Survival Status"
)
# Display the plot in Streamlit
st.plotly_chart(fig)
#-------------------------------------PLOT 7----------------------------------------------------------#
st.subheader("Fare Distribution of Titanic Passengers")
# Create a histogram using Plotly
fig = px.histogram(cl_data, x="fare", nbins=30, 
                #  title="Age Distribution of Titanic Passengers",
                   labels={"fare": "Fare"},
                   color_discrete_sequence=px.colors.qualitative.Set2)  # Equivalent to Seaborn 'Set2'
# Update layout for better readability
fig.update_layout(
    xaxis_title="Fare",
    yaxis_title="Count",
    bargap=0.1) # Adjust spacing between bars
# Display the plot in Streamlit
st.plotly_chart(fig)
#-------------------------------------PLOT 8----------------------------------------------------------#

cl_data["pclass"] = cl_data["pclass"].astype(int)
# Filter rows where 'pclass' equals 1
filtered_df = cl_data[cl_data['pclass'] == 1]
st.subheader("Distribution of Fare by Passenger Class 1 and Passenger Type")
# Create a histogram using Plotly
fig = px.histogram(filtered_df, x="fare", color="who", nbins=30, 
                #  title="Age Distribution of Titanic Passengers",
                   labels={"fare": "Fare"},
                   color_discrete_sequence=px.colors.qualitative.Set2)  # Equivalent to Seaborn 'Set2'
# Update layout for better readability
fig.update_layout(
    xaxis_title="Fare",
    yaxis_title="Frequency",
    bargap=0.1) # Adjust spacing between bars
# Display the plot in Streamlit
st.plotly_chart(fig)
#-------------------------------------PLOT 9----------------------------------------------------------#
# Streamlit App
# Filter rows where 'pclass' equals 2
filtered_df = cl_data[cl_data['pclass'] == 2]
st.subheader("Distribution of Fare by Passenger Class 2 and Passenger Type")
# Create a histogram using Plotly
fig = px.histogram(filtered_df, x="fare", color="who", nbins=30, 
                #  title="Age Distribution of Titanic Passengers",
                   labels={"fare": "Fare"},
                   color_discrete_sequence=px.colors.qualitative.Set2)  # Equivalent to Seaborn 'Set2'
# Update layout for better readability
fig.update_layout(
    xaxis_title="Fare",
    yaxis_title="Frequency",
    bargap=0.1) # Adjust spacing between bars
# Display the plot in Streamlit
st.plotly_chart(fig)
#-------------------------------------PLOT 10----------------------------------------------------------#
# Streamlit App
# Filter rows where 'pclass' equals 3
filtered_df = cl_data[cl_data['pclass'] == 3]
st.subheader("Distribution of Fare by Passenger Class 3 and Passenger Type")
# Create a histogram using Plotly
fig = px.histogram(filtered_df, x="fare", color="who", nbins=30, 
                #  title="Age Distribution of Titanic Passengers",
                   labels={"fare": "Fare"},
                   color_discrete_sequence=px.colors.qualitative.Set2)  # Equivalent to Seaborn 'Set2'
# Update layout for better readability
fig.update_layout(
    xaxis_title="Fare",
    yaxis_title="Frequency",
    bargap=0.1) # Adjust spacing between bars
# Display the plot in Streamlit
st.plotly_chart(fig)

#-------------------------------------PLOT 11----------------------------------------------------------#
st.subheader("Distribution of Fare on the Titanic by Passenger Type and Passenger class")
# Convert 'pclass' to a categorical variable for proper color mapping
cl_data["pclass"] = cl_data["pclass"].astype(str)

# Define a distinct color mapping for Pclass
custom_colors = {"1": "red", "2": "green", "3": "blue"}  # Distinct colors for each class
# Create a 2D scatter plot using Plotly
fig = px.scatter(cl_data, x="who", y="fare", 
                 color="pclass",
                 labels={"fare": "Fare", "pclass": "Passecenger Class", "who": "Passenger Type"},
                 color_discrete_map=custom_colors)  # Assign custom colors

# Update layout for better readability
fig.update_layout(
    xaxis_title="WHO",
    yaxis_title="Fare"
)
# Display the plot in Streamlit
st.plotly_chart(fig)
#-------------------------------------PLOT 12----------------------------------------------------------#
st.subheader("Distribution of Fare on the Titanic by Age and Passenger class")
# Convert 'pclass' to a categorical variable for proper color mapping
cl_data["pclass"] = cl_data["pclass"].astype(str)

# Define a distinct color mapping for Pclass
custom_colors = {"1": "salmon", "2": "skyblue", "3": "blue"}  # Distinct colors for each class
# Create a 2D scatter plot using Plotly
fig = px.scatter(cl_data, x="age", y="fare", 
                 color="pclass",
                 labels={"fare": "Fare", "pclass": "Passecenger Class", "age": "Age"},
                 color_discrete_map=custom_colors)  # Assign custom colors

# Update layout for better readability
fig.update_layout(
    xaxis_title="Age",
    yaxis_title="Fare"
)
# Display the plot in Streamlit
st.plotly_chart(fig)
##-------------------------------------PLOT 13----------------------------------------------------------#
st.subheader("Fare vs. Age by Passenger Type")
# Convert 'survived' and 'pclass' to categorical for better plotting
cl_data["survived"] = cl_data["survived"].astype(str)  
cl_data["pclass"] = cl_data["pclass"].astype(str)  

# Define a distinct color mapping for Survival Status
custom_colors = {"0": "skyblue", "1": "salmon"}  # Not Survived = Blue, Survived = Green

# Create an enhanced facet scatter plot using Plotly
fig = px.scatter(cl_data, 
                 x="fare", 
                 y="age", 
                 color="survived", 
                 facet_col="who",  # Creates separate plots for each 'who' value
                 labels={"fare": "Fare", "age": "Age", "who": "Passenger Type", "survived": "Survival"},
                 color_discrete_map=custom_colors,  # Ensure correct survival colors
                 opacity=0.8,  # Improves overlapping visibility
                 hover_data=["pclass", "embarked"])  # Show extra info in tooltips

# Improve layout readability
fig.update_layout(
    xaxis_title="Fare",
    yaxis_title="Age",
    legend_title="Survival Status",
    margin=dict(t=50, l=50, r=50, b=50),
    font=dict(size=12),
)

# Adjust facet titles and spacing
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))  # Clean facet titles
fig.update_xaxes(matches=None)  # Allow independent x-axes for better visualization

# Display the plot in Streamlit
st.plotly_chart(fig)
#-------------------------------------PLOT 14----------------------------------------------------------#

# Show more insights on survival by Class and Embarked
cl_data["survived"] = cl_data["survived"].astype(int)  
cl_data["pclass"] = cl_data["pclass"].astype(int) 
st.subheader("Survival Rate by Class and Embarked")
fig = px.bar(cl_data.groupby(['pclass', 'embarked'])['survived'].mean().reset_index(), 
             x='pclass', y='survived', color='embarked', barmode='group', title="Survival Rate by Class and Embarked")
st.plotly_chart(fig)
#-------------------------------------PLOT 15----------------------------------------------------------#
# Show more insights on survival by Who and Embarked
st.subheader("Survival Rate by Passenger Type and Embarked")
fig = px.bar(cl_data.groupby(['who', 'embarked'])['survived'].mean().reset_index(), 
             x='who', y='survived', color='embarked', barmode='group', title="Survival Rate by Who and Embarked")
st.plotly_chart(fig)
#-------------------------------------PLOT 16----------------------------------------------------------#
# Actual vs Predicted Comparison
# Split the dataset into training and testing sets
X = cl_data.drop("survived", axis=1)  # Features
y = cl_data["survived"]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.subheader("Actual vs Predicted Survival Rates")
# Get actual vs predicted survival data from test set
y_pred_test = model.predict(X_test)
actual_vs_predicted = pd.DataFrame({
    'Category': ['Actual', 'Predicted'],
    'Survival Rate': [y_test.mean(), y_pred_test.mean()]
})

# Create an improved bar chart using Plotly
fig = px.bar(actual_vs_predicted, 
             x="Category", 
             y="Survival Rate", 
             color="Category", 
             text=actual_vs_predicted["Survival Rate"].apply(lambda x: f"{x:.2%}"),  # Display as percentage
             labels={"Survival Rate": "Survival Rate (%)"},
             color_discrete_map={"Actual": "skyblue", "Predicted": "salmon"})

# Improve layout
fig.update_traces(textposition='outside')  # Show text labels outside bars
fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Survival Rate (%)")  # Format y-axis as percentage

# Display the plot in Streamlit
st.plotly_chart(fig)

######################### THE END #####################################################################




