import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('./speed_data_data.csv')
data.dropna(subset=['attr', 'dec'], inplace=True)

# Assuming and 'career' are categorical, you might use mode (most frequent)
data['career'].fillna(data['career'].mode()[0], inplace=True)

# Impute 'goal' by the mode (most common value) within each age group
data['goal'] = data.groupby('age')['goal'].transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else "Fallback_Mode"))

# Impute 'income' by the mean within each career group
data['income'] = data.groupby('career')['income'].transform(lambda x: x.fillna(x.mean()))

# Fill missing numeric data with the median of the column
for col in ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'prob', 'met']:
    if data[col].isnull().any():
        data[col].fillna(data[col].median(), inplace=True)

# If age is critically important and should not be guessed, drop rows where age is missing
data.dropna(subset=['age'], inplace=True)

# Streamlit title
st.title("Speed Dating Data Analysis")

# Filters for gender and age range
gender_options = {1: 'Male', 0: 'Female'}
selected_genders = st.multiselect('Select Gender', options=list(gender_options.values()), default=list(gender_options.values()))
gender_filter = [key for key, value in gender_options.items() if value in selected_genders]
age_filter = st.slider('Select Age Range', min_value=int(data['age'].min()), max_value=int(data['age'].max()), value=(int(data['age'].min()), int(data['age'].max())))

# Attractiveness Rating Filter
attr_rating_filter = st.slider('Select Attractiveness Rating Range', min_value=int(data['attr'].min()), max_value=int(data['attr'].max()), value=(int(data['attr'].min()), int(data['attr'].max())))


income_filter = st.slider('Select Income Range', min_value=int(data['income'].min()), max_value=int(data['income'].max()), value=(int(data['income'].min()), int(data['income'].max())))

# Goals Filter
goal_options = {
    1: 'Fun night out',
    2: 'Meet new people',
    3: 'Get a date',
    4: 'Serious relationship',
    5: 'To say I did it',
    6: 'Other'
}
selected_goals = st.multiselect('Select Goals', options=list(goal_options.values()), default=list(goal_options.values()))

# Convert selected labels back to numeric
goal_filter = [key for key, value in goal_options.items() if value in selected_goals]

# Apply all filters to data
filtered_data = data[(data['gender'].isin(gender_filter)) &
                     (data['age'].between(*age_filter)) &
                     (data['attr'].between(*attr_rating_filter)) &
                     (data['income'].between(*income_filter) if 'income' in data.columns else True) &
                     (data['goal'].isin(goal_filter))]


# Filter out rows where 'attr' or 'dec' is NaN
filtered_data = filtered_data.dropna(subset=['attr', 'dec'])

# Calculate success rate and count for each attractiveness score
grouped_data = filtered_data.groupby('attr').agg(average_success_rate=('dec', 'mean'), count=('dec', 'size')).reset_index()
grouped_data = grouped_data[grouped_data['count'] >= 10]

# Create a bubble chart
fig1 = go.Figure()

# Calculate size reference for the bubbles
max_count = grouped_data['count'].max()
sizeref = 4. * max_count / (100. ** 2)  # Adjusting size reference for visibility

# Add the bubbles to the plot
fig1.add_trace(
    go.Scatter(
        x=grouped_data['attr'],
        y=grouped_data['average_success_rate'],
        mode='markers',
        marker=dict(
            size=grouped_data['count'],
            color=['Maroon', 'Brown', 'Navy', 'Purple', 'Olive', 'Red', 'Green', 'Teal', 'Blue', 'Magenta', 'Orange', 'Pink', 'Gray', 'Lavender', 'Yellow', 'Lime', 'Cyan'],
            sizemode='area',
            sizeref=sizeref,
            sizemin=4
        ),
        text=[f'Attr Score: {attr} - Count: {count}' for attr, count in zip(grouped_data['attr'], grouped_data['count'])]
    )
)

# Update layout
fig1.update_layout(
    title='Average Success Rate by Attractiveness Score',
    xaxis_title='Attractiveness Score',
    xaxis=dict(tickmode='linear', tick0=1, dtick=1),  # Ensuring the x-axis goes from 1 to 10
    yaxis_title='Average Success Rate',
    yaxis=dict(tickformat=".0%")
)

# Display the Streamlit application
st.header("Average Success Rate by Attractiveness Score")

# Bubble chart for success rate by attractiveness score
st.plotly_chart(fig1)



# Group filtered data by goals and decision
goals_data = filtered_data.groupby(['goal', 'dec']).size().unstack(fill_value=0)

# Add descriptions for goals
goal_labels = {
    1: 'Fun night out',
    2: 'Meet new people',
    3: 'Get a date',
    4: 'Serious relationship',
    5: 'To say I did it',
    6: 'Other'
}
goals_data.index = [goal_labels[i] for i in goals_data.index]

# Sorting options
sort_option = st.selectbox(
    "Sort the data by:",
    ('Alphabetically', 'Total Decisions Descending', 'Total Decisions Ascending'))

if sort_option == 'Alphabetically':
    goals_data.sort_index(inplace=True)
elif sort_option == 'Total Decisions Descending':
    goals_data['Total'] = goals_data.sum(axis=1)
    goals_data.sort_values(by='Total', ascending=False, inplace=True)
    goals_data.drop(columns='Total', inplace=True)
elif sort_option == 'Total Decisions Ascending':
    goals_data['Total'] = goals_data.sum(axis=1)
    goals_data.sort_values(by='Total', ascending=True, inplace=True)
    goals_data.drop(columns='Total', inplace=True)

# Create subplots for absolute numbers and relative proportions
st.header("Decision Outcomes by Goal")
fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Absolute Numbers", "Relative Proportions"),
                     specs=[[{"type": "bar"}, {"type": "bar"}]])

# Add bar charts for absolute numbers
fig2.add_trace(
    go.Bar(x=goals_data.index, y=goals_data[0], name='No', marker_color='red'), row=1, col=1
)
fig2.add_trace(
    go.Bar(x=goals_data.index, y=goals_data[1], name='Yes', marker_color='blue'), row=1, col=1
)

# Calculate relative proportions
relative_data = goals_data.div(goals_data.sum(axis=1), axis=0)

# Add bar charts for relative proportions
fig2.add_trace(
    go.Bar(x=relative_data.index, y=relative_data[0], name='No (relative)', marker_color='red'), row=1, col=2
)
fig2.add_trace(
    go.Bar(x=relative_data.index, y=relative_data[1], name='Yes (relative)', marker_color='blue'), row=1, col=2
)

# Update layout for clarity
fig2.update_layout(barmode='group', title_text="Decision Outcomes by Goal")
st.plotly_chart(fig2)

# Recalculate success rates by age and gender for filtered data
filtered_data['success'] = filtered_data['dec'].map({1: 'Success', 0: 'Failure'})
success_data = filtered_data.groupby(['age', 'gender']).agg(success_rate=('dec', 'mean')).reset_index()

# Filter data for males and females
male_data = success_data[success_data['gender'] == 1]
female_data = success_data[success_data['gender'] == 0]

# Create a Plotly figure for success rates by age and gender
st.header("Success Rates by Age and Gender")
fig3 = go.Figure()

# Add Male trace for success rate
fig3.add_trace(go.Scatter(x=male_data['age'], y=male_data['success_rate'], name='Male', mode='lines+markers',
                          line=dict(color='blue'), marker=dict(color='blue')))

# Add Female trace for success rate
fig3.add_trace(go.Scatter(x=female_data['age'], y=female_data['success_rate'], name='Female', mode='lines+markers',
                          line=dict(color='red'), marker=dict(color='red')))

# Update plot layout
fig3.update_layout(
    title="Success Rates by Age and Gender",
    xaxis_title="Age",
    yaxis_title="Success Rate",
    yaxis=dict(
        title="Success Rate (%)",
        range=[0, 1],  # Assuming success rate is between 0 and 1
        tickformat=".0%"  # Format as percentage
    ),
    legend_title="Gender"
)

st.plotly_chart(fig3)


imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data.select_dtypes(include=np.number)), columns=data.select_dtypes(include=np.number).columns)
label_encoder = LabelEncoder()
for column in data.select_dtypes(include=object).columns:
    data_imputed[column] = label_encoder.fit_transform(data[column].astype(str))
columns_to_remove = ['success_rate', 'success', 'decision_name', 'attr_group']
data_imputed = data_imputed.drop(columns=[col for col in columns_to_remove if col in data_imputed.columns])
X = data_imputed.drop(columns=['dec'])
y = data_imputed['dec']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Streamlit code for visualization
st.title('Feature Importance Analysis')
st.subheader('Feature Importance Bar Plot')

# Plotting using Seaborn
fig4, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)

plt.title('The most influential features on date success')
plt.xlabel('Importance')
plt.ylabel('Feature')
st.pyplot(fig4)
