import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.figure_factory as ff
from factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
import inspect
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


target_colors = {
        'satisfied': '#387149', 
        'dissatisfied': '#8b324d'
}

# Set up the Streamlit page configuration
st.set_page_config(page_title='Airline', layout='wide', page_icon="✈️")

# Set up the background image
def set_bg_hack_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("https://www.sita.aero/globalassets/images/banners/airlines-1137400812.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-filter: blur(10px);
            opacity: 0.99;  /
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_bg_hack_url()

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Introduction", "Data", "EDA", "Visualization", "PCA", "Factor Analysis", "Model","Prediction", "Conclusion"],
        icons=["newspaper", "receipt", "coin", "newspaper", "receipt", "coin", "newspaper", "umbrella", "coin", "coin", "page_with_curl", "receipt"],
        menu_icon="house",
        default_index=0
    )

# Custom CSS for sidebar
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background: url("https://t4.ftcdn.net/jpg/06/29/77/59/360_F_629775977_6SN4n28Br3FSZKZk0kd6fSnpa5kMj1gw.jpg");
        background-size: contain;
    }
</style>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv('airline (1) (1).csv')
df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].mean(), inplace=True)

# Introduction page
if selected == "Introduction":
    st.divider()
    st.markdown("<h2 style='text-align: left; font-weight: bold; font-size: 28px;'>Objective</h2>", unsafe_allow_html=True)
    st.markdown("""
    <style>
        .objective-text {
            font-family: 'Berlin Sans FB', sans-serif;
            color: Black;
            background-color: rgba(255, 255, 255, 0.5);
            font-size: 21px;
            line-height: 2.0;
            border-left: 4px solid Black;
            border-right: 4px solid Black;
            padding-left: 10px;
            margin-bottom: 20px;
        }
    </style>
    <div class="objective-text">
        The objective of this project is to develop a web application that predicts airline customer satisfaction based on various features related to customer demographics, flight characteristics, and in-flight services. This app will be built using Streamlit and will serve as an interactive tool for users to input data and receive predictions on whether a customer is satisfied or dissatisfied with their airline experience.
    </div>
    """, unsafe_allow_html=True)

# Data page
if selected == "Data":
    st.markdown("<h2 style='text-align: left; font-weight: bold; font-size: 28px;'>Description</h2>", unsafe_allow_html=True)
    st.divider()
    st.subheader("Data")
    st.write(df)

# EDA page
if selected == "EDA":
    st.markdown("<h2 style='text-align: left; font-weight: bold; font-size: 28px;'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    st.divider()
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    placeholder = st.empty()
    with col1:
        if st.button("📊 Basic Statistics"):
            with placeholder.container():
                st.write(df.describe())

    with col2:
        if st.button("🔍 Data Types"):
            st.write(df.dtypes)

    with col3:
        if st.button("🚫 Missing Values"):
            df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].mean(), inplace=True)
            st.write(df.isnull().sum())

    with col4:
        if st.button("📈Correlation Matrix"):
            with placeholder.container():
                numeric_df = df.select_dtypes(include=['number'])
                st.write(numeric_df.corr())
                corr_matrix = numeric_df.corr()
                fig, ax = plt.subplots(figsize=(22, 15))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
                st.pyplot(fig)

# Visualization page
target_colors=['#387149','#8b324d']
if selected == "Visualization":
    st.markdown("<h2 style='text-align: left; font-weight: bold; font-size: 28px;'>Visualizations</h2>", unsafe_allow_html=True)
    st.divider()

    # Dropdown list for various visualizations
    viz_type = st.selectbox("Select Visualization Type", [
        "Output Variable Visualization", 
        "Passenger Profile Visualization", 
        "Class Donut Chart",
        "Age Distribution Chart", 
        "Ordinal Data Visualization", 
        "Outlier Detection"
    ])

    if viz_type == "Output Variable Visualization":
        # Get the satisfaction counts
        satisfaction_counts = df['satisfaction'].value_counts()

        # Create the figure with two subplots
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=("Satisfaction Count", "Target Weight"),
            column_widths=[0.5, 0.5],
            specs=[[{"type": "xy"}, {"type": "domain"}]]  # Specify 'xy' for the bar chart and 'domain' for the pie chart
        )

        # First subplot: Satisfaction Count (Bar chart)
        fig.add_trace(
            go.Bar(
                x=satisfaction_counts.index,
                y=satisfaction_counts.values,
                marker=dict(color=target_colors),
            ),
            row=1, col=1
        )

        # Second subplot: Target Weight (Pie chart)
        fig.add_trace(
            go.Pie(
                labels=satisfaction_counts.index,
                values=satisfaction_counts.values,
                pull=[0, 0.090],  # 'explode' effect: first slice stays, second slice is pulled out
                marker=dict(colors=target_colors),
                textinfo='percent+label',  # Shows both percentage and label
                rotation=98  # Equivalent to 'startangle' in Matplotlib
            ),
            row=1, col=2
        )

        # Update layout for the entire figure
        fig.update_layout(
            title_text="Target Distribution",
            title_font=dict(size=24, family="moonspace", color="black"),
            title_x=0.5,  # Center the title
            height=600,  # Adjust height to match figsize
            width=1100,  # Adjust width to match figsize
            showlegend=False  # Optionally hide the legend if not needed
        )

        # Update axes titles and layout for the bar chart
        fig.update_xaxes(title_text="Satisfaction", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)

        # Display the figure
        st.plotly_chart(fig)

    elif viz_type == "Passenger Profile Visualization":
        # Add buttons to show each type of visualization
         col1, col2 = st.columns(2)
         with col1:
                gender_pie_chart = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Gender", "Customer Type"),
                    specs=[[{"type": "domain"}, {"type": "domain"}]],
                    column_widths=[0.5, 0.5],
                )

                # Gender distribution pie chart
                gender_pie_chart.add_trace(
                    go.Pie(
                        labels=df['Gender'].value_counts().index,
                        values=df['Gender'].value_counts().values,
                        pull=[0, 0.090],
                        marker=dict(colors=target_colors),
                        textinfo='percent+label',
                        rotation=90,
                    ),
                    row=1, col=1
                )

                # Customer Type distribution pie chart
                gender_pie_chart.add_trace(
                    go.Pie(
                        labels=df['Customer Type'].value_counts().index,
                        values=df['Customer Type'].value_counts().values,
                        pull=[0, 0.2],
                        marker=dict(colors=target_colors),
                        textinfo='percent+label',
                        rotation=90,
                    ),
                    row=1, col=2
                )

                gender_pie_chart.update_layout(
                    title_text='Passenger Profile',
                    title_font=dict(size=24, family='moonspace', color='black'),
                    title_x=0.5,
                    height=700,
                    width=1000,
                    showlegend=True
                )

                st.plotly_chart(gender_pie_chart)
                with col2:
                # Prepare data for Gender and Customer Type distribution
                       gender_counts = df.groupby(['Gender', 'satisfaction']).size().reset_index(name='count')

                # Gender Distribution bar chart
                fig1 = px.bar(
                    gender_counts,
                    x='Gender',
                    y='count',
                    color='satisfaction',
                    color_discrete_map=target_colors,
                    text='count',
                    title='Gender Distribution'
                )

                fig1.update_traces(texttemplate='%{text}', textposition='outside')
                fig1.update_layout(
                    title_text='Gender Distribution',
                    xaxis_title='Gender',
                    yaxis_title='Count',
                    xaxis_title_font=dict(size=20, family='moonspace'),
                    yaxis_title_font=dict(size=20, family='moonspace'),
                    xaxis=dict(tickangle=-45)
                )

                st.plotly_chart(fig1)

    elif viz_type == "Class Donut Chart":
        # Data for the pie chart
        labels = list(df['Class'].value_counts().index)
        values = list(df['Class'].value_counts().values)

        # Create the pie chart
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,  # If you want a donut chart, set this to a value between 0 and 1
            marker=dict(colors=target_colors),  # Set colors
            textinfo='label+percent'  # Show label and percentage
        )])

        # Update layout with title and ensure the pie chart is circular
        fig.update_layout(
            title_text='Class',
            title_font=dict(size=18),
            autosize=True
        )

        # Show the figure
        st.plotly_chart(fig)

    elif viz_type == "Age Distribution Chart":
        # Data for Age Distribution
        x = df["Age"]
        hist_data = [x]
        group_labels = ['Age'] # name of the dataset

        # Create the distplot
        fig = ff.create_distplot(hist_data, group_labels, curve_type='kde')
        fig.update_layout(template = 'plotly_dark')
        st.plotly_chart(fig)

#     elif viz_type == "Satisfaction Level Visualization":
#         # Prepare data for Gender and Class Distribution
#         target_colors = {
#     'Satisfied': 'blue',
#     'Neutral or Dissatisfied': 'red'
# }
#         gender_counts = df.groupby(['Gender', 'satisfaction']).size().reset_index(name='count')
#         class_counts = df.groupby(['Class', 'satisfaction']).size().reset_index(name='count')
#         fig1 = px.bar(
#         gender_counts,
#         x='Gender',
#         y='count',
#         color='satisfaction',
#         color_discrete_map=target_colors,
#         text='count',
#         title='Gender Distribution'
# )
#         fig1.update_traces(texttemplate='%{text}', textposition='outside')
#         fig1.update_layout(
#             title_text='Gender Distribution',
#             xaxis_title='Gender',
#             yaxis_title='Count',
#             xaxis_title_font=dict(size=20, family='moonspace'),
#             yaxis_title_font=dict(size=20, family='moonspace'),
#             xaxis=dict(tickangle=-45)  # Rotate x-axis labels if needed
# )
#         fig2 = px.bar(
#         class_counts,
#         x='Class',
#         y='count',
#         color='satisfaction',
#         color_discrete_map=target_colors,
#         text='count',
#         title='Class Distribution'
# )
#         fig2.update_traces(texttemplate='%{text}', textposition='outside')
#         fig2.update_layout(
#         title_text='Class Distribution',
#         xaxis_title='Class',
#         yaxis_title='Count',
#         xaxis_title_font=dict(size=20, family='moonspace'),
#         yaxis_title_font=dict(size=20, family='moonspace'),
#         xaxis=dict(tickangle=-45)  # Rotate x-axis labels if needed
# )
#         fig = make_subplots(rows=1, cols=2, subplot_titles=('Gender Distribution', 'Class Distribution'))
#         for trace in fig1.data:
#             fig.add_trace(trace, row=1, col=1)
#             for trace in fig2.data:
#                 fig.add_trace(trace, row=1, col=2)
#                 fig.update_layout(
#                     title_text='Passenger Satisfaction Level',
#                     title_font=dict(size=24, family='moonspace'),
#                     height=800,
#                     width=1200,
#                     showlegend=True
# )
#                 fig.show()

    elif viz_type == "Ordinal Data Visualization":
        # Selecting the columns of interest
        need_cols = df.loc[:, 'Inflight wifi service':'Cleanliness']

        # Creating subplots
        fig = make_subplots(
            rows=(len(need_cols.columns) // 4) + 1, 
            cols=4, 
            subplot_titles=need_cols.columns,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        # Adding count plots for each feature
        for i, col in enumerate(need_cols.columns):
            row = (i // 4) + 1
            col_position = (i % 4) + 1
            fig.add_trace(go.Bar(
                x=need_cols[col].value_counts().index,
                y=need_cols[col].value_counts().values,
                marker=dict(color='purple'),
                text=need_cols[col].value_counts().values,
                textposition='outside',
                showlegend=False
            ), row=row, col=col_position)

        # Update layout
        fig.update_layout(
            height=800,
            width=1100,
            title_text='Ordinal Features Visualization',
            showlegend=False
        )

        # Show the figure
        st.plotly_chart(fig)

    elif viz_type == "Outlier Detection":
        # Selecting columns for outlier detection
        num_cols = df.select_dtypes(include=[np.number]).columns

        # Box plots for each numerical column
        fig = make_subplots(
            rows=(len(num_cols) // 4) + 1,
            cols=4,
            subplot_titles=num_cols,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        # Adding box plots to the subplots
        for i, col in enumerate(num_cols):
            row = (i // 4) + 1
            col_position = (i % 4) + 1
            fig.add_trace(go.Box(
                y=df[col],
                marker=dict(color='blue')
            ), row=row, col=col_position)

        # Update layout
        fig.update_layout(
            height=800,
            width=1100,
            title_text='Outlier Detection Visualization'
        )

        # Show the figure
        st.plotly_chart(fig)



# PCA page
if selected == "PCA":
    st.markdown("<h2 style='text-align: left; font-weight: bold; font-size: 28px;'>PCA </h2>", unsafe_allow_html=True)
    st.divider()
    df['satisfaction'] = df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})
    df = df.iloc[:, 7:-3]  # Adjust slicing as per your data
    # PCA Analysis
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    pca = PCA()
    pca.fit(data_scaled)

    eigen_values = pca.explained_variance_
    variance_ratio = pca.explained_variance_ratio_ * 100
    cumulative_ratio = variance_ratio.cumsum()

    pca_results = pd.DataFrame({
    'eigenvalue': eigen_values,
    'percentage of variance': variance_ratio,
    'cumulative percentage of variance': cumulative_ratio
})
    pca_results.index = ['comp ' + str(i+1) for i in pca_results.index]

    #Correlation Mtarix and Heatmap
    if st.button("Correlation Matrix and Heatmap"):
       st.title("Correlation Matrix and Heatmap")
       z = df.corr()
       plt.figure(figsize=(16, 16))
       sns.heatmap(z, annot=True)
       st.pyplot(plt.gcf())


    # Hypothesis in a blue box
    if st.button("Bartlett’s Test Results"):
       st.markdown("""
       <div style="background-color: #e6f7ff; padding: 10px; border-radius: 5px;">
          <h3>Adequacy Test: Bartlett’s Test of Sphericity</h3>
          <b>HYPOTHESIS:</b>
          <ul>
            <li><b>Null Hypothesis (H0):</b> The observed variables in the dataset are not correlated, and therefore, the correlation matrix is an identity matrix (spherical).</li>
            <li><b>Alternative Hypothesis (H1):</b> The observed variables in the dataset are correlated, and the correlation matrix is not an identity matrix (non-spherical).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
       
       chi_square_value, p_value = calculate_bartlett_sphericity(df)
       st.markdown(f"""
    <div style="background-color: #cce5ff; padding: 10px; border-radius: 5px;">
        <h4>Bartlett’s Test Results</h4>
        <p><b>Chi Square Value:</b> {chi_square_value}</p>
        <p><b>P Value:</b> {p_value}</p>
    </div>
    """, unsafe_allow_html=True)

# Button for displaying KMO Test results
    if st.button("KMO Test Results"):
    # Hypothesis in a blue box
       st.markdown("""
       <div style="background-color: #e6f7ff; padding: 10px; border-radius: 5px;">
           <h3>Kaiser-Meyer-Olkin (KMO) Test</h3>
         <b>HYPOTHESIS:</b>
         <ul>
            <li><b>Null Hypothesis (H0):</b> The observed variables in the dataset are not suitable for structure detection, indicating that the partial correlations are close to zero.</li>
            <li><b>Alternative Hypothesis (H1):</b> The observed variables in the dataset are suitable for structure detection, indicating that the partial correlations are significantly different from zero.</li>
         </ul>
      </div>
    """, unsafe_allow_html=True)
       kmo_all, kmo_model = calculate_kmo(df)
       st.markdown(f"""
    <div style="background-color: #cce5ff; padding: 10px; border-radius: 5px;">
        <h4>KMO Test Results</h4>
        <p><b>KMO Value:</b> {kmo_model}</p>
    </div>
    """, unsafe_allow_html=True)


    #Scree Plot
    if st.button("Scree Plot"):
        plt.figure(figsize=(10, 6))
        plt.plot(eigen_values, marker='o')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        plt.grid(True)
        st.pyplot(plt.gcf())


    # Display PCA Results Table
    if st.button("PCA Results Table"):
        st.write(pca_results)
    
    # Plot Cumulative Variance
    if st.button("Cumulative Variance Plot"):
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_ratio, marker='o', color='green')
        plt.title('Cumulative Variance Explained by Principal Components')
        plt.xlabel('Principal Component')
        plt.ylabel('Cumulative Variance (%)')
        plt.grid(True)
        st.pyplot(plt.gcf())



# Factor Analysis page


# Streamlit app

if selected == "Factor Analysis":
    le=LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Class'] = le.fit_transform(df['Class'])
    df['Customer Type'] = le.fit_transform(df['Customer Type'])
    df['satisfaction'] = le.fit_transform(df['satisfaction'])
    df['Type of Travel'] = le.fit_transform(df['Type of Travel'])
    df_processed=df
    factor_names = ["Overall_Inflight_Experience", "E-flight_Experience", "Luggage_Logistics","Age","Off_flight_Experience", "Check-IN_Experience"]

    st.markdown("<h2 style='text-align: left; font-weight: bold; font-size: 28px;'>Factor Analysis</h2>", unsafe_allow_html=True)
    st.divider()
    
    # Select the number of factors
    n_factors = st.slider("**Select Number of Factors**", 1, 10)
    st.divider()
    st.write(f"**Extracting {n_factors} factors...**")
    
    # Perform Factor Analysis with the selected number of factors
    fa = FactorAnalyzer(n_factors=n_factors, rotation='equamax', method='principal')
    fa.fit(df)
    
    # Get Factor Loadings
    factor_loadings = pd.DataFrame(fa.loadings_, index=df_processed.columns)
    
    # Get Factor Rotation Matrix
    rotation_matrix = pd.DataFrame(fa.rotation_matrix_)
    
    # Buttons to display results
    if st.button("Factor Loadings"):
        st.write("Factor Loadings:")
        st.dataframe(factor_loadings)
    
    if st.button("Factor Rotation Matrix"):
        st.write("Factor Rotation Matrix:")
        st.dataframe(rotation_matrix)

    if st.button("Factor Names"):
        st.write("Factor Names:")
        st.write(factor_names[:n_factors])


# Model page
df_processed = pd.get_dummies(df, drop_first=True)
df_numeric = df_processed.select_dtypes(include=['number'])
df_processed = pd.get_dummies(df, drop_first=True)

# Use the stored number of factors
n_factors = st.session_state.get('n_factors',6)  # Default to 6 if not set

# Perform Factor Analysis
fa = FactorAnalyzer(n_factors=n_factors, rotation='equamax', method='principal')
fa.fit(df_numeric)
fscore = fa.transform(df_numeric)
fscore_data = pd.DataFrame(fscore)

# Splitting data for model training
x = fscore_data
y = df.iloc[:, -1]  # Assuming the target variable is in the last column
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

if selected == "Model":
    st.markdown("<h2 style='text-align: left; font-weight: bold; font-size: 28px;'>Model</h2>", unsafe_allow_html=True)
    st.divider()

    if st.button("Logistic Regression"):
        st.markdown("<h2 style='text-align: left; font-weight: bold; font-size: 28px;'>Logistic Regression with GridSearchCV</h2>", unsafe_allow_html=True)
        st.divider()
        classifier = LogisticRegression()
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        }
        classifier_regressor = GridSearchCV(classifier, param_grid=param_grid, scoring='accuracy', cv=5)
        classifier_regressor.fit(x_train, y_train)
        st.write(f"Best Parameters: {classifier_regressor.best_params_}")
        st.write(f"Best Cross-Validation Score: {classifier_regressor.best_score_:.4f}")
        y_predict = classifier_regressor.predict(x_test)
        score = accuracy_score(y_test, y_predict)

        if st.button("Show Results"):
            st.markdown("<h3>Model Results</h3>", unsafe_allow_html=True)
            st.write(f"Accuracy: {score:.4f}")

            conf_matrix = confusion_matrix(y_test, y_predict)
            st.write("Confusion Matrix:")
            st.write(conf_matrix)

            x_1 = x.head(20)
            y_1 = classifier_regressor.predict(x_1)
            y_2_actual = y.head(20)

            Final = pd.DataFrame({"Predicted": y_1, "Actual": y_2_actual})
            st.write("First 20 Predictions vs Actual:")
            st.write(Final)

            intercept = classifier_regressor.best_estimator_.intercept_
            coefficients = classifier_regressor.best_estimator_.coef_.flatten().tolist()

            st.write(f"Intercept: {intercept}")
            st.write("Coefficients:")
            st.write(coefficients)


    if st.button("Decision Tree"):
        Dc = DecisionTreeClassifier(criterion='entropy')
        Dc.fit(x_train, y_train)
        y_predict = Dc.predict(x_test)
        accuracy = accuracy_score(y_test, y_predict)
        conf_matrix = confusion_matrix(y_test, y_predict)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h3 style='text-align: center;'>Accuracy</h3>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-size: 24px; font-weight: bold;'>{accuracy:.2f}</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<h3 style='text-align: center;'>Confusion Matrix</h3>", unsafe_allow_html=True)
            st.write(pd.DataFrame(conf_matrix, index=["Actual Negative", "Actual Positive"], 
                                           columns=["Predicted Negative", "Predicted Positive"]))

    if st.button("Random Forest"):
        clf =RandomForestClassifier(random_state=42)
        clf.fit(x_train, y_train)
        y_pred_r = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred_r)
        conf_matrix = confusion_matrix(y_test, y_pred_r)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h3 style='text-align: center;'>Accuracy</h3>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-size: 24px; font-weight: bold;'>{accuracy:.2f}</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<h3 style='text-align: center;'>Confusion Matrix</h3>", unsafe_allow_html=True)
            st.write(pd.DataFrame(conf_matrix, index=["Actual Negative", "Actual Positive"], 
                                           columns=["Predicted Negative", "Predicted Positive"]))



if selected == "Prediction":
    st.markdown("<h2 style='text-align: left; font-weight: bold; font-size: 28px;'>Prediction</h2>", unsafe_allow_html=True)
    st.divider()

    # Logistic regression model coefficients and intercept
    intercept = -11.3  
    coefficients = {
        "Overall Inflight Experience": 0.96,
        "E-flight Experiece": 1.07,
        "Luggage Logistics": 0.69,
        "Off flight Experience": 0.49,
        "Check-IN Experience": 0.56
    }

    # Create sliders for each factor to take input on a scale of 1 to 5 for satisfaction
    inputs = {}
    for factor in coefficients.keys():
        value = st.slider(f"Rate your satisfaction with {factor} (1-5)", min_value=1, max_value=5, value=3)
        inputs[factor] = value

    # Compute the linear combination of inputs and coefficients using the provided formula
    linear_combination = intercept
    linear_combination += (
        coefficients["Overall Inflight Experience"] * inputs["Overall Inflight Experience"] +
        coefficients["E-flight Experiece"] * inputs["E-flight Experiece"] +
        coefficients["Luggage Logistics"] * inputs["Luggage Logistics"] +
        coefficients["Off flight Experience"] * inputs["Off flight Experience"] +
        coefficients["Check-IN Experience"] * inputs["Check-IN Experience"]
    )
if selected == "Conclusion":
    st.markdown("<h2 style='text-align: left; font-weight: bold; font-size: 28px;'>Conclusion</h2>", unsafe_allow_html=True)
    st.divider()

    if st.button("Logistic Regression"):
        st.markdown("""
        <div style="background-color: #e6f7ff; padding: 10px; border-radius: 5px;">
            <h3>Logistic Regression</h3>
            <p><b>Accuracy:</b> Indicates the proportion of correct predictions.</p>
            <p><b>Confusion Matrix:</b> Shows the detailed performance across different classes.</p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Decision Tree"):
        st.markdown("""
        <div style="background-color: #e6f7ff; padding: 10px; border-radius: 5px;">
            <h3>Decision Tree</h3>
            <p><b>Accuracy:</b> Reflects the model's classification performance.</p>
            <p><b>Confusion Matrix:</b> Provides insights into prediction errors and correct classifications.</p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Random Forest"):
        st.markdown("""
        <div style="background-color: #e6f7ff; padding: 10px; border-radius: 5px;">
            <h3>Random Forest</h3>
            <p><b>Accuracy:</b> Measures the overall effectiveness, leveraging multiple decision trees.</p>
            <p><b>Confusion Matrix:</b> Highlights predictive strengths and weaknesses.</p>
        </div>
        """, unsafe_allow_html=True)
