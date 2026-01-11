import streamlit as st

# ---- LOGIN PAGE ---- #
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

def login():
    st.title("🔐 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if username == "admin" and password == "admin123":
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("✅ Login successful!")
            st.rerun() 
        else:
            st.error("❌ Invalid username or password")

def logout():
    st.sidebar.write(f"👤 Logged in as: `{st.session_state.username}`")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun() 

if not st.session_state.authenticated:
    login()
    st.stop()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split  # Corrected line
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import time

st.markdown(
    """
    <style>
        /* Background color for the main content */
        div.stApp {
            background-color: #FFF8E7;
        }

        /* Styling the heading */
        h1 {
            color: #007bff;
            text-align: center;
        }

        /* Background color for the sidebar */
        section[data-testid="stSidebar"] {
            background-color: #EFBFA5;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App Title
st.title("📊 Cyber Security Intrusion Detection Model")

# Sidebar Navigation (Added new option)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "Data Processing", "Visualization", "Model Training", "Real-Time Data Filtering"])

# ===================== DATA UPLOAD ====================== #
if page == "Upload Data":
    st.header("📂 Upload Train & Test Data")
    train_file = st.file_uploader("Upload Train Data (CSV)", type=["csv"])
    test_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"])

    if train_file and test_file:
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        data = pd.concat([train_data, test_data], ignore_index=True)
        st.session_state['data'] = data
        st.success("✅ Data Uploaded Successfully!")

# ===================== DATA PROCESSING ====================== #
elif page == "Data Processing":
    st.header("⚙️ Data Processing")
    
    if 'data' in st.session_state:
        data = st.session_state['data']
        
        st.subheader("Dataset Information")
        st.write(data.info())

        st.subheader("Missing Values")
        missing_values = data.isnull().sum()
        st.write(missing_values[missing_values > 0])
        data.dropna(inplace=True)

        st.subheader("Duplicate Rows")
        st.write(f"🔄 {data.duplicated().sum()} duplicate rows found.")
    else:
        st.warning("⚠️ Please upload datasets first!")

# ===================== DATA VISUALIZATION ====================== #
elif page == "Visualization":
    st.header("📊 Data Visualization")

    if 'data' in st.session_state:
        data = st.session_state['data']
        num_cols = data.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = data.select_dtypes(include=['object']).columns
        
        st.subheader("First 5 Rows of the Dataset")
        st.write(data.head())

        st.subheader("Summary Statistics")
        st.write(data.describe())

        if 'class' in data.columns:
            st.subheader("Class Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x='class', data=data, palette='coolwarm', ax=ax)
            st.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(6, 6))
            data['class'].value_counts().plot.pie(autopct='%1.1f%%', cmap='coolwarm', startangle=90, wedgeprops={'edgecolor': 'black'}, ax=ax)
            plt.title("Class Distribution")
            st.pyplot(fig)

        # Box Plot for Outlier Detection
        st.subheader("📦 Box Plot (Outliers Detection)")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=data[num_cols], ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)

        # KDE Plot
        st.subheader("📈 Kernel Density Estimation (KDE) Plot")
        for col in num_cols[:5]:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.kdeplot(data[col], fill=True, color="skyblue", ax=ax)
            st.pyplot(fig)

        # Regression Plot
        st.subheader("📊 Regression Plots (Feature Relationships)")
        for col in num_cols[:5]:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.regplot(x=col, y=num_cols[-1], data=data, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax)
            st.pyplot(fig)

        # Pie Charts for Categorical Features
        st.subheader("🥧 Pie Charts")
        selected_columns = ["protocol_type", "service", "flag", "land", "logged_in"]
        for col in selected_columns:
            if col in data.columns:
                fig, ax = plt.subplots()
                data[col].value_counts().nlargest(5).plot.pie(autopct='%1.1f%%', cmap='coolwarm', ax=ax)
                ax.set_title(col)
                st.pyplot(fig)

        # Feature Importance Visualization
        st.subheader("🌟 Feature Importance (Random Forest)")
        if 'class' in data.columns:
            X = data.drop(columns=['class'], errors='ignore')
            y = data['class']
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X, y)
            importances = rf_model.feature_importances_
            feature_names = X.columns
            feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', ax=ax)
            plt.title("Top 10 Feature Importances (Random Forest)", fontsize=14, pad=10)
            plt.xlabel("Importance Score", fontsize=12)
            plt.ylabel("Feature", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            st.write("Top 10 Features by Importance:")
            st.write(feature_importance_df)
        else:
            st.warning("⚠️ 'class' column not found in the dataset for feature importance calculation.")

        # Histogram
        st.subheader("📈 Feature Distributions (Histograms)")
        fig, ax = plt.subplots(figsize=(12, 10))
        data[num_cols].hist(bins=30, edgecolor='black', ax=ax)
        plt.suptitle("Feature Distributions", fontsize=16)
        plt.tight_layout()
        st.pyplot(fig)

        # Separate Histogram
        st.subheader("📈 Feature Distributions (Histograms)")
        num_cols = data.select_dtypes(include=['number']).columns[:5]
        fig, axes = plt.subplots(len(num_cols), 1, figsize=(10, 20))
        for i, col in enumerate(num_cols):
            axes[i].hist(data[col], bins=15, edgecolor='black')
            axes[i].set_title(col)
        plt.tight_layout()
        st.pyplot(fig)

        # Correlation Heatmap
        st.subheader("🌡️ Feature Correlation Heatmap")
        fig = plt.figure(figsize=(12, 8))
        sns.heatmap(data[num_cols].corr(), annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title("Feature Correlation Heatmap")
        st.pyplot(fig)

        # Violin Plots by Class
        if 'class' in data.columns:
            st.subheader("🎻 Violin Plots by Class")
            for col in num_cols[:5]:
                fig = plt.figure(figsize=(6, 4))
                sns.violinplot(x='class', y=col, data=data, palette="coolwarm")
                plt.title(f"Violin Plot of {col} by Class")
                st.pyplot(fig)

    else:
        st.warning("⚠️ Please upload datasets first!")

# ===================== MODEL TRAINING ====================== #
elif page == "Model Training":
    st.header("🧠 Train Machine Learning Models")

    if 'data' in st.session_state:
        data = st.session_state['data']

        # Encoding categorical features
        for col in data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

        # Splitting data into features & labels
        X = data.drop(columns=['class'], errors='ignore')
        y = data['class']

        # Feature Scaling
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Define models with some tuned parameters
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(criterion="entropy", random_state=42),
            "SVM": SVC(kernel="linear", random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights='distance'),
            "Linear Regression": LogisticRegression(max_iter=10000, random_state=42)
        }

        # Store results for comparison
        results = {}
        training_times = {}

        # Progress bar for better UX
        st.write("Training models... Please wait.")
        progress_bar = st.progress(0)
        total_models = len(models)
        
        for i, (name, model) in enumerate(models.items()):
            start_time = time.time()
            
            if name == "Linear Regression":
                model.fit(X_train, y_train)
                y_pred_cont = model.predict(X_test)
                y_pred = (y_pred_cont >= 0.5).astype(int)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            training_time = time.time() - start_time

            results[name] = {
                "Accuracy": acc,
                "F1 Score": f1,
                "Training Time (s)": training_time
            }
            training_times[name] = training_time

            st.subheader(f"Model: {name}")
            st.write(f"✅ Accuracy: {acc:.4f}")
            st.write(f"✅ F1 Score: {f1:.4f}")
            st.write(f"⏱️ Training Time: {training_time:.2f} seconds")
            st.text("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))
            st.text("Classification Report:")
            st.write(classification_report(y_test, y_pred))

            progress_bar.progress((i + 1) / total_models)

        # Model Performance Comparison
        st.subheader("📊 Model Performance Comparison")
        results_df = pd.DataFrame(results).T
        st.write(results_df)

        fig, ax = plt.subplots(figsize=(12, 6))
        results_df[["Accuracy", "F1 Score"]].plot(kind="bar", ax=ax, color=['#1f77b4', '#ff7f0e'])
        plt.xticks(rotation=45, ha='right')
        plt.title("Accuracy & F1 Score of Models", fontsize=14, pad=10)
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("⏱️ Training Time Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=list(training_times.keys()), y=list(training_times.values()), palette='viridis', ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.title("Training Time of Models (seconds)", fontsize=14, pad=10)
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Time (s)", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

        best_model = results_df['F1 Score'].idxmax()
        st.success(f"🏆 Best Performing Model (by F1 Score): **{best_model}** with F1 Score: {results_df.loc[best_model, 'F1 Score']:.4f}")

    else:
        st.warning("⚠️ Please upload datasets first!")

# ===================== REAL-TIME DATA FILTERING (NEW PAGE) ====================== #
elif page == "Real-Time Data Filtering":
    st.header("🔍 Real-Time Data Filtering")

    if 'data' in st.session_state:
        data = st.session_state['data']

        # Column Selection for Display
        st.subheader("Select Columns to Display")
        columns_to_show = st.multiselect(
            "Choose columns",
            options=data.columns,
            default=data.columns[:5].tolist(),
            key="real_time_columns"
        )
        
        # Numeric Column Filtering
        st.subheader("Filter by Numeric Values")
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        filter_col = st.selectbox(
            "Select a numeric column to filter",
            options=numeric_cols,
            key="real_time_filter_col"
        )
        
        if filter_col:
            min_val, max_val = float(data[filter_col].min()), float(data[filter_col].max())
            range_vals = st.slider(
                f"Filter {filter_col} range",
                min_val,
                max_val,
                (min_val, max_val),
                key="real_time_range"
            )
            filtered_data = data[(data[filter_col] >= range_vals[0]) & (data[filter_col] <= range_vals[1])]
        else:
            filtered_data = data  # No filtering if no column selected

        # Categorical Column Filtering
        st.subheader("Filter by Categorical Values")
        cat_cols = data.select_dtypes(include=['object']).columns
        cat_filter_col = st.selectbox(
            "Select a categorical column to filter",
            options=["None"] + cat_cols.tolist(),
            key="real_time_cat_col"
        )
        
        if cat_filter_col != "None":
            cat_values = st.multiselect(
                f"Select values for {cat_filter_col}",
                options=data[cat_filter_col].unique(),
                default=data[cat_filter_col].unique()[:2].tolist(),
                key="real_time_cat_values"
            )
            if cat_values:
                filtered_data = filtered_data[filtered_data[cat_filter_col].isin(cat_values)]

        # Display Filtered Data
        st.subheader("Filtered Dataset Preview")
        if columns_to_show:
            st.dataframe(filtered_data[columns_to_show].head(10))  # Show first 10 rows for performance
        else:
            st.write("Please select at least one column to display.")

        # Optional: Simple Visualization
        st.subheader("Quick Visualization")
        if filter_col and columns_to_show:
            fig, ax = plt.subplots(figsize=(6, 4))
            filtered_data[filter_col].hist(bins=20, edgecolor='black', ax=ax)
            plt.title(f"Histogram of {filter_col}")
            st.pyplot(fig)

    else:
        st.warning("⚠️ Please upload datasets first!")

