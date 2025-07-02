# Standard Libraries
import io
import time
import pickle
import joblib
import tempfile
import warnings

# Core Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit
import streamlit as st
import streamlit.components.v1 as components

# Visualization & Profiling
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sweetviz as sv
import ppscore as pps

# Scikit-learn - Model Selection
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV
)

# Scikit-learn - Classification Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Scikit-learn - Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Third-Party Regressors & Classifiers
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb

# Scikit-learn - Metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score
)

# Scikit-learn - Preprocessing & Utilities
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    StandardScaler
)
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.exceptions import FitFailedWarning


import warnings
warnings.filterwarnings("ignore")

st.title("🧠 Unified Machine Learning App")
st.markdown("<h3 style='text-align: center;'>📈Classification & 📉Regression</h3>", unsafe_allow_html=True)


#data= st.file_uploader("upload file",type=['csv','xlsx'])

#EXPLORATORY DATA ANALYSIS
def eda(data):

    col1, col2 = st.columns([2,1])

    #Duplicates
    def drop_dup(d):
        d.drop_duplicates(inplace=True)
        return d

   


    #Null values
    def null(d):

        if d[d.isnull().any(axis=1)].shape[0] == 0:
            st.success("NO NULL Values")
        else:
            st.subheader('⚠️ Null values found')
            st.write(d[d.isnull().any(axis=1)])
            a=st.sidebar.radio('How to handle Null',['Fill NA','DROP NA'])
            
            if a =='Fill NA':
                for i in d.columns:
                    if d[i].isnull().sum() > 0:  # Only process columns that have nulls
                        if d[i].dtype == 'object':
                            d[i] = d[i].fillna(d[i].mode()[0])
                        else:
                            d[i] = d[i].fillna(d[i].mean())
                st.success('✅ Filled Null values')
            else:
                d.dropna(inplace=True)
                st.success(' 🗑️Droped Null values')

        return d

    #Label Encoding
    def labelencoding(d):
        le = LabelEncoder()

        for i in d.select_dtypes(['object']).columns:
            d[i]=le.fit_transform(d[i].astype(str))

        st.success("✅ Label encoding applied to object columns")
        return d

    # Rename columns
    def rename(d):
        st.subheader("🔤 Rename Columns")
        old_names = st.multiselect("Select columns to rename", list(d.columns))
        new_names = []

        for i, old in enumerate(old_names):
            new_name = st.text_input(f"Rename '{old}' to:", key=f"rename_{i}")
            new_names.append(new_name)

        if st.button("Apply Renaming"):
            if len(old_names) == len(new_names):
                rename_dict = dict(zip(old_names, new_names))
                d.rename(columns=rename_dict, inplace=True)
                st.success("✅ Columns renamed successfully")
            else:
                st.warning("Number of new names must match selected columns")

        return d
#Datatypes
    def change_column_dtype(df,key_prefix=""):
       

        col = st.sidebar.selectbox("Column", df.columns,key=f"{key_prefix}_col")
        dtype = st.sidebar.selectbox("Type", ["object", "int64", "float64", "bool", "datetime64"],key=f"{key_prefix}_dtype")
        if st.button("Convert",key=f"{key_prefix}_convert"):
            try:
                df[col] = pd.to_datetime(df[col]) if dtype == "datetime64" else df[col].astype(dtype)
                st.success("Converted dtypes")
            except Exception as e:
                st.error(e)
        return df
    

    def split(d):
        target_col = st.selectbox("🎯 Select Target (Y) Column", d.columns,key="target_split")
        if target_col:
                # Step 3: Split DataFrame
                y = d[[target_col]]
                x = d.drop(columns=[target_col])

                # Step 4: Show DataFrames
                st.subheader("📂 X (Features)")
                st.dataframe(x)

                st.subheader("🎯 Y (Target)")
                st.dataframe(y)

                # Step 5: Download Buttons
                x_csv = x.to_csv(index=True).encode('utf-8')
                y_csv = y.to_csv(index=False).encode('utf-8')

                st.download_button("📥 Download X.csv", x_csv, "X.csv", "text/csv")
                st.download_button("📥 Download Y.csv", y_csv, "Y.csv", "text/csv")
        return x,y


    def balance_data(d, target_col):
        classes = d[target_col].value_counts().index
        
        min_count = d[target_col].value_counts().min()
        d_balanced = pd.concat([
            resample(d[d[target_col] == cls], replace=True, n_samples=min_count, random_state=42)
            for cls in classes
        ])
        return d_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    def datavizz(d):
        st.subheader("📊 Advanced Data Visualization")

        plot_group = st.selectbox("Select Plot Category", [
            "📊 Basic Plots", 
            "📈 Trend & Comparison", 
            "🧠 Advanced Distributions", 
            "🧪 Experimental"
        ])

        if plot_group == "📊 Basic Plots":
            plot_type = st.selectbox("Choose Plot Type", [
                "Histogram", 
                "Boxplot", 
                "Countplot (for categorical)", 
                "Scatterplot",
                "Pairplot",
                "Heatmap (Correlation)"
            ])

            if plot_type == "Histogram":
                col = st.selectbox("Select column", d.select_dtypes(include=['int64', 'float64']).columns)
                fig, ax = plt.subplots()
                sns.histplot(d[col], kde=True, ax=ax)
                st.pyplot(fig)

            elif plot_type == "Boxplot":
                col = st.selectbox("Select column", d.select_dtypes(include=['int64', 'float64']).columns)
                fig, ax = plt.subplots()
                sns.boxplot(data=d, y=col, ax=ax)
                st.pyplot(fig)

            elif plot_type == "Countplot (for categorical)":
                col = st.selectbox("Select column", d.select_dtypes(include='object').columns)
                fig, ax = plt.subplots()
                sns.countplot(data=d, x=col, order=d[col].value_counts().index[:20], ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

            elif plot_type == "Scatterplot":
                num_cols = d.select_dtypes(include=['int64', 'float64']).columns
                col1 = st.selectbox("X-axis", num_cols, key='scatter_x')
                col2 = st.selectbox("Y-axis", num_cols, key='scatter_y')
                fig, ax = plt.subplots()
                sns.scatterplot(data=d, x=col1, y=col2, ax=ax)
                st.pyplot(fig)

            elif plot_type == "Pairplot":
                st.info("⚠️ May be slow for large datasets. Select 2+ columns.")
                num_cols = d.select_dtypes(include=['int64', 'float64']).columns.tolist()
                selected_cols = st.multiselect("Select numeric columns", num_cols, default=num_cols[:4])
                if len(selected_cols) >= 2:
                    fig = sns.pairplot(d[selected_cols])
                    st.pyplot(fig)
                else:
                    st.warning("Select at least 2 columns for pairplot.")

            elif plot_type == "Heatmap (Correlation)":
                fig, ax = plt.subplots()
                sns.heatmap(d.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)

        elif plot_group == "📈 Trend & Comparison":
            plot_type = st.selectbox("Choose Plot Type", ["Lineplot", "Barplot"])

            if plot_type == "Lineplot":
                num_cols = d.select_dtypes(include=['int64', 'float64']).columns
                x_col = st.selectbox("X-axis", d.columns)
                y_col = st.selectbox("Y-axis", num_cols)
                fig, ax = plt.subplots()
                sns.lineplot(data=d, x=x_col, y=y_col, ax=ax)
                st.pyplot(fig)

            elif plot_type == "Barplot":
                col = st.selectbox("Select categorical column", d.select_dtypes(include='object').columns)
                num_col = st.selectbox("Select numeric column", d.select_dtypes(include=['int64', 'float64']).columns)
                fig, ax = plt.subplots()
                sns.barplot(data=d, x=col, y=num_col, ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

        elif plot_group == "🧠 Advanced Distributions":
            plot_type = st.selectbox("Choose Plot Type", ["Violinplot", "Swarmplot", "Stripplot", "Boxenplot"])

            cat_col = st.selectbox("Categorical Column", d.select_dtypes(include='object').columns)
            num_col = st.selectbox("Numeric Column", d.select_dtypes(include=['int64', 'float64']).columns)

            if plot_type == "Violinplot":
                fig, ax = plt.subplots()
                sns.violinplot(data=d, x=cat_col, y=num_col, ax=ax)
                st.pyplot(fig)

            elif plot_type == "Swarmplot":
                fig, ax = plt.subplots()
                sns.swarmplot(data=d, x=cat_col, y=num_col, ax=ax)
                st.pyplot(fig)

            elif plot_type == "Stripplot":
                fig, ax = plt.subplots()
                sns.stripplot(data=d, x=cat_col, y=num_col, ax=ax)
                st.pyplot(fig)

            elif plot_type == "Boxenplot":
                fig, ax = plt.subplots()
                sns.boxenplot(data=d, x=cat_col, y=num_col, ax=ax)
                st.pyplot(fig)

        elif plot_group == "🧪 Experimental":
            plot_type = st.selectbox("Choose Plot Type", ["Treemap", "Andrews Curves"])

            if plot_type == "Treemap":
                import squarify
                col = st.selectbox("Column for Treemap", d.select_dtypes(include='object').columns)
                counts = d[col].value_counts()
                fig = plt.figure()
                squarify.plot(sizes=counts.values, label=counts.index, alpha=0.8)
                plt.axis('off')
                st.pyplot(fig)

            elif plot_type == "Andrews Curves":
                from pandas.plotting import andrews_curves
                target = st.selectbox("Target column (categorical)", d.select_dtypes(include='object').columns)
                fig, ax = plt.subplots()
                andrews_curves(d, target=target, ax=ax)
                st.pyplot(fig)

    #Scalling
    def scaling(x):
        st.subheader("📐 Feature Scaling Options")
        method = st.sidebar.selectbox(
            "Select a scaling method for features:",
            ["None", "Min-Max Scaling", "Standard Scaling"]
        )

        if method == 'Min-Max Scaling':
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(x)
            st.success("✅ Min-Max Scaling applied.")
            return pd.DataFrame(x_scaled, columns=x.columns, index=x.index), scaler

        elif method == 'Standard Scaling':
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            st.success("✅ Standard Scaling applied.")
            return pd.DataFrame(x_scaled, columns=x.columns, index=x.index), scaler

        else:
            st.info("ℹ️ No scaling applied.")
            return x, None

#pps score
    def show_pps(d):
        d1=d.copy()
        st.sidebar.subheader("🔍 Predictive Power Score (PPS)")
        target_col = st.sidebar.selectbox("🎯 Select Target/ Y Column", d1.columns)
       
        if target_col:

            pps_df = pps.matrix(d1)
            filtered_pps = pps_df[pps_df["y"] == target_col][["x", "ppscore"]]
            st.sidebar.dataframe(filtered_pps.sort_values(by="ppscore", ascending=False).reset_index(drop=True))


# Drop columns
    def drop_columns(d):
        st.subheader("Drop Unwanted Columns")
        
        cols_to_drop = st.multiselect("Select columns to drop", options=d.columns)

        if cols_to_drop:
            d_dropped = d.drop(columns=cols_to_drop)
            st.success(f"✅ Dropped columns: {', '.join(cols_to_drop)}")
            st.subheader("Updated Data")
            st.dataframe(d_dropped)

            # Optional: Return cleaned data
            return d_dropped
        else:
            return d

    df=data.copy()

    st.sidebar.subheader("Additional EDA process")  
    fig, ax = plt.subplots()
    sns.barplot(df,ax=ax)
    st.pyplot(fig)

    with col1:
        st.write("📄 Raw Data")
        st.write(data)
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_string = buffer.getvalue()
        st.code(info_string)

    st.write("📊Data Description")
    st.write(df.describe())
    #Null values
    null(df)
   

 
#Datatypes
    if st.sidebar.toggle("Change Dtypes", key="change_dtypes_toggle"):
        df=change_column_dtype(df, key_prefix="toggle")


    #labelEncoding
    if st.sidebar.toggle('Label Encode'):                         
        df=labelencoding(df)

    #RENAME
    if st.sidebar.toggle("rename"):
        df=rename(df)

    
    if st.sidebar.toggle("Scale Data"):
        df, scaler = scaling(df)

    show_pps(df)
    #drop columns
    df=drop_columns(df)

    st.write("Final Data")
    st.write(df)

   
    #Data visualization
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    if st.sidebar.toggle("📈 Univariate Plot"):
        column = st.selectbox("Select column", df.columns)
        if df[column].dtype != 'object':
            st.bar_chart(df[column].value_counts())
        else:
            st.bar_chart(df[column].value_counts().head(20))

    

    

    if st.sidebar.toggle("⚖️ Balance Dataset"):
        target_col = st.selectbox("Select Target Column", df.columns)
        df = balance_data(df, target_col)
        st.success("✅ Dataset balanced")
        st.bar_chart(df[target_col].value_counts())
    
    with tab3:
        datavizz(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📁 Download Final Processed Data", csv, "processed_data.csv", "text/csv")

    
    split(df)


    if st.button('Report'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            report = sv.analyze(df)
            report.show_html(f.name)
            st.components.v1.html(open(f.name, 'r', encoding='utf-8').read(), height=1000, scrolling=True)


    return df

def apply_pca(data):
    st.subheader("🔍 PCA - Principal Component Analysis")

    # Optional: Scale data
    scale = st.checkbox("Scale data before PCA", value=True)
    X = StandardScaler().fit_transform(data) if scale else data.copy()

    # Select number of components
    n_components = st.slider("Select number of components", 2, min(len(data.columns), 10), value=2)

    # Run PCA
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(X)

    joblib.dump(pca_data, "pca_model.pkl")
    with open("pca_model.pkl", "rb") as f:
        st.download_button("📥 Download PCA Model", f, file_name="pca_model.pkl")

    # Explained variance plot
    st.write("🔎 Explained Variance Ratio:")
    fig, ax = plt.subplots()
    ax.plot(range(1, n_components + 1), pca.explained_variance_ratio_.cumsum(), marker='o')
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Explained Variance")
    st.pyplot(fig)

    # Show transformed data (first 2 components)
    if n_components >= 2:
        df_pca = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(n_components)])
        st.write("📉 PCA 2D Projection")
        fig2, ax2 = plt.subplots()
        ax2.scatter(df_pca["PC1"], df_pca["PC2"], alpha=0.7)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        st.pyplot(fig2)

    return pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(n_components)])



#CLASSIFICATION
def classification(x,y):

    best_model = None
    st.subheader("📈 Classification")


    #logistic ression
    def logistic_r (train_x,train_y,test_x,test_y):
        le = LogisticRegression()
        model_lr = le.fit(train_x,train_y)

        lr_train_predict = model_lr.predict(train_x)
        lr_test_predict = model_lr.predict(test_x)

        lr_train_acc = accuracy_score(train_y,lr_train_predict)*100
        lr_test_acc = accuracy_score(test_y,lr_test_predict)*100

        return lr_train_acc,lr_test_acc,model_lr

    #Random forest
    def random_forest (train_x, train_y,test_x,test_y):
        kfold = KFold(n_splits=10, random_state=5,shuffle=True)
        n_estimators = np.array(range(10,50)) 
        max_feature = [2,3,4,5,6]
        param_grid = dict(n_estimators =n_estimators,max_features=max_feature)

        model_rfc = RandomForestClassifier()
        grid_rfc = GridSearchCV(estimator=model_rfc, param_grid=param_grid)
        grid_rfc.fit(train_x, train_y)

        RFC_Model = RandomForestClassifier(n_estimators=grid_rfc.best_params_['n_estimators'],max_features=grid_rfc.best_params_['max_features'])
        RFC_Model.fit(train_x,train_y)

        RFC_train_predict = RFC_Model.predict(train_x)
        RFC_test_predict = RFC_Model.predict(test_x)

        rfc_train_acc = accuracy_score(train_y,RFC_train_predict)*100
        rfc_test_acc = accuracy_score(test_y,RFC_test_predict)*100

        return rfc_train_acc,rfc_test_acc, model_rfc

    #support vector clasifer

    #Support Vector clasiffiers
    def svc(train_x,train_y,test_x,test_y):


        clf = SVC()
        param_grid_svc = [{'kernel':['rbf','sigmoid','poly'],'gamma':[0.5,0.1,0.005],'C':[25,20,10,0.1,0.001] }]
        
        # Determine the minimum number of samples in any class
        train_y_series = pd.Series(train_y)

        # Safely compute number of samples per class
        min_class_samples = train_y_series.value_counts().min()
        safe_cv = min(5, min_class_samples)

        # Warn if dataset is too small
        if safe_cv < 2:
            raise ValueError("Not enough samples in some classes for cross-validation.")

        # Avoid warning spam from failed fits
        warnings.simplefilter('ignore', FitFailedWarning)
        
        svc= RandomizedSearchCV(clf,param_grid_svc,cv=safe_cv)
        svc.fit(train_x,train_y)

        svc_train_predict = svc.predict(train_x)
        svc_test_predict = svc.predict(test_x)

        svc_train_acc = accuracy_score(train_y,svc_train_predict)*100
        svc_test_acc = accuracy_score(test_y,svc_test_predict)*100

        return svc_train_acc,svc_test_acc,svc

    #bagging
    def bagging(train_x,train_y,test_x,test_y):
        cart = DecisionTreeClassifier()

        model_bag = BaggingClassifier(estimator=cart, n_estimators= 10, random_state=6)
        model_bag.fit(train_x,train_y)

        bag_train_predict = model_bag.predict(train_x)
        bag_test_predict = model_bag.predict(test_x)

        bag_train_acc = accuracy_score(train_y,bag_train_predict)*100
        bag_test_acc = accuracy_score(test_y,bag_test_predict)*100

        return bag_train_acc,bag_test_acc,model_bag

    #xgb
    def xgb(train_x,train_y,test_x,test_y):
        n_estimators =np.array(range(10,80,10))
        xgb_model = XGBClassifier(n_estimators=70,max_depth=5)
        xgb_model.fit(train_x,train_y)

        xgb_train_predict = xgb_model.predict(train_x)
        xgb_test_predict = xgb_model.predict(test_x)

        xgb_train_acc = accuracy_score(train_y,xgb_train_predict)*100
        xgb_test_acc = accuracy_score(test_y,xgb_test_predict)*100

        return xgb_train_acc,xgb_test_acc,xgb_model

    #LGBM
    def lgbm(train_x,train_y,test_x,test_y):

        params = {}
        params['learning_rate'] = 1
        params['boosting_type'] = 'gbdt'
        params['objective'] = 'binary'
        params['metric'] = 'binary_logloss'
        params['sub_feature'] = 0.5
        params['num_leaves'] = 5
        params['min_data'] = 10
        params['max_depth'] = 5

        lgbm_model = lgb.LGBMClassifier()
        lgbm_model.fit(train_x,train_y)

        lgbm_train_predict = lgbm_model.predict(train_x)
        lgbm_test_predict = lgbm_model.predict(test_x)

        lgbm_train_acc = accuracy_score(train_y,lgbm_train_predict)*100
        lgbm_test_acc = accuracy_score(test_y,lgbm_test_predict)*100

        return lgbm_train_acc,lgbm_test_acc,lgbm_model

    #NaiveByaes
    def NB(train_x,train_y,test_x,test_y):
        nb_model = GaussianNB()
        nb_model.fit(train_x, train_y)

        nb_train_predict=nb_model.predict(train_x)
        nb_test_predict=nb_model.predict(test_x)

        nb_train_acc = accuracy_score(train_y,nb_train_predict)*100
        nb_test_acc = accuracy_score(test_y,nb_test_predict)*100

        return nb_train_acc,nb_test_acc,nb_model


    #KNN
    def knn(train_x,train_y,test_x,test_y):

        n_neighbors = np.array(range(2,30))
        param_grid = dict(n_neighbors=n_neighbors)

        model = KNeighborsClassifier()
        grid = GridSearchCV(estimator=model, param_grid=param_grid)
        grid.fit(train_x, train_y)

        knn_model = KNeighborsClassifier(grid.best_params_['n_neighbors'])
        knn_model.fit(train_x, train_y)

        knn_train_predict=knn_model.predict(train_x)
        knn_test_predict=knn_model.predict(test_x)

        knn_train_acc = accuracy_score(train_y,knn_train_predict)*100
        knn_test_acc = accuracy_score(test_y,knn_test_predict)*100

        return knn_train_acc,knn_test_acc,knn_model

    #Decision Tree
    def decision_tree(train_x, train_y, test_x, test_y):

        criterion_choice = st.selectbox("Decision Tree Criterion", options=["gini", "entropy"], index=0)

        if criterion_choice == 'gini':
            dt_model = DecisionTreeClassifier(criterion='gini', random_state=42)
            dt_model.fit(train_x, train_y)
        return accuracy_score(train_y, dt_model.predict(train_x)) * 100, accuracy_score(test_y, dt_model.predict(test_x)) * 100, dt_model


    def df(train_x,train_y,test_x,test_y):

        list= [logistic_r (train_x,train_y,test_x,test_y), 
        random_forest (train_x, train_y,test_x,test_y),
        svc(train_x,train_y,test_x,test_y),
        bagging(train_x,train_y,test_x,test_y),
        xgb(train_x,train_y,test_x,test_y),
        lgbm(train_x,train_y,test_x,test_y),
        NB(train_x,train_y,test_x,test_y),
        knn(train_x,train_y,test_x,test_y),
        decision_tree(train_x, train_y, test_x, test_y) ]

        acc_data = pd.DataFrame(list,columns=('Train accuracy','Test accuracy','Model'),index=['logistic','Random_forest','SVC','Bagging','XGB','LGBM','NB',"KNN","Decission Tree"])

        return acc_data

    data = None
    best_model = None

    
        

    start_time= time.time()

    if x is not None and y is not None:
        

        
        st.dataframe(x.head(5))
        st.dataframe(y.head(5))

        st.info("📞 Use PCA for better functionality and efficiency in modeling.")

        if st.checkbox('🚠Apply PCA'):
            x= apply_pca(x)


        train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=10)
        le = LabelEncoder()
        train_y = le.fit_transform(train_y.values.ravel())
        try:
            test_y = le.transform(test_y.values.ravel())
        except ValueError as e:
                st.warning("⚠️ Your target variable contains only unique values.")
                st.info("💡 Classification requires repeating labels. Please check if this column is suitable as a target.")
                st.stop()


        data = df(train_x,train_y,test_x,test_y)

        st.dataframe(data)
        


## DOWNLOAD OF DIFFRENT MODELS
        st.sidebar.header('Download required model for Deployment')
        for model_name in data.index:
            model = data.loc[model_name, 'Model']
            file_name = f"{model_name}_model.pkl"
            with open(file_name, "wb") as f:
                pickle.dump(model, f) 

            with open(file_name, "rb") as f:
                
                st.sidebar.download_button(
                    label=f"📥 Download {model_name.capitalize()} Model",
                    data=f.read(),
                    file_name=file_name,
                    mime="application/octet-stream",key=f"download_{model_name}"
                ) 


# 🎯 Simple Best Classification Model Selector
        st.subheader("Best Classification Model")

        if 'Test accuracy' in data.columns:
            best_row = data.sort_values(by='Test accuracy', ascending=False).iloc[0]
            st.success(f"🏆 Best model based on Test Accuracy: **{best_row.name}**")
            st.write(best_row)
            best_model = best_row['Model']
        else:
            st.warning("⚠️ 'Test Accuracy' column not found in results.")
                
        end_time= time.time()

    
        time_taken = end_time-start_time

        st.success(f"Task complited in {time_taken:.2f} seconds")

        

        return data, best_model        
    

#REGRESSION
def regression(x,y):
    st.subheader("📉 Regression")

    

    # ------------------ COMMON EVALUATION FUNCTION ------------------

    def evaluate_model(model, train_x, train_y, test_x, test_y):
        train_pred = model.predict(train_x)
        test_pred = model.predict(test_x)

        train_rmse = np.sqrt(mean_squared_error(train_y, train_pred))
        test_rmse = np.sqrt(mean_squared_error(test_y, test_pred))

        train_r2 = r2_score(train_y, train_pred)
        test_r2 = r2_score(test_y, test_pred)

        return train_rmse, test_rmse, train_r2, test_r2, model

    # ------------------ REGRESSION MODEL FUNCTIONS WITH GRID SEARCH ------------------

    def grid_search_model(model_class, param_grid, train_x, train_y, test_x, test_y):
        grid = GridSearchCV(model_class(), param_grid, cv=5)
        grid.fit(train_x, train_y)
        best_model = grid.best_estimator_
        return evaluate_model(best_model, train_x, train_y, test_x, test_y)

    def linear_regression(train_x, train_y, test_x, test_y):
        model = LinearRegression()
        model.fit(train_x, train_y)
        return evaluate_model(model, train_x, train_y, test_x, test_y)

    def ridge_regression(train_x, train_y, test_x, test_y):
        param_grid = {'alpha': np.linspace(0.01, 10.0, 10)}
        return grid_search_model(Ridge, param_grid, train_x, train_y, test_x, test_y)

    def lasso_regression(train_x, train_y, test_x, test_y):
        param_grid = {'alpha': np.linspace(0.01, 1.0, 10)}
        return grid_search_model(Lasso, param_grid, train_x, train_y, test_x, test_y)

    def decision_tree_reg(train_x, train_y, test_x, test_y):
        param_grid = {'max_depth': list(range(3, 16))}
        return grid_search_model(DecisionTreeRegressor, param_grid, train_x, train_y, test_x, test_y)

    def random_forest_reg(train_x, train_y, test_x, test_y):
        param_grid = {'n_estimators': list(range(50, 151, 25))}
        return grid_search_model(RandomForestRegressor, param_grid, train_x, train_y, test_x, test_y)

    def xgb_reg(train_x, train_y, test_x, test_y):
        param_grid = {'n_estimators': list(range(50, 151, 25)), 'max_depth': list(range(3, 7))}
        return grid_search_model(XGBRegressor, param_grid, train_x, train_y, test_x, test_y)

    def lgbm_reg(train_x, train_y, test_x, test_y):
        param_grid = {'num_leaves': list(range(20, 61, 10)), 'max_depth': list(range(-1, 16, 5))}
        return grid_search_model(lgb.LGBMRegressor, param_grid, train_x, train_y, test_x, test_y)

    def knn_reg(train_x, train_y, test_x, test_y):
        param_grid = {'n_neighbors': list(range(3, 11))}
        return grid_search_model(KNeighborsRegressor, param_grid, train_x, train_y, test_x, test_y)

    def elastic_net_reg(train_x, train_y, test_x, test_y):
        param_grid = {
            'alpha': np.linspace(0.01, 1.0, 10),
            'l1_ratio': np.linspace(0.1, 0.9, 5)}
        return grid_search_model(ElasticNet, param_grid, train_x, train_y, test_x, test_y)
    
    def svr_reg(train_x, train_y, test_x, test_y):
        param_grid = {
            'C': [0.1, 1, 10],
            'epsilon': [0.01, 0.1, 1],
            'kernel': ['rbf', 'linear']}
        return grid_search_model(SVR, param_grid, train_x, train_y, test_x, test_y)


    def drop_columns(d):
        st.subheader("Drop Unwanted Columns")
        
        cols_to_drop = st.multiselect("Select columns to drop", options=d.columns)

        if st.button("Drop Selected Columns"):
            d_dropped = d.drop(columns=cols_to_drop)
            st.success(f"✅ Dropped columns: {', '.join(cols_to_drop)}")
            st.subheader("Updated Data")
            st.dataframe(d_dropped)

            # Optional: Return cleaned data
            return d_dropped
        else:
            return d

    # ------------------ DATAFRAME GENERATOR ------------------

    def df_regression(train_x, train_y, test_x, test_y):
        models = [
            linear_regression,
            ridge_regression,
            lasso_regression,
            decision_tree_reg,
            random_forest_reg,
            xgb_reg,
            lgbm_reg,
            knn_reg,
            elastic_net_reg, 
            svr_reg
        ]
        results = []
        names = ['Linear', 'Ridge', 'Lasso', 'DecisionTree', 'RandomForest', 'XGB', 'LGBM', 'KNN', "ElasticNet",'SVR']

        for model_func in models:
            result = model_func(train_x, train_y, test_x, test_y)
            results.append(result)

        df_result = pd.DataFrame(results, columns=["Train RMSE", "Test RMSE", "Train R2", "Test R2", "Model"], index=names)
        return df_result

    # ------------------ MAIN LOGIC ------------------


    start_time = time.time()

    if x is not None and y is not None:
        

        st.dataframe(x.head())
        st.dataframe(y.head())


        st.info("📞 Use PCA for better functionality and efficiency in modeling.")

        if st.checkbox('🚠Apply PCA'):
            x= apply_pca(x)

        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

        data = df_regression(train_x, train_y.values.ravel(), test_x, test_y.values.ravel())
        st.dataframe(data)
        

        # DOWNLOAD MODELS
        st.sidebar.header("Download Trained Regression Models")
        for model_name in data.index:
            model = data.loc[model_name, 'Model']
            filename = f"{model_name}_reg_model.pkl"

            with open(filename, "wb") as f:
                pickle.dump(model, f)
            with open(filename, "rb") as f:
                st.sidebar.download_button(
                    label=f"📥 Download {model_name} Model",
                    data=f.read(),
                    file_name=filename,
                    mime="application/octet-stream",
                    key=f"download_{model_name}"
                )

        time_taken = time.time() - start_time
        st.success(f"Task completed in {time_taken:.2f} seconds")

        
        metric_choice = st.radio("Select metric to choose best model:", ['Test RMSE', 'Test R2'], horizontal=True)

        if metric_choice == 'Test RMSE':
            best_model_row = data.sort_values(by='Test RMSE').iloc[0]
        else:
            best_model_row = data.sort_values(by='Test R2', ascending=False).iloc[0]

        best_model_name = best_model_row.name
        best_model = best_model_row['Model'] 
        st.success(f"🏆 Best Model Based on {metric_choice}: **{best_model_name}**")
        st.write(f"✅ Test RMSE: {best_model_row['Test RMSE']:.4f}")
        st.write(f"✅ Test R²: {best_model_row['Test R2']:.4f}")
        


    return data,best_model

#VISUALIZE
def visualize_data(df):
    st.subheader("📊 Data Visualization")
    
    if df is None or df.empty:
        st.warning("No data available for visualization.")
        return

    st.markdown("### 📌 Choose Column(s) to Visualize")
    columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    col = st.selectbox("Select column for univariate plots", columns)

    if col in numeric_cols:
        st.write("#### Histogram")
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

        st.write("#### Box Plot")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)

    elif col in categorical_cols:
        st.write("#### Count Plot")
        fig, ax = plt.subplots()
        sns.countplot(x=col, data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Pairplot for numeric columns
    if len(numeric_cols) >= 2:
        st.write("#### Pairplot")
        fig = sns.pairplot(df[numeric_cols])
        st.pyplot(fig)

    # Correlation heatmap
    st.write("#### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Scatter plot
    st.write("#### Scatter Plot")
    x_axis = st.selectbox("X-axis", numeric_cols, key="scatter_x")
    y_axis = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
    color = st.selectbox("Color (optional)", categorical_cols + [None], index=0)
    fig = px.scatter(df, x=x_axis, y=y_axis, color=color)
    st.plotly_chart(fig)

    st.success("✅ Visualizations loaded.")




# Initialize session state variables
if "df" not in st.session_state:
    st.session_state.df = None
if "target" not in st.session_state:
    st.session_state.target = None
if "task" not in st.session_state:
    st.session_state.task = None
if "result" not in st.session_state:
    st.session_state.result = None
if "best_model" not in st.session_state:
    st.session_state.best_model = None

# App layout
tab1, tab2, tab3 = st.tabs(["\U0001F4CA EDA", "\U0001F9E0 Modeling", "\U0001F4C8 Visualization"])

with tab1:
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file, encoding="latin1")
        else:
            excel_file = pd.ExcelFile(uploaded_file, engine='openpyxl')
            if len(excel_file.sheet_names) > 1:
                sheet = st.radio('Select sheet', excel_file.sheet_names)
                df = pd.read_excel(excel_file, sheet_name=sheet)
            else:
                df = pd.read_excel(uploaded_file)

        st.session_state.df = df
        st.header("\U0001F9EE Exploratory Data Analysis (EDA)")

        if len(df) > 5000:
            st.warning(f"Dataset has {len(df)} rows. Showing EDA on 2000-row sample for speed.")

            df=df.sample(n=2000, random_state=42)
  
        df = eda(df)
        #df = eda(df)
        st.session_state.df = df 
        st.session_state.target = st.selectbox("\U0001F3AF Select Target Column", df.columns)

with tab2:
    if st.session_state.df is not None and st.session_state.target is not None:
        df = st.session_state.df
        target = st.session_state.target

        x = df.drop(columns=[target])
        y = df[[target]]

        task = st.radio("\U0001F4CC Task Type", ["-- Select Task --","Classification", "Regression"])
        st.session_state.task = task

        
        if task == "Classification":
            result = classification(x, y)  # This function does not need to use session_state
        elif task == "Regression":
            result = regression(x, y)
        else:
            result = None

        if result is not None:
            st.session_state.result, st.session_state.best_model = result
        elif task in ["Classification", "Regression"]:
            st.session_state.result, st.session_state.best_model = None, None
        else:
            st.info("Please select a task type to continue.")

   # {if st.session_state.result is not None:
       # st.write("### 📝 Model Results")
        #st.dataframe(st.session_state.result)

        #st.write("### 🏆 Best Model")
       # st.write(st.session_state.best_model)}


st.markdown(
    """
    <hr>
    <div style="text-align: center; font-size: 14px; color: gray;">
        Made with ❤️ by <a href="https://github.com/suradayakarreddy" target="_blank">DAYANREDDY</a> using Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
