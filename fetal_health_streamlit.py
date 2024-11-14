import streamlit as st
import pandas as pd
import pickle
import warnings
import numpy as np
warnings.filterwarnings('ignore')

st.title('Fetal Health Classification: A Machine Learning App')
st.image('fetal_health_image.gif', use_column_width = True, 
         caption = "Utilize our advanced Machine Learning appliation to predict local health classification")

@st.cache_resource
def load_model():
    model_pickle = open('decision_tree_pickle.pkl', 'rb') 
    dt_model = pickle.load(model_pickle) 
    model_pickle.close()

    model_pickle = open('random_forest_pickle.pkl', 'rb') 
    rf_model = pickle.load(model_pickle) 
    model_pickle.close()

    model_pickle = open('adaBoost_pickle.pkl', 'rb') 
    ada_model = pickle.load(model_pickle) 
    model_pickle.close()

    model_pickle = open('softVoting_pickle.pkl', 'rb') 
    vote_model = pickle.load(model_pickle) 
    model_pickle.close()

    return dt_model, rf_model, ada_model, vote_model

@st.cache_data
def load_data():
    default_df = pd.read_csv('fetal_health.csv')
    return(default_df)

dt_model, rf_model, ada_model, vote_model = load_model()
sample_df = load_data()

st.sidebar.header("Fetal Health Features Input")

file_upload = st.sidebar.file_uploader(label="Upload your data", help="File must be in CSV Format")
st.sidebar.warning(body = " *Ensure your data exactly matches the format outlined below*", icon = "⚠️")
st.sidebar.write(sample_df.head(5))

file_upload = pd.read_csv('fetal_health_user.csv')

if file_upload is not None:
    st.success(body = "*CSV file uploaded successfully.*", icon="✅")
    model_selection = st.sidebar.radio(label = 'Choose Model for Prediction', options = ['Random Forest',
                                                            'Decision Tree',
                                                            'ADAboost',
                                                            'Soft Voting'])
    st.sidebar.info(body = f" *You selected: {model_selection}*", icon="✅")

    if model_selection == 'Random Forest':
        clf = rf_model
        suffix = 'rf'
    elif model_selection == 'Decision Tree':
        clf = dt_model
        suffix = 'dt'
    elif model_selection == 'ADAboost':
        clf = ada_model
        suffix = 'ada'
    else:
        clf = vote_model
        suffix = 'vot'

    user_df = file_upload
    user_pred = clf.predict(user_df)
    user_pred = pd.DataFrame(user_pred)
    pred_df = pd.DataFrame(clf.predict_proba(user_df))
    pred_df['prediction_probability'] = pred_df.apply(max, axis=1) * 100
    pred_df = pred_df.round(1)
    
    def encode_fh(value):
        if value == 1:
            return "Normal"
        elif value == 2:
            return "Suspect"
        elif value == 3:
            return "Pathological"
        else:
            return "Error"

    def highlight(col):
        return ["background-color: lime" if cell == "Normal"
        else "background-color: yellow" if cell == "Suspect"
        else "background-color: orange" if cell == "Pathological"
        else ""
        for cell in col] #Used gemini for help with this function here: https://gemini.google.com/app/bb556d0fb0998353
        
    user_df['fetal_health'] = user_pred.iloc[:, 0].apply(encode_fh)
    user_df['prediction_probability'] = pd.to_numeric(pred_df['prediction_probability'])
    user_df = user_df.round({'prediction_probability': 1})
    user_df = user_df.style.apply(highlight, subset=['fetal_health']).format({'prediction_probability': '{:.1f}'}) 
    #Used gemini to help round the decimals in the display: https://gemini.google.com/app/6b363acc590a855b
    st.write(user_df)

    st.subheader("Prediction Performance")
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])
    with tab1:
        st.write("### Feature Importance")
        st.image(f'feature_imp_{suffix}.svg')
        st.caption("Relative importance of features in prediction.")
    with tab2:
        st.write("### Confusion Matrix")
        st.image(f'confusion_mat_{suffix}.svg')
        st.caption("Confusion matrix between predicted and target features.")
    with tab3:
        st.write("### Classification Report")
        class_df = pd.read_csv(f'class_report_{suffix}.csv')
        st.write(class_df.style.background_gradient(cmap='RdBu', axis=1).format(precision=2)) #Used Gemini to assist with color coding here
        st.caption("Range of predictions with confidence intervals.")
        
else:
    st.info(body = " *Please upload data to proceed.*", icon="ℹ️")
