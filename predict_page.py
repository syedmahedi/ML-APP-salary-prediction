import  streamlit as st 
import pickle
import numpy as np



def load_model():
    with open("saved_steps.pkl","rb") as file:
        data=pickle.load(file)
    return data

data=load_model()

regressor=data['model']
le_country=data["le_country"]
le_education=data["le_education"]


def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict your salary""")



    countries=(

        'United States',        
        'India'                
        'United Kingdom',        
        'Germany',               
        'Canada',             
        'Brazil',                 
        'France',                 
        'Spain',                  
        'Australia',             
        'Netherlands',            
        'Poland',                 
        'Italy',                  
        'Russian Federation',    
        'Sweden',       
    )



    countries=st.selectbox("Country",countries)


    education=(
        'Bachelor’s degree',
        'Master’s degree',
        'Post grad',
        'Less than a Bachelors',
    )


    education=st.selectbox("Education Level",education)

    experience=st.slider("Years of Experience",0,50,3)

    push=st.button("Submit")

    if push:
        x=np.array([[countries,education,experience]])
        x[:,0]=le_country.transform(x[:,0])
        x[:,1]=le_education.transform(x[:,1])
        x=x.astype(float)

        salary=regressor.predict(x)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")

