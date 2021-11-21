from dataclasses import make_dataclass
from pandas.core.indexing import IndexSlice
import torch

from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

import streamlit as st
import altair as alt
import warnings
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle
import numpy as np

warnings.filterwarnings('ignore')
st.set_page_config(page_title='TheSeer',layout='wide',initial_sidebar_state='auto')

model_qa = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer_qa = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model_sent = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
tokenizer_sent = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

#navigation bar 
category = ("About TheSeer","Employment OutLook DashBoard","TheSeer QA System","What about me?🤔","Sentiment Analysis",)
navigation = st.sidebar.selectbox("Please Choose:", category)

if navigation == "About TheSeer":
    st.header("Welcome To TheSeer")
    st.image("datasets/OIP.jfif")
    st.markdown('''
    Data science is one of the most valuable tools one can use to change the world in terms of the 4IR.
    But how can we use data science in the future of work in the 4IR🤔? During the covid 19 period many people 
    lost their jobs, many businesses were closed, were not profitable and productive, it was really challenging.

    But with data science we can help businesses to make good decisions when it comes to customer service, 
    forecasting their finances, like their expenses and income statements and become profitable

    Data science can also help people to help people to choose a better career that will be applicable
    for the 4IR, like through chatbots and a survey that uses machine learning in the background to recommend you a suitable carrer
    for the 4IR.
    ''')

    st.subheader("With this web you can use:")
    st.write("1.Time Series Forecasting : For Forecasting Analysis")
    st.write("2.Question Answering System : For asnwering any questions related to the future of work in the 4IR")
    st.write("3.Questionnaires : Will my skills and my experiences be beneficial for the 4IR? || Do i have what is required for me to be employed?")
    st.write("4.Sentiment Analysis : Sentiment Analysis on the impact of the 4IR on employment")
    st.write("5.Dashboard : Outlook of jobs by industry in SA per province as well as forecasts on employed people per industry")

elif navigation == "TheSeer QA System":
    st.subheader("TheSeer QA System")
    st.write("Hello user😁, Ask any question regarding the 4IR and the future of work😁")

    col1_img, col2_img= st.beta_columns(2)

    with col1_img:
        pic_1 = st.image('datasets/chatbot.jpg')

    with col2_img:
        pic_2 = st.image('datasets/botlogo.jfif')

    def answer_question(question, answer_text):
        #tokenize
        input_ids = tokenizer_qa.encode(question, answer_text)
        sep_index = input_ids.index(tokenizer_qa.sep_token_id)
        num_seg_a = sep_index + 1
        num_seg_b = len(input_ids) - num_seg_a
        segment_ids = [0]*num_seg_a + [1]*num_seg_b
        assert len(segment_ids) == len(input_ids)

        outputs = model_qa(torch.tensor([input_ids]),token_type_ids=torch.tensor([segment_ids]),return_dict=True) 

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        #find the tokens with the highest scores
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)

        tokens = tokenizer_qa.convert_ids_to_tokens(input_ids)

        answer = tokens[answer_start]

        for i in range(answer_start + 1, answer_end + 1):
            if tokens[i][0:2] == '##':
                answer += tokens[i][2:]
            else:
                answer += ' ' + tokens[i]

        st.write('Answer: "' + answer + '"')

    question = st.text_input("Please type in your question here")
    answer_text = open('./datasets/answersdata.txt').read()
    answer_question(question, answer_text)

elif navigation == "Employment OutLook DashBoard":

    cat_dash = st.selectbox("Select a DashBoard",("Employment By industry DashBoard","MacroEconomics Dashboard"))

    st.header("Employment Per Province By Industry")
    st.write("Welcome to the dashboard😁")

    df = pd.read_csv("./datasets/ByProv.csv")
    print(df.head())

    st.header("Select a Province to show Employment By industry")
    prov = st.selectbox('Select Province',df['Province'][:9])

    st.header("Select The type of industry")

    industry = st.selectbox("Select Industry", 
    ("Agriculture","Mining","Manufacturing","Utilities","Construction",
    "Trade","Transport","Finance","Community and social services","Private Household"))

    st.cache(persist=True)
    def load_data():
        df_d = pd.read_csv('./datasets/ByProv.csv')
        df_d['Date'] = pd.to_datetime(df_d['Date'],format="%d-%m-%Y")
        latest = df_d[df_d['Date'] == "2021-03-30"][["Province","Agriculture","Mining","Manufacturing","Construction",
                                                "Trade","Transport","Finance","Community and social services"]]
        return df_d,latest
    df_d,latest = load_data()

    st.cache
    df = df[df["Province"]==prov]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by="Date")
    df = df[['Date',industry,"Province"]]

    indus_data = df[industry].values
    indus_data = indus_data.reshape((-1,1))

    split_tt = int(0.9* len(indus_data))

    indus_data_train = indus_data[:split_tt]
    indus_data_test = indus_data[split_tt:]

    date_train = df['Date'][:split_tt]
    date_test = df['Date'][split_tt:]

    train_generator = TimeseriesGenerator(indus_data_train, indus_data_train,length=10,batch_size=32)

    model = Sequential()
    model.add(LSTM(10,activation='relu',input_shape=(10,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit_generator(train_generator,epochs=50,verbose=1)

    indus_data = indus_data.reshape((-1))
    look_back = 10

    def predict(num_prediction, model):
        prediction_list = indus_data[-look_back:]

        for _ in range(num_prediction):
            x = prediction_list[-look_back:]
            x = x.reshape((1,look_back,1))
            output = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, output)
        prediction_list = prediction_list[look_back-1:]
        return prediction_list

    def predict_dates(num_prediction):
        last_date = df['Date'].values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
        return prediction_dates

    num_prediction = 1
    forecast = predict(num_prediction, model)
    forecast_dates = predict_dates(num_prediction)

    trace1 = go.Scatter(x = df['Date'].tolist(),y = indus_data,mode = 'lines',name = 'Data')
    trace2 = go.Scatter(x = forecast_dates,y = forecast,mode = 'lines',name = 'Prediction')
    layout = go.Layout(title = f"{industry}",xaxis = {'title' : "Date"},yaxis = {'title' : "data"},width=1000,height=600)

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    st.plotly_chart(fig)

    df_all = pd.read_csv('./datasets/EmplLatest.csv')
    st.header(f"View Employment by industry in {prov}")
    fig_prov_emp = px.pie(df_all[df_all["Province"]==prov], values='Number of People', names='Industry', title=f'Employment By industry in {prov}')
    st.plotly_chart(fig_prov_emp)

    industry_compare = st.selectbox("Select Industry", 
    ("Mining","Finance","Agriculture","Manufacturing"))

    options =st.multiselect('Select Multiple provinces', df_d['Province'][:9])
    
    fire=alt.Chart(df_d[df_d["Province"].isin(options)],width=650,height=400).mark_circle().encode(
    x="Date",
    y="Province",
    tooltip=["Date","Province",industry_compare],
    color="Province",
    size=industry_compare).interactive()

    bar1 = alt.Chart(df_d[df_d["Province"].isin(options)]).mark_bar().encode(
    y=f"sum({industry_compare})",
    x=alt.X("Province",sort="-y"),
    color="Province",
    tooltip = f"sum({industry_compare})").interactive()

    st.altair_chart(fire | bar1)

elif navigation == "What about me?🤔":

    st.header("Will my interests and skills remain in value??")
    st.markdown('''
        This section consists of two different "questionnaires":

        The first one which will classify whether your skills and experience are going to be uselful during the 4IR

        The second one which tells whether you are likely going to be employed.
    ''')

    type_ques = st.selectbox("Choose a questionnaire",("Will my skills be important during the 4IR?","Am i likey to get employed this times?"))

    if type_ques == "Am i likey to get employed this times?":

        st.subheader("Please answer this questionnaire honestly")

        sex = {"Male":1,"Female":2}

        prov =  {"Western Cape":1,"Eastern Cape":2,"Northen Cape":3,
        "Free State":4,"Kwa Zulu Natal":5,"North West":6,"Gauteng":7,"Mpumalanga":8,"Limpopo":9}

        geo_type = {"Urban Formal":1,"Urban Informal":2,"Tribal Areas":4,"Rural Formal":5}

        edu_status = {"Not Schooling":1,"Less than primary completed":2,"Primary completed":3,
            "Secondary not completed":4,"Secondary completed":5,"Tertiary":6,"Other":7}

        edu_lvl = {"Grade R":0,"Grade 2":2,"Grade 3":3,"Grade 4":4,"Grade 5":5,"Grade 6":6,
            "Grade 7":7,"Grade 8":8,"Grade 9":9,"Grade 10":10,"Grade 11":11,"Grade 12":12,
            "NTC l/N1/NIC/(v) Level":13,"NTC II/N2/NIC/(v) Level3":14,"NTC III/N3/NIC/(v) Level14":15,
            "N4/NTC 4":16,"N5/NTC":17,"N6/NTC":18,"Certificate with less than Grade 12/Std 10":19,
            "Diploma with less than Grade 12/Std 10":20,"Certificate with Grade 12/Std 10":21,
            "Diploma with Grade 12/Std 10":22,"HigherDiploma":23,"PostHigherDiploma":24,"BachelorsDegree":25,
            "BachelorsDegree and PostGraduateDiploma":26,"Honours Degree":27,"Higher Degree (Masters/Phd)":28,
            "Other":29,"I don't know":30,"No Schooling":98}


        study_field = {"Other":0,"Agriculture or RenewableNaturalResources":1,"Architecture or EnvironmentalDesign":2,
                "ArtsVisual or Performing":3,"Business,Commerce and ManagementStudies":4,"Communication":5,
                "ComputerScience":6,"Education,Training,Development":7,"Engineering or EngineeringTechnology":8,
                "Health Care":9,"Home Economics":10,"IndustrialArts,Traders or Technology":11,"Literature":12,
                "Law":13,"Libraries":14,"LifeSciences or PhysicalSciences":15,"MathematicalSciences":16,
                "MilitarySciences":17,"Philosophy":18,"PhysicalEducation":19,"Psychology":20,"SocialServices":21,
                "SocialSciences":22,"Other":23,"Management":24,"Marketing":25,"IT/CS":26,"Finance,Economics and Accounting":27,
                "OfficeAdministration":28,"Electrical Infrastructure Construction":29,"CivilEngineering":30,"Engineering":31,
                "PrimaryAgriculture":32,"Hospitality":33,"Tourism":34,"SafetyInSociety":35,"Mechatronics":36,
                "Education And Development":37,"Other":38}
        lf_work = {"Yes":1,"No":2}

        neet = {"Not Applicable":0,"Yes":1,"No":2}

        sex_chosen = st.selectbox("Choose you sex",(sex))
        sex_num = sex[sex_chosen]
        prov_chosen = st.selectbox("What is your province",(prov))
        prov_num = prov[prov_chosen]
        geo_type_chosen = st.selectbox("What is your geographical area type",(geo_type))
        geo_type_num = geo_type[geo_type_chosen]
        edu_status_chosen = st.selectbox("Choose your education status",(edu_status))
        edu_status_num = edu_status[edu_status_chosen]
        edu_lvl_chosen = st.selectbox("What is your education level",(edu_lvl))
        edu_lvl_chosen_num = edu_lvl[edu_lvl_chosen]
        study_field_chosen = st.selectbox("Choose your study field",(study_field))
        study_field_chosen_num = study_field[study_field_chosen]
        lf_work_chosen = st.selectbox("Are you looking for work?",(lf_work))
        lf_work_chosen_num = lf_work[lf_work_chosen]
        neet_chosen = st.selectbox("Not in employment, education and training?",(neet))
        neet_chosen_num = neet[neet_chosen]

        submit = st.button('Submit my answers')

        if submit:
            user_input = np.array([sex_num, prov_num, geo_type_num, edu_status_num, study_field_chosen_num,edu_lvl_chosen_num, lf_work_chosen_num, neet_chosen_num])
            pickle_in = open('./models/KNN-Emp.pickle' , 'rb')
            clf = pickle.load(pickle_in)
            prediction = clf.predict([user_input])
            Categories = ['Not Sure','Employed','Unemployed','Discouraged job seeker','not economically active']
            st.warning('You are likely to be ' + Categories[int(prediction)])

            if Categories[int(prediction)] == "Unemployed":
                st.success("""
                            Consider Your Career Path
                            NETWORK, NETWORK, NETWORK
                            Improve your skillset
                            Consider doing temporary or online jobs
                            Lastly, if you are into tech, consider taking part in hackathons :)""")

            elif Categories[int(prediction)] == 'Discouraged job seeker':
                st.success('''What to do if you are discouraged from looking for a job?
                            Send an email (or a LinkedIn message) to the individual you would potentially be reporting to. 
                            The email should be company specific, mention the issues they are facing and how your background can help them.''')
            elif Categories[int(prediction)] == "not economically active":
                st.success('''A person is considered to be economically inactive if they were able and available to work 
                        in the week prior to the survey but did not work, did not look for work and did not try to start their own business. 
                        This includes people such as university students and adults caring for children at home.
                    ''')


    elif type_ques == "Will my skills be important during the 4IR?":
        
        st.subheader("Please type in your skills and interest  below and hit submit😁")

        df = pd.read_csv('./datasets/JobSkills.csv')
        df['category_id'] = df['Category'].factorize()[0]
        category_id_df = df[['Category','category_id']].drop_duplicates().sort_values('category_id')
        category_to_id = dict(category_id_df.values)
        id_to_category = dict(category_id_df[['category_id','Category']].values)

        tfidf = TfidfVectorizer(sublinear_tf=True,min_df=5,encoding='utf-8',ngram_range=(1,2),stop_words='english')
        features = tfidf.fit_transform(df['cleaned_resume']).toarray()
        labels = df['category_id']

        X_train, X_test, y_train, y_test = train_test_split(df['cleaned_resume'],df['Category'], random_state=0)
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        clf = MultinomialNB().fit(X_train_tfidf, y_train)

        input_text = st.text_area("Please type in your skills and experiences here",height=250)
        
        submit_int = st.button("Submit")

        if submit_int:
            prediction = clf.predict(count_vect.transform([input_text]))
            st.success(prediction)

elif navigation == "Sentiment Analysis":
    st.header("Twitter Sentiment Analysis on the impact of the 4IR on employment")

    st.image('./datasets/twitter.jfif',width=1000)

    st.markdown('''
    The aim of this section is to analyze people's sentiments about the impact of technology on employment.
    People are getting nervous, depressed and even having suicidal thoughts due to unemployment. Many people
    rely on their jobs to make a living, but due to improvements in technology, seems like people will lose their jobs.

    If people have negative sentiments regarding the technology advancements, it is safe to assume that such people 
    are generally nervous, depressed, or concerned about losing their jobs to technology in the near future. 
    
    If people are nervous or depressed, that means their psychological health is upset or going to be affected 
    shortly once their employment status will be affected by technology
    ''')


    user_input = ["First, robots came for blue-collar jobs. (50 are already automatable). Now, AI is coming for white-collar jobs.",
            "Robots taking over jobs can be an opportunity. They could work for our Basic Income, and give us the cognitive space to think what we actually want to do.Instead of preparing us for jobs, education could prepare us on how to live a meaningful live.",
            "This is going to suck for Starbucks employees in about 5 to 10 years.  Robots are taking over now. So sad....",
            "my family is watching the new home alone and they're telling me the bandits are this financially struggling couple with kids and the guy can't get a job because of the automation of his career and that is genuinely so sad what the hell",
            "In the future  food service jobs could be carried out by robots, and this will put some people at risk of lossing their jobs which is very sad",
            "“Anyone that thinks of automation as a job killer is looking at this completely wrong.” –Donald Engineering’s Mark Gauthier on how #automation has presented new opportunities for his business & helped improve #workersafety at our hearing on the economic impact of tech innovation.",
            ]

    for t in user_input[:7]:
        tokens = tokenizer_sent.encode(t, return_tensors='pt')
        result = model_sent(tokens)
        sentiment_result = int(torch.argmax(result.logits))+1
        st.write("======================================================================================================================")
        st.markdown(t)
        if sentiment_result == 1:
            st.write(sentiment_result," : Extremely Negative")
        elif sentiment_result == 2:
            st.write(sentiment_result," : Negative")
        elif sentiment_result == 3:
            st.write(sentiment_result," : Neutral")
        elif sentiment_result == 4:
            st.write(sentiment_result," : Positive")
        elif sentiment_result == 5:
            st.write(sentiment_result," : Extreamely Positive")
