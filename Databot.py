import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import urllib.parse
import plotly.express as px
import pandas as pd
from streamlit_lottie import st_lottie
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
# to load the environment files
load_dotenv()

# Hello WoodPecker Team We are ,Team HackMate . 
# To run and test our project read the detailed description in readme file We have give all of your instructions.

# code written to establish the database connection 
def get_table_info(db_connection):
    query = "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE()"
    cursor = db_connection.cursor(dictionary=True)
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return [row['table_name'] for row in result]

def init_database(user: str, password: str, host: str, port: str, database: str):
    password_encoded = urllib.parse.quote(password)
    db_uri = f"mysql+mysqlconnector://{user}:{password_encoded}@{host}:{port}/{database}"
    try:
        engine = create_engine(db_uri)
        Session = sessionmaker(bind=engine)
        return Session()
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        return None

def get_table_names(session, database_name):
    query = text("""
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_SCHEMA = :database_name;
    """)
    try:
        result = session.execute(query, {"database_name": database_name})
        table_names = [row[0] for row in result.fetchall()]
        return table_names
    except Exception as e:
        st.error(f"Error fetching table names: {e}")
        return []

def fetch_table_data(session, table_name):
    query = text(f"SELECT * FROM {table_name}")
    try:
        result = session.execute(query)
        rows = result.fetchall()
        columns = result.keys()
        df = pd.DataFrame(rows, columns=columns)
        return df
    except Exception as e:
        st.error(f"Error fetching table data: {e}")
        return pd.DataFrame()

def load_lottie_url(url: str):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def init_database_for_chat(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    password_encoded = urllib.parse.quote(password)
    db_uri = f"mysql+mysqlconnector://{user}:{password_encoded}@{host}:{port}/{database}"
    try:
        return SQLDatabase.from_uri(db_uri)
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        return None
    
# code written to convert the user query into sql query by using langchain
def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    #using grop api to convert the query
    llm = ChatGroq(model="Llama3-8b-8192", temperature=0)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

# code to written the response to user after exuceuting in the sql query in database . Response was in natural language to user.

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response Don't return the sql query.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })







# Streamlit app setup
st.set_page_config(page_title="DATABOT", page_icon=":speech_balloon:")

# Sidebar navigation
st.sidebar.title("DATABOT")
page = st.sidebar.radio("",["ChatBase", "Analytics Tool","BI Tool"])

# code for navigation bar
if page == "ChatBase":
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm your SQL assistant. Ask me anything about your database."),
        ]

    st.title("Query your Database")

    with st.sidebar:
        st.subheader("INFORMATION")
        st.write("This is a Chat application to query MySQL Database  in Natural Language. Connect to the database and start chatting.")
        
        st.text_input("Host", value="localhost", key="Host")
        st.text_input("Port", value="3306", key="Port")
        st.text_input("User", value="root", key="User")
        st.text_input("Password", type="password", value="Pondy@123", key="Password")
        st.text_input("Database", value="SchoolDb", key="Database")
        
        if st.button("Connect"):
            with st.spinner("Connecting to database..."):
                db = init_database_for_chat(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Database"]
                )
                if db:
                    st.session_state.db = db
                    st.success("Connected to database!")
                else:
                    st.error("Failed to connect to database.")

    if st.button("Clear Chat History"):
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
        ]
        st.success("Chat history cleared!")

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            st.markdown(f"""
            <div style="color:black; font-size: 16px; background-color:white; padding: 10px; border-radius: 5px; margin-top:5px;">
                <strong>AI:</strong> {message.content} <span role="img" aria-label="robot">🤖</span>
            </div>
            """, unsafe_allow_html=True)
        elif isinstance(message, HumanMessage):
            st.markdown(f"""
            <div style="color: #1f77b4; font-size: 16px; background-color: #e6f7ff; padding: 10px; border-radius: 5px; margin-top:5px;">
                <strong>You:</strong> {message.content} <span role="img" aria-label="user">👤</span>
            </div>
            """, unsafe_allow_html=True)
    # input for the user query
    user_query = st.chat_input("Type a message...")
    if user_query and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        st.markdown(f"""
        <div style="color: #1f77b4; font-size: 16px; background-color: #e6f7ff; padding: 10px; border-radius: 5px; margin-top:5px;">
            <strong>You:</strong> {user_query} <span role="img" aria-label="user">👤</span>
        </div>
        """, unsafe_allow_html=True)
        # Response for the user
        if "db" in st.session_state:
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
            
            st.markdown(f"""
            <div style="color:black; font-size: 16px; background-color:white; padding: 10px; border-radius: 5px; margin-top:5px;">
                <strong>AI:</strong> {response} <span role="img" aria-label="robot">🤖</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.chat_history.append(AIMessage(content=response))
        else:
            st.error("Database not connected. Please connect to the database.")


# code for the Analysis of Datababse
elif page == "Analytics Tool":
    import pandas as pd
    import plotly.express as px
    import streamlit as st
    from streamlit_lottie import st_lottie

    st.markdown("""
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #f5f5f5;
                color: #333;
            }
            .header {
                text-align: center;
                font-size: 36px;
                font-weight: bold;
                color: #4CAF50;
                margin: 20px 0;
            }
            .section-header {
                text-align: center;
                font-size: 24px;
                font-weight: bold;
                color: #007BFF;
                margin: 30px 0 10px;
            }
            .container {
                margin-top: 30px;
                padding: 20px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .button {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 18px;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }
            .button:hover {
                background-color: #45a049;
            }
            .sidebar .element-container {
                margin-bottom: 20px;
            }
            .input-label {
                font-size: 16px;
                margin-bottom: 5px;
            }
            .input-field {
                font-size: 16px;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ddd;
                width: 100%;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header">Database Analysis</div>', unsafe_allow_html=True)
    st_lottie(load_lottie_url("https://lottie.host/b508948f-c6d5-4ec1-98be-307b446c0b59/llzwGAIgpm.json"), speed=1, width=600, height=400, key="analytics")

    with st.sidebar:
        st.subheader("Upload Data")
        data_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

        st.subheader("Database Connection")
        st.text_input("Host", value="localhost", key="Analytics_Host")
        st.text_input("Port", value="3306", key="Analytics_Port")
        st.text_input("User", value="root", key="Analytics_User")
        st.text_input("Password", type="password", value="Pondy@123", key="Analytics_Password")
        st.text_input("Database", value="SchoolDb", key="Analytics_Database")

        chart_type = st.sidebar.selectbox(
            "Select Chart Type", 
            ["Scatter Plot", "Bar Chart", "Pie Chart", "Area Chart", "Line Chart", "Histogram", "Box Plot"]
        )
        
        if st.button("Connect to Database", key="connect_db"):
            with st.spinner("Connecting to database..."):
                session = init_database(
                    st.session_state["Analytics_User"],
                    st.session_state["Analytics_Password"],
                    st.session_state["Analytics_Host"],
                    st.session_state["Analytics_Port"],
                    st.session_state["Analytics_Database"]
                )
                if session:
                    st.session_state.analytics_db_session = session
                    st.session_state.analytics_database_name = st.session_state["Analytics_Database"]
                    st.success("Connected to database")

    df = pd.DataFrame()
    # code for the Analysis of the Database
    if data_file is not None:
        try:
            if data_file.name.endswith(".csv"):
                df = pd.read_csv(data_file)
            elif data_file.name.endswith(".xlsx"):
                df = pd.read_excel(data_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")

    if "analytics_db_session" in st.session_state:
        session = st.session_state.analytics_db_session
        database_name = st.session_state.analytics_database_name
        table_names = get_table_names(session, database_name)
        selected_table = st.selectbox("Select a table from the database", table_names)

        if selected_table:
            df = fetch_table_data(session, selected_table)

    if not df.empty:
        st.markdown('<div class="section-header">Data Overview</div>', unsafe_allow_html=True)
        st.dataframe(df)

        st.markdown('<div class="section-header">Data Analysis</div>', unsafe_allow_html=True)

        # code for selecting the visualization type of analysis
        if chart_type == "Scatter Plot":
            st.subheader("Scatter Plot")
            scatter_x = st.selectbox("X-axis", df.columns, key="scatter_x")
            scatter_y = st.selectbox("Y-axis", df.columns, key="scatter_y")
            scatter_color = st.selectbox("Color", [None] + list(df.columns), key="scatter_color")

            fig = px.scatter(df, x=scatter_x, y=scatter_y, color=scatter_color, title="Scatter Plot")
            st.plotly_chart(fig)

        elif chart_type == "Bar Chart":
            st.subheader("Bar Chart")
            bar_x = st.selectbox("X-axis", df.columns, key="bar_x")
            bar_y = st.selectbox("Y-axis", df.columns, key="bar_y")
            bar_color = st.selectbox("Color", [None] + list(df.columns), key="bar_color")

            fig = px.bar(df, x=bar_x, y=bar_y, color=bar_color, title="Bar Chart")
            st.plotly_chart(fig)

        elif chart_type == "Pie Chart":
            st.subheader("Pie Chart")
            pie_values = st.selectbox("Values", df.columns, key="pie_values")
            pie_names = st.selectbox("Names", df.columns, key="pie_names")

            fig = px.pie(df, values=pie_values, names=pie_names, title="Pie Chart")
            st.plotly_chart(fig)

        elif chart_type == "Area Chart":
            st.subheader("Area Chart")
            area_x = st.selectbox("X-axis", df.columns, key="area_x")
            area_y = st.selectbox("Y-axis", df.columns, key="area_y")

            fig = px.area(df, x=area_x, y=area_y, title="Area Chart")
            st.plotly_chart(fig)

        elif chart_type == "Line Chart":
            st.subheader("Line Chart")
            line_x = st.selectbox("X-axis", df.columns, key="line_x")
            line_y = st.selectbox("Y-axis", df.columns, key="line_y")

            fig = px.line(df, x=line_x, y=line_y, title="Line Chart")
            st.plotly_chart(fig)

        elif chart_type == "Histogram":
            st.subheader("Histogram")
            hist_x = st.selectbox("X-axis", df.columns, key="hist_x")

            fig = px.histogram(df, x=hist_x, title="Histogram")
            st.plotly_chart(fig)

        elif chart_type == "Box Plot":
            st.subheader("Box Plot")
            box_x = st.selectbox("X-axis", df.columns, key="box_x")
            box_y = st.selectbox("Y-axis", df.columns, key="box_y")

            fig = px.box(df, x=box_x, y=box_y, title="Box Plot")
            st.plotly_chart(fig)

        # code for performing Statistical Analysis
        st.markdown('<div class="section-header">Statistical Analysis</div>', unsafe_allow_html=True)
        
        st.subheader("Statistical Summary")
        stats_df = df.describe(include='all')
        st.dataframe(stats_df)

        st.subheader("Advanced Statistics")
        st.write("Select columns for advanced statistics analysis:")

        col_options = df.columns.tolist()
        stat_col = st.selectbox("Select Column", col_options)

        if stat_col:
            st.write(f"Statistics for {stat_col}:")
            col_data = df[stat_col]
            st.write(f"Mean: {col_data.mean()}")
            st.write(f"Median: {col_data.median()}")
            st.write(f"Mode: {col_data.mode().values}")
            st.write(f"Standard Deviation: {col_data.std()}")
            st.write(f"Variance: {col_data.var()}")
            st.write(f"Minimum: {col_data.min()}")
            st.write(f"Maximum: {col_data.max()}")

    else:
        st.info("Upload a dataset or connect to a database to get started.")

# code generation BI dashboard form the Database 
# The generation of BI dashboard is under development as we are tesing with xl files
elif page == "BI Tool":
    import streamlit as st
    import pandas as pd
    import pygwalker as pyg
    from streamlit_lottie import st_lottie
    import streamlit.components.v1 as components

    st_lottie(load_lottie_url("https://lottie.host/3591b49e-3ae2-4137-945a-d741f9064a49/c8HNZeYMZB.json"), speed=1, width=600, height=400, key="BI Tool")

    

    st.title('Tableau')
    choice = st.selectbox("Select chart type:", ["XLSX", "CSV"])

    if choice == "XLSX":
        uploaded_file = st.file_uploader('Choose an XLSX file', type='xlsx')
        if uploaded_file is None:
            st.error("Please select a dataset.")
        else:
            df = pd.read_excel(uploaded_file)
            pyg_output = pyg.walk(df, env="streamlit", dark='dark')
            pyg_html = pyg_output.to_html()  
            components.html(pyg_html, height=1000, scrolling=True)

    if choice == "CSV":
        uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
        if uploaded_file is None:
            st.error("Please select a dataset.")
        else:
            df = pd.read_csv(uploaded_file)
            pyg_output = pyg.walk(df, env="streamlit", dark='dark')
            pyg_html = pyg_output.to_html()  
            components.html(pyg_html, height=1000, scrolling=True)

