import streamlit as st
import requests
import pandas as pd
import datarobot as dr
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

#Configure the page title, favicon, layout, etc
st.set_page_config(page_title="Time Series Lift Chart",
                   page_icon=":chart:",
                   layout="wide")

# Define some functions
@st.cache(show_spinner=False,allow_output_mutation=True, suppress_st_warning=True)
def getStackedPredictions(project, model, datasetid):
    try:
        holdout_predict_job = model.request_training_predictions(dr.enums.DATA_SUBSET.HOLDOUT)
        holdout_predictions = holdout_predict_job.get_result_when_complete()
    except:
        holdout_predictions = [tp for tp in dr.TrainingPredictions.list(project.id)
                               if tp.model_id == model.id
                               and tp.data_subset == 'holdout'][0]

    try:
        backtest_predict_job = model.request_training_predictions(dr.enums.DATA_SUBSET.ALL_BACKTESTS)
        backtest_predictions = backtest_predict_job.get_result_when_complete()
    except:
        backtest_predictions = [tp for tp in dr.TrainingPredictions.list(project.id)
                                if tp.model_id == model.id
                                and tp.data_subset == 'allBacktests'][0]

    df_holdout_preds = holdout_predictions.get_all_as_dataframe()
    df_backtest_preds = backtest_predictions.get_all_as_dataframe()

    df = pd.concat([df_backtest_preds, df_holdout_preds], axis=0)

    df["timestamp"] = pd.to_datetime(df["timestamp"].str[:10])
    df["forecast_point"] = pd.to_datetime(df["forecast_point"].str[:10])
    df = df.sort_values(["series_id", "timestamp", "forecast_distance"])
    #df = df.loc[df["forecast_distance"] == 1]

    train_df = dr.Dataset.get(dataset_id=datasetid).get_as_dataframe()

    train_df[dr.DatetimePartitioning.get(project.id).datetime_partition_column.replace(" (actual)","")] = pd.to_datetime(train_df[dr.DatetimePartitioning.get(project.id).datetime_partition_column.replace(" (actual)","")])
    train_df["series_id"] = train_df[dr.DatetimePartitioning.get(project.id).multiseries_id_columns[0].replace(" (actual)","")]
    train_df["timestamp"] = train_df[dr.DatetimePartitioning.get(project.id).datetime_partition_column.replace(" (actual)","")]

    result = df.merge(train_df, how="left", on=["timestamp", "series_id"])
    result.reset_index(drop=True, inplace=True)
    return result

#This is a multipage app. Pages are defined as functions, but you can also setup pages as separate .py files.
def introPage():
    st.title("Time Series Charting App")
    st.subheader("Choose a page from the sidebar.")
    logo = Image.open(requests.get('https://datatechvibe.com/wp-content/uploads/2021/11/Company-Closeup-DataRobot-Cover-696x392.jpg', stream=True).raw)
    st.image(logo, width=600, caption="https://datatechvibe.com/wp-content/uploads/2021/11/Company-Closeup-DataRobot-Cover-696x392.jpg")

#First page
def liftChart():
    with st.sidebar.form(key="form1"):
        API_KEY = st.text_input(label="DataRobot API Key")
        URL = st.text_area(label="Paste a Time Series Model URL from the DataRobot leaderboard.")
        submit_button = st.form_submit_button()

    if submit_button:
        try:
            # Title
            st.header("Accuracy Over Time and Lift Charts")
            st.write("Quickly get to the details behind DataRobot lift charts.")

            #Connect to DataRobot, get the model
            dr.Client(token=API_KEY, endpoint='https://app.datarobot.com/api/v2')
            projectid = URL.split("projects/")
            projectid = projectid[1][:24]
            modelid = URL.split("models/")
            modelid = modelid[1][:24]
            project = dr.Project.get(project_id=projectid)
            model = dr.Model.get(project=project, model_id=modelid)
            datasetid = project.get_dataset().id
        except Exception as e:
            st.write(e)

    try:
        # Get the predictions
        with st.spinner("Processing..."):
            data = getStackedPredictions(project,model,datasetid)
    except Exception as e:
        st.write(e)

    try:
        data["partition_id"] = data["partition_id"].str.replace(".0","")
        selected_series = st.selectbox(label="Choose a series", options=project.get_multiseries_names())

        #Accuracy Over Time Plot for the selected series
        data1 = data.loc[(data["series_id"]==selected_series) & (data["forecast_distance"] == 1)].copy()
        fig1 = px.line(title="Accuracy over Time for " + selected_series + " All Backtests, Forecast Distance +1")
        fig1.add_trace(go.Scatter(
            x=data1["timestamp"],
            y=data1[project.target.replace(" (actual)","")],
            mode="lines",
            yhoverformat=",.2f",
            name=project.target,
            line_shape="spline",
            line=dict(color="#ffbd00", width=2)
        ))
        fig1.add_trace(go.Scatter(
            x=data1[dr.DatetimePartitioning.get(project.id).datetime_partition_column.replace(" (actual)","")],
            y=data1["prediction"],
            mode="lines",
            yhoverformat=",.2f",
            name="Prediction",
            line_shape="spline",
            line=dict(color="#1d3557", width=2)
        ))
        fig1.update_layout(hovermode="x unified")
        st.plotly_chart(fig1, theme="streamlit", use_container_width=True)


        #Lift Chart for all points, selected series
        data2 = data1.copy()
        data2.sort_values("prediction", inplace=True)
        data2.reset_index(drop=True, inplace=True)
        fig2 = px.line(title="Lift Chart (Not Binned) for " + selected_series + " All Backtests, Forecast Distance +1")
        fig2.add_trace(go.Scatter(
            x=data2.index,
            y=data2[project.target.replace(" (actual)", "")],
            mode="lines",
            yhoverformat=",.2f",
            name=project.target,
            line_shape="spline",
            line=dict(color="#ffbd00", width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=data2.index,
            y=data2["prediction"],
            mode="lines",
            yhoverformat=",.2f",
            name="Prediction",
            line_shape="spline",
            line=dict(color="#1d3557", width=2)
        ))
        fig2.update_layout(hovermode="x unified")
        st.plotly_chart(fig2, theme="streamlit", use_container_width=True)

        #Binned Lift chart settings
        st.subheader("Binned Lift Charts All Forecast Distances")
        st.write("These charts are just like the lift charts in DataRobot. Click the expander to see the full table of values below. ")
        bins = st.number_input(label="Number of Bins", value=10, min_value=5, max_value=100)
        backtest_options = list(range(0,dr.DatetimePartitioning.get(project.id).number_of_backtests))
        if dr.DatetimePartitioning.get(project.id).disable_holdout == False:
            backtest_options.append("Holdout")
        backtest = st.selectbox(label="Choose a Backtest", options=backtest_options)

        #Lift Chart Binned Selected Series and Backtest
        data3 = data.loc[(data["series_id"]==selected_series)].copy()
        data3 = data3.loc[data3["partition_id"] == str(backtest)]
        data3["rank"] = data3["prediction"].rank(method="first")
        data3["Bin"] = pd.qcut(data3["rank"], q=bins, labels=False)
        data3r = data3.copy()
        data3 = data3[["partition_id","Bin", "prediction",project.target.replace(" (actual)", "")]].groupby(["partition_id","Bin"]).agg(['mean','count']).reset_index(drop=False)
        data3.columns = data3.columns.to_flat_index()
        data3.columns = ["Backtest","Bin","Prediction Mean","Prediction Count",project.target.replace(" (actual)","")+" Mean","Rows in Bin"]

        fig3 = px.line(title="Lift Chart Binned for " + selected_series, hover_data=[data3["Rows in Bin"]])
        fig3.add_trace(go.Scatter(
            x=data3["Bin"],
            y=data3[project.target.replace(" (actual)","")+" Mean"],
            mode="lines+markers",
            yhoverformat=",.2f",
            name=project.target,
            line_shape="spline",
            line=dict(color="#ffbd00", width=2)
        ))
        fig3.add_trace(go.Scatter(
            x=data3["Bin"],
            y=data3["Prediction Mean"],
            mode="lines+markers",
            yhoverformat=",.2f",
            name="Prediction",
            line_shape="spline",
            line=dict(color="#1d3557", width=2)
        ))
        fig3.update_layout(hovermode="x unified")
        st.plotly_chart(fig3, theme="streamlit", use_container_width=True)
        with st.expander("See data"):
            st.subheader("Predictions are sorted from lowest to highest.")
            st.dataframe(data3r.sort_values("prediction"), use_container_width=True)
            st.subheader("Then grouped into equally sized bins. The lift chart plots the mean values of predictions and actuals for each bin.")
            st.dataframe(data3, use_container_width=True)

        # Lift Chart Binned All Series and Selected Backtest
        data4 = data.copy()
        data4 = data4.loc[data4["partition_id"] == str(backtest)]
        data4["rank"] = data["prediction"].rank(method="first")
        data4["Bin"] = pd.qcut(data4["rank"], q=bins, labels=False)
        data4r = data4.copy()
        data4 = data4[["partition_id","Bin", "prediction", project.target.replace(" (actual)", "")]].groupby(["partition_id","Bin"]).agg(['mean', 'count']).reset_index(drop=False)
        data4.columns = data4.columns.to_flat_index()
        data4.columns = ["Backtest","Bin", "Prediction Mean", "Prediction Count", project.target.replace(" (actual)", "") + " Mean", "Rows in Bin"]

        fig4 = px.line(title="Lift Chart Binned for all Series")
        fig4.add_trace(go.Scatter(
            x=data4["Bin"],
            y=data4[project.target.replace(" (actual)", "") + " Mean"],
            mode="lines+markers",
            yhoverformat=",.2f",
            name=project.target,
            line_shape="spline",
            line=dict(color="#ffbd00", width=2)
        ))
        fig4.add_trace(go.Scatter(
            x=data4["Bin"],
            y=data4["Prediction Mean"],
            mode="lines+markers",
            yhoverformat=",.2f",
            name="Prediction",
            line_shape="spline",
            line=dict(color="#1d3557", width=2)
        ))
        fig4.update_layout(hovermode="x unified")
        st.plotly_chart(fig4, theme="streamlit", use_container_width=True)
        with st.expander("See data"):
            st.subheader("Predictions are sorted from lowest to highest.")
            st.dataframe(data4r.sort_values("prediction"), use_container_width=True)
            st.subheader("Then grouped into equally sized bins. The lift chart plots the mean values of predictions and actuals for each bin.")
            st.dataframe(data4, use_container_width=True)
    except:
        pass

#Second page
def page2():
    st.header("Blank page 2")

#Main app
def _main():
    hide_streamlit_style = """
    <style>
    # MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) # This let's you hide the Streamlit branding

    # Navigation to the different pages in the app
    page_names_to_funcs = {
        "Welcome": introPage,
        "Lift Chart": liftChart
    }

    page_name = st.sidebar.selectbox("Choose a Page", page_names_to_funcs.keys())
    page_names_to_funcs[page_name]()

if __name__ == "__main__":
    _main()


