# Covid-19 Forecasting via Deep Learning and Topological Data Analysis


Our proposed methodology has two main modules, see the graphical workflow in the below Figure:



<p float="left">
  <img src="Images/RNN_Architecture_page-0001.jpg" width="130" /> 
  <img  width="40" />
  <img src="Images/WorkflowMethod_page-0001.jpg" width="650" /> 
  <img  width="90" />
  <b>(a)</b>
  <img  width="400" />
  <b>(b)</b>
</p>

Topological LSTM (a) RNN architecture. (b) Our proposed methodology


This package includes the source codes and datasets used in this research project. We encourage the reader to review the submitted paper: Covid-19 Forecasting via Deep Learning and Topological Data Analysis, and its supplementary material.

*Our experiments have been carried out using collected data of California and Washington  states.  Particularly,  our  methodology  produces  daily  COVID-19 progression and hospitalization forecasts at county-level resolution.*

The datasets for this research project are obtained from the below websites and repository:
* https://www.ncei.noaa.gov/
* https://github.com/CSSEGISandData/COVID-19
* https://midasnetwork.us/covid-19/

Please find the significance of each file and directory below:

* **Datasets Directory**: it contains all the publicly available datasets required for this research project. 
* **ChangeFormat_Hospitalizations_XXXX.R**: These Scripts change the original format of county for each state data to our format for the deep learning model. The extracted datasets are stored in the **CSV** directory.
* **LSTM_TDA_California_XXXX.py**: This script fits a LSTM model on Covid data using files available in the **InputLSTM** directory and saves the forecastings into the **Saved** directory
* **ExtractWeekly_Features_Washington**:  This Script changes the original format of county for each state data to our own format for the deep learning model.
* **Dynamic Network.py**: This script builds the Dynamic Network and extract topological summaries, and saves these results in a CSV file.




