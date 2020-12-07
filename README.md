# Covid-19 Forecasting via Deep Learning and Topological Data Analysis

Understanding the impact of atmospheric and weather conditions on **SARS-CoV2** is critical to model **COVID-19** dynamics  and shed a light on future spread around the world.Furthermore, geographic distribution of expected clinical severity of **COVID-19** may be closely link to prior history of respiratory diseases and changes of humidity, temperature, and air quality. In this context, we postulate that by tracking topological features of weather conditions over time we can provide aquantifiable structural distribution of atmospheric changes that are likely to be related to **COVID-19** progression rates.  As such, we apply the machinery of persistence homology on time series of graphs to extract topological signatures and to follow geographical changes in humidity and temperature. We develop an integrative machine learning framework via **Geometric Deep Learning (GDL)** and test its predictive capabilities on forecasting the progression of **SARS-CoV2** cases. We validate our novel GDL framework in application to number of confirmed cases and hospitalization rates from Washington and California states in the United States. Our results demonstrate the predictive potential of GDL to forecasting the transmission of **COVID-19** and modeling its complex spatio-temporal spread dynamics.

Our proposed methodology has two main modules, see the graphical workflow in the below Figure:



<p float="left">
  <img src="Images/RNN_Architecture_page-0001.jpg" width="130" /> 
  <img  width="40" />
  <img src="Images/WorkflowMethod_page-0001.jpg" width="650" /> 
  <img  width="70" />
  <b>(a)
  <img  width="400" />
  <b>(b)
</p>

Topological LSTM (a) RNN architecture. (b) Our proposed methodology
