# PredictionIO electric load forecasting engine
## Overview
This is a PredictionIO engine for electric load forecasting.

## Usage
To use this engine you should first get yourself familiar with [PredicionIO](https://prediction.io/). You can do so easily by going through one of the PredictionIO official quick start guides, like this one: [Quick Start - Classification Engine Template](https://docs.prediction.io/templates/classification/quickstart/).

## Data format
The data that we are training our engine on contains values of hourly energy consumption for a group of circuits.

More formally, each of our events contains of 3 components: circuit id (represented by an integer number), date (represented by unix time stamp) and value of energy consumption (represented by floating point number).

You can see example data looking at [data.csv](data/data.csv). The file follows comma-separated values format. Rows in the first column contains timestamps. Every column, but first, begins with an integer value representing the circuit id. The [import_data.py](data/import_data.py) file is specifically created for importing data of such a format to our application. Example usage of the file:

```
python data/import_eventserver.py --access_key <APP_ACCESS_KEY> --file <DATA_FILE>
```

## Global Energy Forecasting Competition 2012
You can also easily import data of format as in [GEFComp2012](https://www.kaggle.com/c/global-energy-forecasting-competition-2012-load-forecasting/) Load_history.csv file using [import_data_gefcom2012.py](data/import_data_gefcom2012.py). The zone_id is imported to our application as circuit id.

## Query format
Query consists of circuit id (represented by an integer number) and time (represented by unix time stamp). Sending example query using PredictionIO Python SDK:

```
import predictionio
engine_client = predictionio.EngineClient(url="http://localhost:8000")
print engine_client.send_query({"circuit_id":1, "timestamp":1422298800})
```

## Algorithm
The algorithm used in this engine is linear regression with stochastic gradient descent from Spark MLlib.

## Evaluation
The engine uses root mean squared error as metric and k-fold splitting technique for evaluation. To evaluate the model on your data, after uploading it, all you need to do is type in: 

```
pio eval detrevid.predictionio.loadforecasting.RMSEEvaluation detrevid.predictionio.loadforecasting.EngineParamsList
``` 

### Note
For a better performance, you may want to expand JVM memory by adding "-- --driver-memory <MEMERY_AMOUNT>". For example:

```
pio eval detrevid.predictionio.loadforecasting.RMSEEvaluation detrevid.predictionio.loadforecasting.EngineParamsList -- --driver-memory 5g
```

You can find algorithms parameters used in evaluation in [Evaluation.scala](src/main/scala/Evaluation.scala).
You may want to learn more about evalutaion before you start using it, you can do so by reading [Tuning and Evaluation](https://docs.prediction.io/evaluation/).

