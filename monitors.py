import pandas as pd
import numpy as np
import json
import pickle
from moc_monitors import BiasMonitor, DriftDetector
from moc_schema_infer import set_detector_parameters


# modelop.init
def begin():

    global schema
    
    #Read schema
    schema = pd.read_json(
        'df_sample_scored_input_schema.avsc',
        orient='records'
    )
    # set schema index to be "name"
    schema.set_index('name', inplace=True)

    
# modelop.metrics
def metrics(data):
    
    df_baseline = pd.read_json('df_baseline_scored.json', orient='records', lines=True)
    
    #if 'label_value' in df_baseline.columns and 'label' not in df_baseline.columns:
    #    df_baseline.rename(
    #        columns={'label_value': 'label'},
    #        inplace=True
    #    )
    
    print(df_baseline.columns, flush=True)
    
    print('',flush=True)
    
    print(data.columns, flush=True)
    
    detector_parameters = set_detector_parameters(schema)
    
    drift_detector=DriftDetector(
        df_baseline=df_baseline, 
        df_sample=data, 
        categorical_columns=detector_parameters["categorical_columns"], 
        numerical_columns=detector_parameters["numerical_columns"], 
        score_column=detector_parameters["score_column"][0], 
        label_column=detector_parameters["label_column"][0]
    )
    
    output = drift_detector.calculate_drift(
        pre_defined_metric='jensen-shannon',
        user_defined_metric=None
    )
    
    
    bias_montior = BiasMonitor(
        df=data,
        score_column=detector_parameters["score_column"][0],
        label_column=detector_parameters["label_column"][0],
        protected_class='gender',
        reference_group='male'
    )
    
    output = bias_montior.compute_bias_metrics(
        pre_defined_metric='aequitas_bias',
        user_defined_metric=None,
    ).to_dict(orient='records')
    
    yield output
