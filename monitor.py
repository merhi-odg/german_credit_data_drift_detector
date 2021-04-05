# modelop.slot.0:in-use
# modelop.slot.1:in-use
# modelop.slot.2:in-use


import pandas as pd
import numpy as np
import json
import pickle
from moc_monitors import DriftDetector
from moc_schema_infer import set_detector_parameters


# modelop.init
def begin():

    global schema
    
    # Read schema
    schema = pd.read_json(
        'df_sample_scored_input_schema.avsc',
        orient='records'
    )
    # set schema index to be "name"
    schema.set_index('name', inplace=True)

    
# modelop.metrics
def metrics(data, slot_no):
    
    global sample_df, baseline_df
    
    # print("\ndata: ", data, flush=True)
    print("\nSlot number: ", slot_no, flush=True)
    
    if slot_no==0:
        
        sample_df = data.copy()
        sample_df = pd.DataFrame(sample_df)
        
        print("\nbaseline_df: ", baseline_df, flush=True)
        print("\nsample_df: ", sample_df, flush=True)
        print("\ntype(sample_df): ", type(sample_df), flush=True)

    if slot_no==2:
        
        baseline_df = data.copy()
        baseline_df = pd.DataFrame(baseline_df)
        
        print("\nbaseline_df: ", baseline_df, flush=True)
        print("\ntype(baseline_df): ", type(baseline_df), flush=True)
        print("\nsample_df: ", sample_df, flush=True)
        
        
    if baseline_df is not None and sample_df is not None:
        
        print("\nYielding", flush=True)
        
        detector_parameters = set_detector_parameters(schema)
    
        drift_detector=DriftDetector(
            df_baseline=baseline_df, 
            df_sample=sample_df, 
            categorical_columns=detector_parameters["categorical_columns"], 
            numerical_columns=detector_parameters["numerical_columns"], 
            score_column=detector_parameters["score_column"][0], 
            label_column=detector_parameters["label_column"][0]
        )

        output = drift_detector.calculate_drift(
            pre_defined_metric='jensen-shannon',
            user_defined_metric=None
        )

        yield output

    else:
        print("\nReturning\n", flush=True)
        return
    
    
    
