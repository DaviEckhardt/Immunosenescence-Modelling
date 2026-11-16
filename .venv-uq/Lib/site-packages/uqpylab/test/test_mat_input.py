import pytest
import uqpylab.sessions as uq_session
import numpy as np
import time

def test_mat_vs_json_inputs(request, helpers):
    mySession = helpers.init_session(request)
    
    uq = mySession.cli
    uq.rng(100, 'twister')
    ## 1) Test a simple input    
    InputOpts = {
            'Marginals': [
                {
                'Type':'Uniform',
                'Parameters' : [1, 5]
                }]
            }
    myInput_MAT = uq.createInput(InputOpts, response_format='MAT')
    myInput_JSON = uq.createInput(InputOpts, response_format='JSON')

    assert myInput_JSON["Marginals"] == myInput_MAT["Marginals"]
    assert myInput_JSON["Copula"] == myInput_MAT["Copula"]
  
    ## 2) Test a more elaborate input    
    InputOpts = {
            'Marginals': [
                {
            'Type': 'Gaussian',
            'Parameters': [-1,1]
            },
            {
                'Type': 'Exponential',
                'Parameters': [1]
            },
            {
                'Type': 'Uniform',
                'Parameters': [-1,3]
            }
            ],

            'Copula': {
                'Type': 'CVine',
                'Structure': [[3,1,2]],
                # fix the pair copula families and rotations
                'Families': ['Gaussian', 'Gumbel', 't'],
                'Rotations': [0,90,0],
                'Parameters': [.4, 2, [-.2, 2]]
            }

            }
    myInput_MAT = uq.createInput(InputOpts, response_format='MAT')
    myInput_JSON = uq.createInput(InputOpts, response_format='JSON')
    myInput_JSON["Copula"].pop('CondVars')
    myInput_MAT["Copula"].pop('CondVars')
    assert myInput_JSON["Marginals"] == myInput_MAT["Marginals"]
    assert myInput_JSON["Copula"] == myInput_MAT["Copula"]
  
    # Optional performance comparison
    start_time = time.time()
    samples_MAT = uq.getSample(myInput_JSON, N=1e5, response_format='MAT')
    t_MAT = time.time() - start_time

    start_time = time.time()
    samples_JSON = uq.getSample(myInput_JSON, N=1e5, response_format='JSON')
    t_JSON = time.time() - start_time

    print(f"GetSample timings | MAT: {t_MAT} s, JSON: {t_JSON} s")
    mySession.quit()
    time.sleep(1)    