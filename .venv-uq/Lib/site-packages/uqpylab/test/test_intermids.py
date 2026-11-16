import pytest
import uqpylab.sessions as uq_session
import numpy as np
import time

def test_intermid_computations(request, helpers):
    mySession = helpers.init_session(request)
    
    uq = mySession.cli
    uq.rng(100, 'twister')
    ## 1) Create an input    
    InputOpts = {
            'Marginals': [
                {
                'Type':'Uniform',
                'Parameters' : [1, 5]
                }]
            }
    myInput = uq.createInput(InputOpts)
    ## 2) Specify a computational model locally
    ##  - JSON mode
    ModelOpts_JSON = {
        'Type' : 'Model',
        'ModelFun': 'uqpylab.test.true_models.XsinX',
        'CommFormat': 'JSON'
        }
    myModel_JSON = uq.createModel(ModelOpts_JSON)
    ##  - MAT mode
    ModelOpts_MAT = ModelOpts_JSON
    ModelOpts_MAT['CommFormat'] = 'MAT'
    myModel_MAT = uq.createModel(ModelOpts_MAT)
    ## 3) Use each model to generate an ED during a metamodel creation
    PCEOpts_JSON = {
        'Type' : 'Metamodel',
        'MetaType': 'PCE',
        'FullModel': myModel_JSON['Name'],
        'Input': myInput['Name'],
        'ExpDesign': {
            'NSamples': 100,
            'Sampling': 'LHS'
            },
        'Method': 'OLS',
        'Degree': np.arange(3,16).tolist()
        }
    myPCE_JSON_Model = uq.createModel(PCEOpts_JSON)

    PCEOpts_MAT = PCEOpts_JSON
    PCEOpts_MAT['FullModel'] = myModel_MAT['Name']

    myPCE_MAT_Model = uq.createModel(PCEOpts_MAT)
    ## 4) Perform checks
    coeffs_json = np.array(myPCE_JSON_Model['PCE']['Coefficients'])
    coeffs_mat = np.array(myPCE_MAT_Model['PCE']['Coefficients'])
    assert np.allclose(coeffs_json, coeffs_mat)

    ## 5) Let's repeat the process for a 2D function to make sure
    # that the dimensions are still consistent
    
    InputOpts = {
            'Marginals': [
                {
                'Type':'Gaussian',
                'Parameters' : [0.25, 1]
                },
                {
                'Type':'Gaussian',
                'Parameters' : [0.25, 1]
                }]
            }
    myInput = uq.createInput(InputOpts)
    ModelOpts_JSON = {
        'Type' : 'Model',
        'ModelFun': 'uqpylab.test.true_models.hat2d',
        'CommFormat': 'JSON'
        }
    myModel_JSON = uq.createModel(ModelOpts_JSON)
    ##  - MAT mode
    ModelOpts_MAT = ModelOpts_JSON
    ModelOpts_MAT['CommFormat'] = 'MAT'
    myModel_MAT = uq.createModel(ModelOpts_MAT)
    PCEOpts_JSON = {
        'Type' : 'Metamodel',
        'MetaType': 'PCE',
        'FullModel': myModel_JSON['Name'],
        'Input': myInput['Name'],
        'ExpDesign': {
            'NSamples': 100,
            'Sampling': 'LHS'
            },
        'Method': 'OLS',
        'Degree': np.arange(3,16).tolist()
        }
    myPCE_JSON_Model = uq.createModel(PCEOpts_JSON)

    PCEOpts_MAT = PCEOpts_JSON
    PCEOpts_MAT['FullModel'] = myModel_MAT['Name']

    myPCE_MAT_Model = uq.createModel(PCEOpts_MAT)
    ## 4) Perform checks
    coeffs_json = np.array(myPCE_JSON_Model['PCE']['Coefficients'])
    coeffs_mat = np.array(myPCE_MAT_Model['PCE']['Coefficients'])
    assert np.allclose(coeffs_json, coeffs_mat)
    mySession.quit()
    time.sleep(1)    