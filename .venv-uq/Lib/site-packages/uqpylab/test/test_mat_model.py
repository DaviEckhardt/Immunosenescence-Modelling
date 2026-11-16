import pytest
import uqpylab.sessions as uq_session
import numpy as np
import time

def model(X):
    return X*np.sin(X)

def test_mat_vs_json_models(request, helpers):
    mySession = helpers.init_session(request)
    
    uq = mySession.cli
    uq.rng(100, 'twister')
    InputOpts = {
        'Marginals': [
            {
            'Type':'Uniform',
            'Parameters' : [1, 5]
            }]
        }
    print("Creating an input...")
    myInput = uq.createInput(InputOpts, response_format="MAT")
    print("Done.")
    print("Generating samples...")
    X_ED = uq.getSample(myInput, 100, response_format="MAT")
    print("Done.")

    print("Generating samples...")
    X_val = uq.getSample(N=1000, response_format="MAT")
    print("Done.")


    # create a default model
    print("Creating a true model object...")
    ModelOpts = {'Type' : 'Model',
        'ModelFun': 'uqpylab.test.true_models.XsinX'}
    myModel = uq.createModel(ModelOpts)
    print("Done.")
    print("Generating Y of training and validation set...")
    Y_val = model(X_val)
    Y_ED = model(X_ED)
    print("Done.")

    # do a PCE
    MetaOptsPCE = {
        'Type': 'Metamodel',
        'MetaType' : 'PCE',
        'ExpDesign': {
            'X': X_ED.tolist(),
            'Y': Y_ED.tolist()},
        'Input': myInput['Name'],
        'Method' : 'LARS',
        'Degree' : np.arange(1,15).tolist()
    }
    print("Computing a PCE...")
    myPCE = uq.createModel(MetaOptsPCE, response_format="MAT")
    print(myPCE)
    print("Done.")
    print("Performing PCE model evaluations...")
    print("Case 1: request_format='MAT', response_format='MAT'")
    Y_PCE_val = uq.evalModel(myPCE, X_val, request_format='MAT', response_format="MAT")
    print("Case 2: request_format='JSON', response_format='MAT'")
    Y_PCE_val_2 = uq.evalModel(myPCE, X_val, request_format='MAT', response_format="MAT")
    print("Done.")
    print("Validating results...")
    print(np.max(np.abs(Y_PCE_val -  Y_val)))
    assert np.allclose(Y_PCE_val, Y_PCE_val_2)
    assert np.allclose(Y_PCE_val, Y_val, atol=1e-7)
    print("All good.")
    mySession.quit()
    time.sleep(1)
