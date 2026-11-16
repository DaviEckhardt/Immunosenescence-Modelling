import pytest
import uqpylab.sessions as uq_session
import numpy as np
#from uqlab_standalone import sessions as uq_session
import time

def test_inputs(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')
    # Construct an input object with various input distributions
    InputOpts = {
        'Marginals': [{
            'Type':'Gaussian',
            'Parameters': [0.10, 0.0161812]
            } ,
            {
            'Type':'Lognormal',
            'Parameters' : [7.71, 1.0056]
            },
            {
            'Type':'Uniform',
            'Parameters' : [63070, 115600]
            }]
        }
    print("Creating an input...")
    myInput = uq.createInput(InputOpts)
    print("Done.")
    print("Generating samples...")
    X_ED = uq.getSample(myInput, 100)
    print("Done.")
    # make sure that the session is properly terminated (no dangling background stuff)
    assert X_ED.shape == (100,3)

def test_copula_summary(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')
    X = np.random.rand(5,3)

    iOpts = {
        'Inference': {
                'Data': X.tolist()
            }
        
    }

    iOpts['Copula'] = {
        'Type': 'CVine',
        'Inference': {
            'CVineStructure': [[3,1,2]],
            'PCFamilies': ['Gaussian', 'Gumbel', 't']
        }
    }
    print("Creating an Input...")
    myInput = uq.createInput(iOpts)
    print("Done.")
    print("Generating some copula summaries...")
    uq.CopulaSummary(myInput['Copula'])
    uq.CopulaSummary('CVine', [3,1,2])
    print("No errors. Assuming that they look ok.")
    mySession.quit()
    time.sleep(1)    
