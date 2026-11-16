import pytest
from  uqpylab import sessions 
import numpy as np
#from uqlab_standalone import sessions as uq_session
import time

def function_caller(InputName):
    mySession = sessions.cloud()
    X = mySession.cli.getSample(InputName, 100)
    return X

def cache_checker(ExpectedValue):
    mySession = sessions.cloud()
    return mySession.cache == ExpectedValue

def test_singleton_main(request, helpers):
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

    X = function_caller(myInput['Name'])
    # make sure that the session is properly terminated (no dangling background stuff)
    assert X.shape == (100,3)
    print("Done.")

def test_cache(request, helpers):
    print("Testing cache functionality...")
    mySession = helpers.init_session(request)
    CacheValue = 'Any type of variable can be put here'
    mySession.cache = CacheValue
    assert cache_checker(CacheValue)
    print("Done.")
    mySession.quit()
    time.sleep(1)    

