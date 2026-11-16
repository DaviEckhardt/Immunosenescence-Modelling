import pytest
import uqpylab.sessions as sessions
import numpy as np


def model(X):
    return X*np.sin(X)


mySession = sessions.cloud(host="http://localhost:8000", token="636fd25c8c0aa05e344c0cc66abdcfb8c3af73d8")
# (Optional) Get a convenient handle to the command line interface
uq = mySession.cli
# Reset the session
mySession.reset()

uq.rng(100, 'twister')
print("Creating a true model object...")
ModelOpts = {
    'Type': 'Model',
    'ModelFun': 'uqpylab.test.true_models.simply_supported_beam_9points',
    'isVectorized': 'true'
}
myModel = uq.createModel(ModelOpts)
print("Done.")
print("Creating an input...")
InputOpts = {
    'Marginals': [
        {
            'Name': 'b', # beam width
            'Type': 'Lognormal',
            'Moments': [0.15, 0.0075] # (m)
        },
        {
            'Name': 'h', # beam height
            'Type': 'Lognormal',
            'Moments': [0.3, 0.015] # (m)
        },
        {
            'Name': 'L', # beam length
            'Type': 'Lognormal',
            'Moments': [5, 0.05] # (m)
        },
        {
            'Name': 'E', # Young's modulus
            'Type': 'Lognormal',
            'Moments': [3e10, 4.5e9] # (Pa)
        },
        {
            'Name': 'p', # uniform load
            'Type': 'Lognormal',
            'Moments': [1e4, 2e3] # (N/m) # in Kriging example, there is [1e4 1e3]
        }
    ]
}

myInput = uq.createInput(InputOpts)
print("Done.")
print("Generating samples...")
X_val = uq.getSample(N=10)
Y_val = uq.evalModel(myModel,X_val)
print("Done.")

# do KRG
print("Creating a KRG surrogate...")
MetaOptsKRG = {
    "Type": "Metamodel",
    "MetaType": "Kriging",
    "Input": myInput["Name"],
    "FullModel": myModel["Name"],
    "ExpDesign": {
        "Sampling": "LHS",
        "NSamples": 10
    },
    "Trend": {
        "Type": "linear"
    },
    "Corr": {
        "Family": "exponential"
    },
    "EstimMethod": "ML",
    "Optim": {
        "Method" : "HGA",
        "MaxIter" : 50,
        "HGA" : {
            "nPop" : 50
        }
    }
}
myKRG = uq.createModel(MetaOptsKRG)
print("Done.")
print("Evaluating KRG model.")
YmeanKRG = uq.evalModel(myKRG,X_val)
assert np.allclose(Y_val, YmeanKRG, atol=5e-2)
print("Done.")
print("All good.")

# do PCE
MetaOptsPCE = {
    'Type': 'Metamodel',
    'MetaType': 'PCE',
    'TruncOptions': {'qNorm': 0.75},
    'Degree': np.arange(2,11).tolist(),
    'ExpDesign': {
        "NSamples" : 120,
        "Sampling" : "LHS"
    },
    "Input": myInput["Name"]
}
myPCE = uq.createModel(MetaOptsPCE)
print("Done.")
print("Evaluating PCE model.")
YmeanPCE = uq.evalModel(myPCE,X_val)
assert np.allclose(Y_val, YmeanPCE, atol=5e-2)
print("Done.")
print("All good.")    


# do PCK
print("Creating a PCK surrogate...")
MetaOptsPCK = {
    "Type": "Metamodel",
    "MetaType": "PCK",
    "ExpDesign": {
        "Sampling": "LHS",
        "NSamples": 100
    },
    "Input": myInput["Name"],
    "PCE": {"Degree": 3}
}
myPCK = uq.createModel(MetaOptsPCK)
print("Done.")
print("Evaluating PCK model.")
YmeanPCK = uq.evalModel(myPCK,X_val)
assert np.allclose(Y_val, YmeanPCK, atol=5e-2)
print("Done.")
print("All good.")

