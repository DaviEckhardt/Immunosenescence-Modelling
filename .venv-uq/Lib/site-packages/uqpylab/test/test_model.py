import pytest
import uqpylab.sessions as uq_session
import numpy as np
import sys
from io import StringIO
import time

def model(X):
    return X*np.sin(X)

def create_and_eval_model(uq, name, MetaOpts, X_val, Y_val):
    print(f"Computing a {name} metamodel...")
    myMeta = uq.createModel(MetaOpts)
    print("Done.")
    print(f"Performing {name} model evaluations...")
    Y_Meta_val = uq.evalModel(myMeta, X_val)
    print("Done.")
    print(f"Checking {name} responses...")
    assert Y_Meta_val.shape == Y_val.shape
    print("All good.")

def test_model(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')
    # Prepare the experimental design
    # & validation set from a simple model
    InputOpts = {
        'Marginals': [
            {
            'Type':'Uniform',
            'Parameters' : [1, 5]
            }]
        }
    print("Creating an input...")
    myInput = uq.createInput(InputOpts)
    print("Done.")
    print("Generating samples...")
    X_ED = uq.getSample(myInput, 100)
    print("Done.")

    print("Generating samples...")
    X_val = uq.getSample(N=1000)
    print("Done.")


    # create a default model
    print("Creating a true model object...")
    ModelOpts = {'Type' : 'Model',
        'ModelFun': 'uqpylab.test.true_models.XsinX'}
    myModel = uq.createModel(ModelOpts)
    print("Done.")
    print("Doing some model evaluations...")
    Y_val = model(X_val)
    Y_ED = model(X_ED)
    print(Y_ED)

    print("Done.")

    # do PCE
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
    create_and_eval_model(uq, "PCE", MetaOptsPCE, X_val, Y_val)
    # do Kriging
    MetaOptsKRG = {
        'Type': 'Metamodel',
        'MetaType' : 'Kriging',
        'ExpDesign': {
            'X': X_ED.tolist(),
            'Y': Y_ED.tolist()},
        # We let all other options to default values 
    }
    create_and_eval_model(uq, "Kriging", MetaOptsKRG, X_val, Y_val)
    # do PC-Kriging
    MetaOptsPCK = {
        'Type': 'Metamodel',
        'MetaType' : 'PCK',
        'ExpDesign': {
            'X': X_ED.tolist(),
            'Y': Y_ED.tolist()},
        'Mode': 'sequential',
        'PCE': {
            'Degree': np.arange(1,11).tolist()
        }
    }
    create_and_eval_model(uq, "PCK", MetaOptsPCK, X_val, Y_val)
    # do PC-Kriging with Custom set A
    del MetaOptsPCK["PCE"]
    MetaOptsPCK["PolyTypes"] = ["Legendre"]
    MetaOptsPCK["PolyIndices"] = [1, 4, 6]
    create_and_eval_model(uq, "PCK", MetaOptsPCK, X_val, Y_val)

    # do LRA
    MetaOptsLRA = {
        'Type': 'Metamodel',
        'MetaType' : 'LRA',
        'ExpDesign': {
            'X': X_ED.tolist(),
            'Y': Y_ED.tolist()},
        'Rank': np.arange(1,11).tolist(),
        'Degree': np.arange(1,11).tolist(),
    }
    create_and_eval_model(uq, "LRA", MetaOptsLRA, X_val, Y_val)
    print("Test complete.")

def test_different_Kriging_settings(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    # Prepare the experimental design
    # & validation set from a simple model
    InputOpts = {'Marginals': uq.Marginals(M=1,Type='Uniform', Parameters=[0, 15])}
    print("Creating an input...")
    myInput = uq.createInput(InputOpts)
    print("Done.")
    print("Generating samples...")
    X_ED = uq.getSample(myInput, 10, 'Sobol')
    print("Done.")

    print("Generating samples...")
    X_val = uq.getSample(N=10)
    print("Done.")

    # create a default model
    print("Creating a true model object...")
    ModelOpts = {'Type' : 'Model',
        'ModelFun': 'uqpylab.test.true_models.XsinX'}
    myModel = uq.createModel(ModelOpts)
    print("Done.")
    print("Doing some model evaluations...")
    Y_val = uq.evalModel(myModel,X_val)
    Y_ED = uq.evalModel(myModel,X_ED)

    MetaOptsKRG = {
        'Type': 'Metamodel',
        'MetaType' : 'Kriging',
        'ExpDesign': {
            'X': X_ED.tolist(),
            'Y': Y_ED.tolist()},
        'Trend': {
            'Type': 'ordinary'
        },
        'Corr': {
            'Family': 'matern-5_2'
        },
        'EstimMethod': 'ML',
        'Optim': {
            'Method': 'BFGS'
        }
    }
    create_and_eval_model(uq, "Kriging", MetaOptsKRG, X_val, Y_val)

    MetaOptsKRG['Optim'] = {'Method': 'HGA'}
    create_and_eval_model(uq, "Kriging", MetaOptsKRG, X_val, Y_val)

    MetaOptsKRG['EstimMethod'] = 'CV'
    MetaOptsKRG['Optim'] = {'Method': 'BFGS'}
    create_and_eval_model(uq, "Kriging", MetaOptsKRG, X_val, Y_val)

    MetaOptsKRG['Optim']['MaxIter'] = 100
    MetaOptsKRG['Optim']['Tol'] = 1e-6
    MetaOptsKRG['Optim']['HGA'] = {'nPop': 40}
    create_and_eval_model(uq, "Kriging", MetaOptsKRG, X_val, Y_val)

    MetaOptsKRG['Trend'] = {
        'Type': 'polynomial',
        'Degree': 3
    }
    create_and_eval_model(uq, "Kriging", MetaOptsKRG, X_val, Y_val)

    MetaOptsKRG['Trend'] = {
        'Type': 'custom',
        'CustomF': '@(X) X.^2 + sqrt(abs(X))'
    }
    create_and_eval_model(uq, "Kriging", MetaOptsKRG, X_val, Y_val)
    print("Done.")

def test_different_PCE_settings(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    print("Testing different PCE settings")
    ModelOpts = {
        'Type': 'Model', 
        'ModelFun':'uqpylab.test.true_models.ishigami'
    }
    myModel = uq.createModel(ModelOpts)
    InputOpts = {'Marginals': uq.Marginals(M=3, Type='Uniform', Parameters=[-np.pi, np.pi])}
    myInput = uq.createInput(InputOpts)

    print("Quadrature-based calculation of the coefficients...")
    MetaOpts= {
        "Type": "Metamodel",
        "MetaType": "PCE",
        "Method": "Quadrature",
        "Degree": 14
    }
    myPCE = uq.createModel(MetaOpts)
    X = uq.getSample(myInput,10)
    Y = uq.evalModel(myPCE, X)
    assert Y.shape == (10,1)
    print("Done...")

    print("Least-square calculation of the coefficients...")
    MetaOpts["Method"] = "OLS"
    MetaOpts["Degree"] = np.arange(3,16).tolist()
    MetaOpts["ExpDesign"] = {
        "NSamples": 500,
        "Sampling": "LHS"
    }
    myPCE = uq.createModel(MetaOpts)
    X = uq.getSample(myInput,10)
    Y = uq.evalModel(myPCE, X)
    assert Y.shape == (10,1)
    print("Done...")

    print("Sparse Least-Angle-Regression-based (LARS) calculation of the coefficients...")
    MetaOpts["Method"] = "LARS"
    MetaOpts["Degree"] = np.arange(3,16).tolist()
    MetaOpts["TruncOptions"] = {"qNorm": 0.75}
    MetaOpts["ExpDesign"]["NSamples"] = 150
    myPCE = uq.createModel(MetaOpts)
    X = uq.getSample(myInput,10)
    Y = uq.evalModel(myPCE, X)
    assert Y.shape == (10,1)
    print("Done...")

    print("Orthogonal Matching Pursuit (OMP) calculation of the coefficients...")
    MetaOpts["Method"] = "OMP"
    MetaOpts["Degree"] = np.arange(3,16).tolist()
    MetaOpts["TruncOptions"] = {"qNorm": 0.75}
    myPCE = uq.createModel(MetaOpts)
    X = uq.getSample(myInput,10)
    Y = uq.evalModel(myPCE, X)
    assert Y.shape == (10,1)
    print("Done...")

    print("PCE with Legendre polynomials...")
    MetaOpts= {
        "Type": "Metamodel",
        "MetaType": "PCE",
        "Method": "LARS",
        "Degree": np.arange(3,16).tolist(),
        "PolyTypes": ['Legendre','Legendre','Legendre'],
        "ExpDesign": {"NSamples": 150}
    }
    myPCE = uq.createModel(MetaOpts)
    X = uq.getSample(myInput,10)
    Y = uq.evalModel(myPCE, X)
    assert Y.shape == (10,1)
    print("Done...")

    print("PCE with numerically estimated polynomials")
    MetaOpts["PolyTypes"] = ['Arbitrary','Arbitrary','Arbitrary']
    myPCE = uq.createModel(MetaOpts)
    X = uq.getSample(myInput,10)
    Y = uq.evalModel(myPCE, X)
    assert Y.shape == (10,1)
    print("Done...")

    print("Bootstrap PCE...")
    MetaOpts = {
        "Type": "Metamodel",
        "MetaType": "PCE",
        "FullModel" : myModel["Name"],
        "Degree": 11,
        "ExpDesign": {"NSamples": 15},
        "Bootstrap": {"Replications": 100}, 
    }
    myPCE = uq.createModel(MetaOpts)
    X = uq.getSample(myInput,10)
    Y = uq.evalModel(myPCE, X)
    assert Y.shape == (10,1)
    print("Done...")

def test_Kriging_noisy_model_response(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')
    InputOpts = {'Marginals': uq.Marginals(M=1,Type='Uniform', Parameters=[-3*np.pi, 3*np.pi])}
    myInput = uq.createInput(InputOpts)
    X = uq.getSample(myInput, 50)
    X_val = uq.getSample(myInput, 50)
    ModelOpts = {'Type' : 'Model',
        'ModelFun': 'uqpylab.test.true_models.XsinX',
        'isVectorized': 'true'}
    myModel = uq.createModel(ModelOpts)
    Y = uq.evalModel(myModel, X)
    # Unknown homogeneous (homoscedastic) noise
    Y = Y + 0.2*np.std(Y)*np.random.randn(Y.shape[0],1)
    MetaKRGOpts = {
        'Type': 'Metamodel',
        'MetaType': 'Kriging',
        'ExpDesign': {
            'X': X.tolist(),
            'Y': Y.tolist()            
        },
        'Regression': {
            'SigmaNSQ': 'auto'
        }
    }
    myKrigingRegression1 = uq.createModel(MetaKRGOpts)
    Y_KRG = uq.evalModel(myKrigingRegression1, X_val)
    assert Y.shape == Y_KRG.shape

    # Known homogeneous (homoscedastic) noise
    Y = uq.evalModel(myModel, X)
    noiseVar = 1.0
    Y = Y + np.sqrt(noiseVar)*np.random.randn(Y.shape[0],1) 
    MetaKRGOpts['ExpDesign'] = {
        'X': X.tolist(),
        'Y': Y.tolist()
    }
    MetaKRGOpts['Regression'] = {
        'SigmaNSQ': noiseVar
    }
    myKrigingRegression2 = uq.createModel(MetaKRGOpts)
    Y_KRG = uq.evalModel(myKrigingRegression2, X_val)
    assert Y.shape == Y_KRG.shape  

    # Know non-homogeneous (heteroscedastic) noise
    Y = uq.evalModel(myModel, X)
    noiseVar = np.power((0.3*np.abs(Y)), 2)
    Y = Y + np.sqrt(noiseVar) * np.random.randn(Y.shape[0],1)

    MetaKRGOpts['ExpDesign'] = {
        'X': X.tolist(),
        'Y': Y.tolist()
    }
    MetaKRGOpts['Regression'] = {
        'SigmaNSQ': noiseVar.tolist()
    }
    myKrigingRegression3 = uq.createModel(MetaKRGOpts)
    Y_KRG = uq.evalModel(myKrigingRegression3, X_val)
    assert Y.shape == Y_KRG.shape  

def test_parametricModel(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    ModelOpts = {
        'Type': 'Model', 
        'ModelFun':'uqpylab.test.true_models.ishigami_parametric',
        'isVectorized': True,
        'Parameters': {
            'a': 7,
            'b': 0.1
        }
    }
    myModel = uq.createModel(ModelOpts)
    InputOpts = {'Marginals': uq.Marginals(M=3, Type='Uniform', Parameters=[-np.pi, np.pi])}
    myInput = uq.createInput(InputOpts)
    X = uq.getSample(N=10,Method='LHS')
    Y = uq.evalModel(myModel,X)
    print(Y)
    assert Y.shape == (10,1)

    
def test_multiple_outputs(request, helpers):
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
    myInput = uq.createInput(InputOpts)
    print("Done.")
    print("Generating samples...")
    X_ED = uq.getSample(myInput, N=5)
    X_val = uq.getSample(N=10)
    print("Done.")
    Y_val = model(X_val)
    Y_ED = model(X_ED)
    print("Creating a Kriging surrogate...")
    MetaOptsKRG = {
        'Type': 'Metamodel',
        'MetaType' : 'Kriging',
        'ExpDesign': {
            'X': X_ED.tolist(),
            'Y': Y_ED.tolist()},
        # We let all other options to default values 
    }
    myKRG = uq.createModel(MetaOptsKRG)
    print("Done.")
    print("Testing vargout=1..3 calls of uq.evalModel...")
    Ymean0 = uq.evalModel(myKRG, X_val, nargout=1)
    [Ymean1, Yvar1] = uq.evalModel(myKRG, X_val, nargout=2)
    [Ymean2, Yvar2, Ycov] = uq.evalModel(myKRG, X_val, nargout=3)
    assert (Ymean0 == Ymean1).all()
    assert (Ymean0 == Ymean2).all()
    print(np.max(np.abs(Yvar1 -  Yvar2)))
    assert np.allclose(Yvar1, Yvar2, atol=5e-7)
    assert Ycov.shape == (Ymean0.shape[0],Ymean0.shape[0])
    print("All good.")

def test_scalarSample(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    # Prepare the experimental design
    # & validation set from a simple model
    print("Creating an input...")
    InputOpts = {
        'Marginals': [
            {
            'Type':'Uniform',
            'Parameters' : [1, 5]
            }]
        }
    myInput = uq.createInput(InputOpts)
    print("Done.")
    # create a default model
    print("Creating a true model object...")
    ModelOpts = {'Type' : 'Model',
        'ModelFun': 'uqpylab.test.true_models.XsinX'}
    myModel = uq.createModel(ModelOpts)
    print("Done.")
    print("Creating PCK model...")
    MetaOpts = {
        "Type": "Metamodel",
        "MetaType": "PCK",
        "ExpDesign": {
            "Sampling": "Sobol",
            "NSamples": 10
        },
        "Mode": "sequential"
    }
    myPCK = uq.createModel(MetaOpts)
    print("Done...")
    print("Testing scalars...")
    [ySPCK, ySPCKs2] = uq.evalModel(myPCK, 3, nargout=2)
    [ySPCK_2, ySPCKs2_2] = uq.evalModel(myPCK, 3., nargout=2)
    assert np.allclose(ySPCK[0][0], ySPCK_2[0][0], atol=5e-5)
    assert np.allclose(ySPCKs2[0][0], ySPCKs2_2[0][0], atol=5e-5)
    print("Test done...")

def test_extractFromModel(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')
    # Prepare the experimental design
    # & validation set from a simple model
    print("Creating an input...")
    InputOpts = {
        'Marginals': [
            {
            'Type':'Uniform',
            'Parameters' : [1, 5]
            }]
        }
    myInput = uq.createInput(InputOpts)
    print("Done.")
    # create a default model
    print("Creating a true model object...")
    ModelOpts = {'Type' : 'Model',
        'ModelFun': 'uqpylab.test.true_models.XsinX'}
    myModel = uq.createModel(ModelOpts)
    print("Done.")
    print("Creating PCK model...")
    MetaOpts = {
        "Type": "Metamodel",
        "MetaType": "PCK",
        "ExpDesign": {
            "Sampling": "Sobol",
            "NSamples": 10
        },
        "Mode": "sequential"
    }
    myPCK = uq.createModel(MetaOpts)
    print("Done...")
    print("Extracting PCE model...")
    myPCE = uq.extractFromModel(parentName=myPCK['Name'],objPath='Internal.PCE')
    assert isinstance(myPCE, dict)
    print("Done")

    print("Extracting Kriging model...")
    myKriging = uq.extractFromModel(parentName=myPCK['Name'],objPath='Internal.Kriging')
    assert isinstance(myKriging, dict)
    print("Done")
    print("All good.")    

def test_assignInputandFullModel(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    print("Creating an input...")
    InputOpts = {
        'Marginals': [
            {
            'Type':'Uniform',
            'Parameters' : [0, 15]
            }]
        }
    myInput = uq.createInput(InputOpts)
    print("Done.")
    # create a default model
    print("Creating a true model object...")
    ModelOpts = {'Type' : 'Model',
        'ModelFun': 'uqpylab.test.true_models.XsinX'}
    myModel = uq.createModel(ModelOpts)
    print("Done.")
    print("Creating PCK model...")
    MetaOpts = {
        "Type": "Metamodel",
        "MetaType": "PCK",
        "ExpDesign": {
            "Sampling": "Sobol",
            "NSamples": 10
        },
        "Mode": "sequential",
        "Input": myInput["Name"],
        "FullModel": myModel["Name"],        
    }
    myPCK = uq.createModel(MetaOpts)
    print("All good.")


def test_MultipleOutputModels(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
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
        }
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
        "PCE": {"Degree": 3}
    }
    myPCK = uq.createModel(MetaOptsPCK)
    print("Done.")
    print("Evaluating PCK model.")
    YmeanPCK = uq.evalModel(myPCK,X_val)
    assert np.allclose(Y_val, YmeanPCK, atol=5e-2)
    print("Done.")
    print("All good.")

def test_getModel_listModels(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    ModelOpts1 = {
        'Type': 'Model',
        'ModelFun': 'uqpylab.test.true_models.simply_supported_beam_9points',
        'isVectorized': 'true'
    }
    myModel1 = uq.createModel(ModelOpts1)

    ModelOpts2 = {
        'Type': 'Model',
        'ModelFun': 'uqpylab.test.true_models.XsinX',
        'isVectorized': 'true'
    }
    myModel2 = uq.createModel(ModelOpts2)

    InputOpts = {
        'Marginals': {
            'Type': 'Uniform',
            'Parameters': [0, 15],
        }
    }

    myInput = uq.createInput(InputOpts)

    X = uq.getSample(N=8)
    Y = uq.evalModel(X=X)
    MetaOpts = {
        'Type': 'Metamodel',
        'MetaType': 'Kriging',
        'ExpDesign': {
            'X': X.tolist(),
            'Y': Y.tolist(),
        }
    }
    myKriging = uq.createModel(MetaOpts)

    print("Check if listModels() works, redirecting stdout and checking the output...")

    # Redirect stdout to a StringIO object
    captured_output = StringIO()
    sys.stdout = captured_output

    # Now anything printed will be captured
    uq.listModels()

    # Access the captured output
    captured_output.seek(0)  # Move cursor to the start of StringIO object
    captured_text = captured_output.read()

    # Restore sys.stdout to its original value
    sys.stdout = sys.__stdout__

    assert all(model['Name'] in captured_text for model in [myModel1, myModel2, myKriging])
    print("Looks fine...")

    print('Get the current model')
    myModel4 = uq.getModel('') 
    assert myModel4['Type'] == myKriging['Type']  and  myModel4['Name'] == myKriging['Name']
    print("Looks fine...")

    print('Get the first model using a name')
    myModel5 = uq.getModel(myModel1['Name'])
    assert myModel5['Name'] == myModel1['Name']   
    print("Looks fine...")

    print('Get the first model using an object')
    myModel6 = uq.getModel(myModel1)
    assert myModel6['Name'] == myModel1['Name']   
    print("Looks fine...")

    print("All good!")

# def test_Kriging_ED_Input(request, helpers):
#     mySession = helpers.init_session(request)
#     uq = mySession.cli
#     uq.rng(100, 'twister')

#     ModelOpts = {
#         'Type': 'Model',
#         'ModelFun': 'uqpylab.test.true_models.branin',
#         'Parameters': {
#             'a': 1,
#             'b': 5.1 / (2 * np.pi) ** 2,
#             'c': 5 / np.pi,
#             'r': 6,
#             's': 10,
#             't': 1 / (8 * np.pi),
#         }
#     }    
#     myModel = uq.createModel(ModelOpts)
#     InputOpts = {"Marginals": [uq.Marginals(M=1, Type="Uniform", Parameters=[-5,10])]+ 
#                               [uq.Marginals(M=1, Type="Uniform", Parameters=[0,15])]}
#     myInput = uq.createInput(InputOpts)
#     X = uq.getSample(myInput, 10, 'Sobol')
#     Y = uq.evalModel(myModel, X)

#     MetaOpts = {
#         "Type": "Metamodel",
#         "MetaType": "Kriging",
#         "ExpDesign": {
#             "X": X.tolist(),
#             "Y": Y.tolist(),
#         },
#         "Input": myInput["Name"],
#         "FullModel": myModel["Name"],
#     }
#     myKrigingMat = uq.createModel(MetaOpts)
    mySession.quit()
    time.sleep(1)    