import pytest
import uqpylab.sessions as uq_session
import numpy as np
import time


def model(X):
    return X*np.sin(X)

def create_model_and_check_display(uq, name, MetaOpts):
    print(f"Computing a {name} metamodel...")
    myMeta = uq.createModel(MetaOpts)
    print("Done.")
    print(f"Checking display functionality...")
    uq.display(myMeta, test_mode=True)
    print("All good.")

# @pytest.mark.skip(reason="Leaves garbage (open figures) in current version")
def test_PCE_display(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')
    # Prepare the experimental design
    # & validation set from a simple model
    m_i = {
            'Type':'Uniform',
            'Parameters' : [-np.pi, np.pi]
            }
    InputOpts = {
        'Marginals': [m_i for i in [0,1,2] ]
        }
    print("Creating the input...")
    myInput = uq.createInput(InputOpts)
    print("Done.")
    
    print("Generating samples...")
    X_val = uq.getSample(N=1000)
    print("Done.")

    # create a default model
    print("Creating a true model object...")
    ModelOpts = {'Type' : 'Model',
        'ModelFun': 'uqpylab.test.true_models.ishigami'}
    myModel = uq.createModel(ModelOpts)
    print("Done.")
    print("Doing some model evaluations...")
    Y_val = model(X_val)
    print("Done.")

    # do quadrature PCE
    MetaOptsPCE = {
        'Type': 'Metamodel',
        'MetaType' : 'PCE',
        'FullModel': myModel['Name'],
        'Input': myInput['Name'],
        'Method' : 'Quadrature',
        'Degree' : 14
    }
    create_model_and_check_display(uq, "Quadrature PCE", MetaOptsPCE)
    
    print("Test complete.")

def test_PCE_multipleOutputsModel_display(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')
    # Prepare the experimental design
    # & validation set from a simple model
    InputOpts = {
        'Marginals': [
            {'Name': 'b', 'Type': 'Lognormal', 'Moments': [0.15, 0.0075]}, # beam width (m)
            {'Name': 'h', 'Type': 'Lognormal', 'Moments': [0.3, 0.015]},   # beam height (m)
            {'Name': 'L', 'Type': 'Lognormal', 'Moments': [5, 0.05]},      # beam length (m)
            {'Name': 'E', 'Type': 'Lognormal', 'Moments': [3e10, 4.5e9] }, # Young's modulus (Pa)
            {'Name': 'p', 'Type': 'Lognormal', 'Moments': [1e4, 1e3] },    # uniform load (N/m) 
        ]
    }
    print("Creating the input...")
    myInput = uq.createInput(InputOpts)
    print("Done.")

    # create a default model
    print("Creating a true model object...")
    ModelOpts = {'Type' : 'Model',
        'ModelFun': 'uqpylab.test.true_models.simply_supported_beam_9points'}
    myModel = uq.createModel(ModelOpts)
    print("Done.")

    MetaOptsPCE = {
        'Type': 'Metamodel',
        'MetaType' : 'PCE',
        'FullModel': myModel['Name'],
        'Input': myInput['Name'],
        'TruncOptions': {'qNorm': 0.75},
        'Degree': np.arange(2,11).tolist(),
        'ExpDesign': {
            "NSamples" : 10,
            "Sampling" : "LHS",
        }
    }
    create_model_and_check_display(uq, "Quadrature PCE", MetaOptsPCE)
    
    print("Test complete.")    

def test_Input_display_2independentVars(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    # 2 independent variables
    iOpts = {'Marginals': [uq.Marginals(1,'Gaussian',[-1,1])] + [uq.Marginals(1,'Exponential',[1])]}
    myInput = uq.createInput(iOpts)
    # pytest.self_trace()
    uq.display(myInput, test_mode=True)

def test_Input_display_3independentVars(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    # 3 independent variables
    iOpts = {'Marginals': [uq.Marginals(1,'Gaussian',[-1,1])] + [uq.Marginals(1,'Exponential',[1])] + [uq.Marginals(1, 'Uniform',[-1,3])]}
    myInput = uq.createInput(iOpts)
    uq.display(myInput,test_mode=True)

def test_Input_display_3independentVars_PDF_CDF(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    # 3 independent variables - show PDF and CDF of each component
    iOpts = {'Marginals': [uq.Marginals(1,'Gaussian',[-1,1])] + [uq.Marginals(1,'Exponential',[1])] + [uq.Marginals(1, 'Uniform',[-1,3])]}
    myInput = uq.createInput(iOpts)
    uq.display(myInput, plot_density=True,test_mode=True)
    # show PDF and CDF for only marginal 1 and 3
    uq.display(myInput, idx=[1,3], plot_density=True,test_mode=True)

def test_Input_display_VineCopula(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    # show vine
    iOpts = {
        "Marginals": uq.StdNormalMarginals(2) +
                    [uq.Marginals(1, 'Exponential', [1.5])],
        "Copula": uq.VineCopula('Cvine', [2,1,3],['t', 'Frank', 'Independence'],[[.4, 2], .5, []])
    }
    myInput = uq.createInput(iOpts)
    uq.display(myInput,show_vine=True,test_mode=True)   

def test_Inversion_display(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    ModelOpts = {
        "Type": "Model",
        "mString": "(5/32)*(X(:, 5).*X(:, 3).^4)./(X(:, 4).*X(:, 1).*X(:, 2).^3)",
    }
    myModel = uq.createModel(ModelOpts)

    PriorOpts = {
        "Marginals": [
            {"Name": "b", "Type": "LogNormal", "Moments": [0.15, 0.0075],},  # beam width (m)
            {"Name": "h", "Type": "LogNormal", "Moments": [0.3, 0.015],},  # beam height (m)
            {"Name": "L", "Type": "LogNormal", "Moments": [5, 0.05],},  # beam length (m)
            {"Name": "E", "Type": "LogNormal", "Moments": [3e10, 4.5e9],},  # Young's modulus (N/m^2)
            {"Name": "p", "Type": "LogNormal", "Moments": [1e4, 2e3],},  # constant distributed load (N/m)
        ]
    }
    myPriorDist = uq.createInput(PriorOpts)
    V_mid = np.array([12.84, 13.12, 12.13, 12.19, 12.67]) / 1000  # (m)
    myData = {
        "y": V_mid.tolist(),
        "Name": "Beam mid-span deflection",
    }
    BayesOpts = {
        "Type": "Inversion",
        "Data": myData,
    }
    myBayesianAnalysis = uq.createAnalysis(BayesOpts)
    pointEstimate = 'mean'
    myBayesianAnalysis = uq.postProcessInversion(myBayesianAnalysis,
                            'badChains', [1,2,3], 
                            'pointEstimate', pointEstimate, 
                            'burnIn', .5,  
                            'percentiles', [.025, .975], 
                            'dependence', 1, 
                            'prior', 1000, 
                            'priorPredictive', 1000, 
                            'posteriorPredictive', 1000, 
                            'gelmanRubin', 1,                        
                        )
    fig = uq.display(myBayesianAnalysis, trace='all', test_mode=True)
    assert len(fig) == 6
    fig = uq.display(myBayesianAnalysis, trace=[1,5,3], test_mode=True)
    assert len(fig) == 3
    fig = uq.display(myBayesianAnalysis, acceptance=True, test_mode=True)
    assert len(fig) == 1
    fig = uq.display(myBayesianAnalysis, scatterplot='all', test_mode=True)
    assert len(fig) == 2
    fig = uq.display(myBayesianAnalysis, predDist=True, test_mode=True)
    assert len(fig) == 1
    fig = uq.display(myBayesianAnalysis, meanConvergence='all', test_mode=True)
    assert len(fig) == 6
    fig = uq.display(myBayesianAnalysis, test_mode=True)
    assert len(fig) == 3
    print("All good...")

def test_PCK_display_1D(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    ModelOpts = {
        'Type': 'Model',
        'ModelFun': 'uqpylab.test.true_models.XsinX'
    }
    myModel = uq.createModel(ModelOpts)
    InputOpts = {"Marginals": uq.Marginals(M=1, Type="Uniform", Parameters=[0,15])}
    myInput = uq.createInput(InputOpts)
    X = uq.getSample(myInput, 10, 'Sobol')
    Y = uq.evalModel(myModel, X)
    SeqPCKOpts = {
        "Type": "Metamodel",
        "MetaType": "PCK",
        "Mode": "Sequential",
        "ExpDesign": {
            "X": X.tolist(),
            "Y": Y.tolist(),
        },
        "PCE": {"Degree": np.arange(1,11).tolist()}
    }
    mySeqPCK = uq.createModel(SeqPCKOpts)
    uq.print(mySeqPCK)
    fig = uq.display(mySeqPCK,test_mode=True);    
    assert len(fig) == 1

def test_PCK_display_2D(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    ModelOpts = {
        'Type': 'Model',
        'ModelFun': 'uqpylab.test.true_models.branin',
        'Parameters': {
            'a': 1,
            'b': 5.1 / (2 * np.pi) ** 2,
            'c': 5 / np.pi,
            'r': 6,
            's': 10,
            't': 1 / (8 * np.pi),
        }
    }    
    myModel = uq.createModel(ModelOpts)
    InputOpts = {"Marginals": [uq.Marginals(M=1, Type="Uniform", Parameters=[-5,10])]+ 
                              [uq.Marginals(M=1, Type="Uniform", Parameters=[0,15])]}
    myInput = uq.createInput(InputOpts)
    X = uq.getSample(myInput, 10, 'Sobol')
    Y = uq.evalModel(myModel, X)
    SeqPCKOpts = {
        "Type": "Metamodel",
        "MetaType": "PCK",
        "Mode": "Sequential",
        "ExpDesign": {
            "X": X.tolist(),
            "Y": Y.tolist(),
        },
    }
    mySeqPCK = uq.createModel(SeqPCKOpts)
    uq.print(mySeqPCK)  
    fig = uq.display(mySeqPCK,test_mode=True);    
    assert len(fig) == 2    

def test_PCK_display_1D_multipleOutputsModel(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')    
    
    ModelOpts = {
        'Type': 'Model',
        'ModelFun': 'uqpylab.test.true_models.XsinX_multipleOutputModel'
    }
    myModel = uq.createModel(ModelOpts)
    InputOpts = {"Marginals": uq.Marginals(M=1, Type="Uniform", Parameters=[0,15])}
    myInput = uq.createInput(InputOpts)
    X = uq.getSample(myInput, 10, 'Sobol')
    Y = uq.evalModel(myModel, X)
    SeqPCKOpts = {
        "Type": "Metamodel",
        "MetaType": "PCK",
        "Mode": "Sequential",
        "ExpDesign": {
            "X": X.tolist(),
            "Y": Y.tolist(),
        },
    }
    mySeqPCK = uq.createModel(SeqPCKOpts)
    fig = uq.display(mySeqPCK,outArray = [1,2,3,4],test_mode=True);    
    assert len(fig) == 4 # should print 4 images       


def test_PCK_display_2D_multipleOutputsModel(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')    
    
    ModelOpts = {
        'Type': 'Model',
        'ModelFun': 'uqpylab.test.true_models.hat2d_multipleOutputModel'
    }
    myModel = uq.createModel(ModelOpts)
    InputOpts = {"Marginals": uq.Marginals(M=2, Type="Gaussian", Parameters=[0.25,1])}
    myInput = uq.createInput(InputOpts)
    X = uq.getSample(myInput, 10, 'Sobol')
    Y = uq.evalModel(myModel, X)
    SeqPCKOpts = {
        "Type": "Metamodel",
        "MetaType": "PCK",
        "Mode": "Sequential",
        "ExpDesign": {
            "X": X.tolist(),
            "Y": Y.tolist(),
        },
    }
    mySeqPCK = uq.createModel(SeqPCKOpts)
    fig = uq.display(mySeqPCK,outArray = [1,2],test_mode=True);    
    assert len(fig) == 4 # should print 2*2 images

def test_Kriging_display_1D(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    ModelOpts = {
        'Type': 'Model',
        'ModelFun': 'uqpylab.test.true_models.XsinX'
    }
    myModel = uq.createModel(ModelOpts)
    InputOpts = {"Marginals": uq.Marginals(M=1, Type="Uniform", Parameters=[0,15])}
    myInput = uq.createInput(InputOpts)
    X = uq.getSample(myInput, 10, 'Sobol')
    Y = uq.evalModel(myModel, X)
    MetaOpts = {
        "Type": "Metamodel",
        "MetaType": "Kriging",
        "ExpDesign": {
            "X": X.tolist(),
            "Y": Y.tolist(),
        },
    }
    myKrigingMat = uq.createModel(MetaOpts)
    fig = uq.display(myKrigingMat,test_mode=True);    
    assert len(fig) == 1

def test_Kriging_display_1D_multipleOutputsModel(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')    
    
    ModelOpts = {
        'Type': 'Model',
        'ModelFun': 'uqpylab.test.true_models.XsinX_multipleOutputModel'
    }
    myModel = uq.createModel(ModelOpts)
    InputOpts = {"Marginals": uq.Marginals(M=1, Type="Uniform", Parameters=[0,15])}
    myInput = uq.createInput(InputOpts)
    X = uq.getSample(myInput, 10, 'Sobol')
    Y = uq.evalModel(myModel, X)
    MetaOpts = {
        "Type": "Metamodel",
        "MetaType": "Kriging",
        "ExpDesign": {
            "X": X.tolist(),
            "Y": Y.tolist(),
        },
    }
    myKrigingMat = uq.createModel(MetaOpts)
    fig = uq.display(myKrigingMat,outArray = [1,2,3,4],test_mode=True);    
    assert len(fig) == 4 # should print 4 images       

def test_Kriging_display_2D(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')    
    
    ModelOpts = {
        'Type': 'Model',
        'ModelFun': 'uqpylab.test.true_models.branin',
        'Parameters': {
            'a': 1,
            'b': 5.1 / (2 * np.pi) ** 2,
            'c': 5 / np.pi,
            'r': 6,
            's': 10,
            't': 1 / (8 * np.pi),
        }
    }    
    myModel = uq.createModel(ModelOpts)
    InputOpts = {"Marginals": [uq.Marginals(M=1, Type="Uniform", Parameters=[-5,10])]+ 
                              [uq.Marginals(M=1, Type="Uniform", Parameters=[0,15])]}
    myInput = uq.createInput(InputOpts)
    X = uq.getSample(myInput, 10, 'Sobol')
    Y = uq.evalModel(myModel, X)

    MetaOpts = {
        "Type": "Metamodel",
        "MetaType": "Kriging",
        "ExpDesign": {
            "X": X.tolist(),
            "Y": Y.tolist(),
        },
    }
    myKrigingMat = uq.createModel(MetaOpts)
    fig = uq.display(myKrigingMat,test_mode=True);    
    assert len(fig) == 2 

def test_Kriging_display_2D_InputAvail(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')    
    
    ModelOpts = {
        'Type': 'Model',
        'ModelFun': 'uqpylab.test.true_models.branin',
        'Parameters': {
            'a': 1,
            'b': 5.1 / (2 * np.pi) ** 2,
            'c': 5 / np.pi,
            'r': 6,
            's': 10,
            't': 1 / (8 * np.pi),
        }
    }    
    myModel = uq.createModel(ModelOpts)
    InputOpts = {"Marginals": [uq.Marginals(M=1, Type="Uniform", Parameters=[-5,10])]+ 
                              [uq.Marginals(M=1, Type="Uniform", Parameters=[0,15])]}
    myInput = uq.createInput(InputOpts)

    MetaOpts = {
        "Type": "Metamodel",
        "MetaType": "Kriging",
        "ExpDesign": {
            "Sampling": "LHS",
            "NSamples": 35
        },
        "Input": myInput["Name"],
        "FullModel": myModel["Name"],
    }

    myKrigingMat = uq.createModel(MetaOpts)
    fig = uq.display(myKrigingMat,test_mode=True);    
    assert len(fig) == 2 

def test_Kriging_display_2D_multipleOutputsModel(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')    
    
    ModelOpts = {
        'Type': 'Model',
        'ModelFun': 'uqpylab.test.true_models.hat2d_multipleOutputModel'
    }
    myModel = uq.createModel(ModelOpts)
    InputOpts = {"Marginals": uq.Marginals(M=2, Type="Gaussian", Parameters=[0.25,1])}
    myInput = uq.createInput(InputOpts)
    X = uq.getSample(myInput, 10, 'Sobol')
    Y = uq.evalModel(myModel, X)
    MetaOpts = {
        "Type": "Metamodel",
        "MetaType": "Kriging",
        "ExpDesign": {
            "X": X.tolist(),
            "Y": Y.tolist(),
        },
    }
    myKriging = uq.createModel(MetaOpts)
    fig = uq.display(myKriging,outArray = [1,2],test_mode=True);    
    assert len(fig) == 4 # should print 4 images

def test_Kriging_display_customKriging(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')    
    
    ModelOpts = {
        'Type': 'Model',
        'ModelFun': 'uqpylab.test.true_models.XsinX'
    }
    myModel = uq.createModel(ModelOpts)
    InputOpts = {"Marginals": uq.Marginals(M=1, Type="Uniform", Parameters=[-5,5])}
    myInput = uq.createInput(InputOpts)
    X = uq.getSample(myInput, 10)
    Y = uq.evalModel(myModel, X)
    Y = Y + 0.02*np.std(Y)*np.random.randn(Y.shape[0],1)

    MetaOpts = {
        'Type' : 'Metamodel',
        'MetaType': 'Kriging',
        'ExpDesign': {
            'Sampling': 'User',
            'X': X.tolist(),
            'Y': Y.tolist()
        },
        'Kriging': {
            'Trend':{
                'Type': 'ordinary'
            },
            'beta': -2,
            'sigmaSQ': 3.94372e-04,
            'theta': 0.8,
            'Corr': {'Type': 'ellipsoidal',
                    'Family': 'matern-5_2'}
        },
        'Optim':{
            'Display': 'final'
        },
        'Regression': {
            'SigmaNSQ': 'auto'           # unknown homoscedastic noise
        }
    }
    myKriging = uq.createModel(MetaOpts)
    fig = uq.display(myKriging,test_mode=True);    
    assert len(fig) == 1 

    mySession.quit()
    time.sleep(1)            