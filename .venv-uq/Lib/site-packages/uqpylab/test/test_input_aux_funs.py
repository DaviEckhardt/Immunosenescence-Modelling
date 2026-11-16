import pytest
import uqpylab.sessions as uq_session
#from uqlab_standalone import sessions as uq_session
import numpy as np
import sys
from io import StringIO
import time


def test_input_aux_funs(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')
    iOptsTrue = {
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
        ]
    }
    myInputTrue = uq.createInput(iOptsTrue)

    X = uq.getSample(myInputTrue,200)

    iOpts = {
        'Inference': 
            {
                'Data': X.tolist()
            },
    }
    iOpts['Copula'] = {
        'Type': 'auto',
        'Inference': {
            'BlockIndepTest': {
                'Alpha': 0.05
            }
        },
        'Parameters': []
    }
    InputHat1c = uq.createInput(iOpts)
    
    # make sure that the session is properly terminated (no dangling background stuff)
    assert X.shape == (200,3)

def test_marginal_funs(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    iOptsTrue = {       
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
        ]
    }

    myInputTrue = uq.createInput(iOptsTrue)

    # X = uq.getSample(myInputTrue,200, response_format='JSON')
    # U = uq.all_cdf(X, myInputTrue['Marginals'], request_format='JSON', response_format='JSON')    
    X = uq.getSample(myInputTrue,10)    
    F = uq.all_cdf(X, myInputTrue['Marginals'])
    assert X.shape == F.shape

    f = uq.all_pdf(X, myInputTrue['Marginals'])
    assert X.shape == f.shape

    Xnew = uq.all_invcdf(F, myInputTrue['Marginals'])
    assert np.allclose(X, Xnew, atol=1e-5)

def enrichSampling(request, helpers, SamplingType, FunctionName):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')    

    Input = {"Marginals": [uq.Marginals(1, "Gaussian", [2, .5])]+
                          [uq.Marginals(1, "Gaussian", [1, 1])]}
    myInput = uq.createInput(Input)

    X = uq.getSample(myInput, 100, SamplingType)   
    numEnr = 50
    # one output argument
    XX = eval(FunctionName+"(X0=X, N=numEnr)")
    assert XX.shape == (numEnr,2)
    # two output arguments
    X, U = eval(FunctionName+"(X0=X, N=numEnr, nargout=2)")
    assert X.shape == (numEnr,2)
    assert U.shape == (numEnr,2)

def test_enrichLHS(request, helpers):
    enrichSampling(request, helpers, 'LHS', 'uq.enrichLHS')
def test_enrichSobol(request, helpers):
    enrichSampling(request, helpers, 'Sobol', 'uq.enrichSobol')
def test_enrichHalton(request, helpers):
    enrichSampling(request, helpers, 'Halton', 'uq.enrichHalton')
def test_LHSify(request, helpers):
    enrichSampling(request, helpers, 'MC', 'uq.LHSify')

def test_appendVariables(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    iOpts = {"Marginals": uq.StdNormalMarginals(2)}
    myInput = uq.createInput(iOpts)
    X = uq.getSample(myInput, 200)
    assert X.shape == (200,2)
    iOpts['Marginals'].extend(uq.StdNormalMarginals(1))
    myInput = uq.createInput(iOpts)
    X = uq.getSample(myInput, 200)
    assert X.shape == (200,3)

def test_CopulaSummary(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')
    # Marginals - 2 random variables
    iOpts = {"Marginals": uq.StdNormalMarginals(2)}
    myInput = uq.createInput(iOpts)
    msg = uq.CopulaSummary(myInput['Copula'], no_print=True)
    assert msg is not None
    # Pair copula
    iOpts['Copula'] = {
        'Type': 'Pair',
        'Family': 'Clayton',
        'Rotation': 90,
        'Parameters': 1.5
    }
    myInput = uq.createInput(iOpts)
    msg = uq.CopulaSummary(myInput['Copula'], no_print=True)
    assert msg is not None
    # Marginals - 3 random variables
    iOpts = {'Marginals': uq.StdNormalMarginals(2) + 
                          [uq.Marginals(1, 'Exponential', [1.5])]}
    # Gaussian copula
    iOpts['Copula'] = uq.GaussianCopula([[1, .5, -.3], [.5, 1, .2],[-.3, .2, 1]], 'Linear')
    myInput = uq.createInput(iOpts)
    msg = uq.CopulaSummary(myInput['Copula'], no_print=True)
    assert msg is not None

    # Vine copula
    iOpts['Copula'] = uq.VineCopula('Cvine', [2,1,3],['Gumbel', 'Gaussian', 'Frank'],[1.5, -0.4, [0.3]])          
    # msg = uq.CopulaSummary(iOpts['Copula'], no_print=True)
    # print(msg)
    myInput = uq.createInput(iOpts)
    msg = uq.CopulaSummary(myInput['Copula'], no_print=True)
    assert msg is not None

def test_ConstantVariables(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    Input = {
        'Marginals': [
            {
                'Type': 'Gaussian',
                'Parameters': [0,1],
                'Bounds': [0,0]
            },   
            {
                'Type': 'Gaussian',
                'Parameters': [1, 0]
            },        
            {
                'Type': 'Constant',
                'Parameters': [2]
            }
        ]
    }
    myInput = uq.createInput(Input)
    X = uq.getSample(myInput, 1)
    assert (X == np.array([0,1,2])).all()

def test_estimateMoments(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    Input = {
        'Marginals': 
            {
                'Type': 'Lognormal',
                'Parameters': [2, 0.5]
            }
    }
    moments = uq.estimateMoments(Input['Marginals'])   
    assert abs(moments[0]-np.exp(2+.5**2/2)) < 1e-5
    assert abs(moments[1]-((np.exp(.5**2)-1)*np.exp(2*2+.5**2))**.5) < 1e-5

def test_Transforms(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')    

    # 2D Gaussian Independent variables and Gaussian copula
    Input1 = {
        'Marginals': [uq.Marginals(1, 'Gaussian', [1,1])]+ 
                        [uq.Marginals(1, 'Gaussian', [2,.5])],
        'Copula': {
            'Type': 'Gaussian',
            'Parameters': [[1, 0.8],[0.8, 1]] 
        }
    }
    myInput1 = uq.createInput(Input1)

    # 2D Standard normal independent variables 
    Input2 = {
        'Marginals': uq.StdNormalMarginals(2),
        'Copula'   :  {'Type': 'Independent'}
    }

    # 2D Standard uniform independent variables
    Input3 = {"Marginals": uq.StdUniformMarginals(2)}

    X = uq.getSample(myInput1, 100)
    assert X.shape == (100, 2)

    ## General isoprobabilistic transform
    U = uq.GeneralIsopTransform(X, 
            Input1['Marginals'], Input1['Copula'], 
            Input2['Marginals'], Input2['Copula'])  
    assert U.shape == (100, 2)

    ## Isoprobabilistic transform
    V = uq.IsopTransform(X, Input2['Marginals'], Input3['Marginals'])
    assert V.shape == (100, 2)

    ## Nataf transform
    Unew = uq.NatafTransform(X, Input1['Marginals'], Input1['Copula'])
    assert Unew.shape == (100, 2)

    ## Inverse Nataf transform
    Xnew = uq.invNatafTransform(U, Input1['Marginals'], Input1['Copula'])
    assert Xnew.shape == (100, 2)

    ## Rosenblatt transform
    Vnew = uq.RosenblattTransform(X, Input1['Marginals'], Input1['Copula'])    
    assert Vnew.shape == (100, 2)

    ## Inverse Rosenblatt transform
    Xnew2 = uq.invRosenblattTransform(V, Input1['Marginals'], Input1['Copula'])
    assert Xnew2.shape == (100, 2)

def test_MarginalFields(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    Input = {
        'Marginals':  [
            {'Type': 'Constant',    'Parameters': [1]},
            {'Type': 'Gaussian',    'Parameters': [2, 0.5]},
            {'Type': 'Lognormal',   'Parameters': [0, .1]},
            {'Type': 'Uniform',     'Parameters': [-.5, .5]},
            {'Type': 'Exponential', 'Parameters': [1.5]},
            {'Type': 'Beta',        'Parameters': [.8, .2, .5, 1.5]},
            {'Type': 'Weibull',     'Parameters': [1.5, 1]},
            {'Type': 'Gumbel',      'Parameters': [0, .2]},
            {'Type': 'GumbelMin',   'Parameters': [0, .2]},
            {'Type': 'Gamma',       'Parameters': [1, 1]},
            {'Type': 'Triangular',  'Parameters': [0, 2, 1]},
            {'Type': 'Logistic',    'Parameters': [5, 2]},
            {'Type': 'Laplace',     'Parameters': [0, 1]},
            {'Type': 'Rayleigh',    'Parameters': [1]},
        ]
    }

    marginals = uq.MarginalFields(Input['Marginals'])
    Moments = [marginal['Moments'] for marginal in marginals]
    MomentsUQLab = np.array([
        [1,	0],
        [2,	0.5], 
        [1.00501252085940,	0.100753029446204], 
        [0,	0.288675134594813], 
        [0.666666666666667,	0.666666666666667],
        [1.3,	0.282842712474619],
        [1.5,	1.5],
        [0.115443132980306,	0.256509966032373],
        [-0.115443132980306,	0.256509966032373],
        [1,	1],
        [1,	0.408248290463863],
        [5,	3.62759872846844], 
        [0,	1.41421356237310],
        [1.25331413731550,	0.655136377562034]])

    assert (abs(Moments - MomentsUQLab) < 1e-5).all()

def test_Copulas(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')   

    ## Pair copula
    print("Pair copula defined via fuction...")
    iOpts = {
        "Marginals" : uq.StdNormalMarginals(2),
        "Copula":  uq.PairCopula('Gumbel', 1.5)
    }
    myInput = uq.createInput(iOpts)
    X = uq.getSample(myInput,50)
    assert X.shape == (50,2)

    print("Pair copula defined via dictionary")
    iOpts["Copula"] = {
        'Type': 'Pair',
        'Family': 'Gumbel',
        'Parameters': 1.5       
    }
    myInput2 = uq.createInput(iOpts)
    uq.rng(100, 'twister')
    X2 = uq.getSample(myInput2, 50)
    assert X2.shape == (50,2)
    # samples should be identical from creating the copula via uq.PairCopula 
    # and defining it directly using keyword-value pairs:
    assert np.allclose(X, X2, atol=5e-5)
    print("All good...")

    ## Gaussian copula
    print("Gaussian copula defined via dictionary...")
    iOpts['Copula'] = {
        'Type': 'Gaussian',
        'RankCorr': [[1, 0.8],[0.8, 1]] # the Spearman corr. matrix
    }
    myInput = uq.createInput(iOpts)
    uq.rng(100, 'twister')
    X = uq.getSample(myInput,50)
    assert X.shape == (50,2)

    print("Gaussian copula defined via fuction...")
    iOpts['Copula'] = uq.GaussianCopula([[1, 0.8],[0.8, 1]], 'Spearman')
    myInput2 = uq.createInput(iOpts)
    uq.rng(100, 'twister')
    X2 = uq.getSample(myInput2,50)
    assert X2.shape == (50,2)    

    assert np.allclose(X, X2, atol=5e-5)
    print("All good...")

    ## Vine copula
    print("Vine copula defined via fuction...")
    iOpts = {
        "Marginals": uq.StdNormalMarginals(2) +
                    [uq.Marginals(1, 'Exponential', [1.5])],
        "Copula": uq.VineCopula('Cvine', [2,1,3],['t', 'Frank', 'Independence'],[[.4, 2], .5, []])
    }

    myInput = uq.createInput(iOpts)
    uq.rng(100, 'twister')
    X = uq.getSample(myInput,50)
    assert X.shape == (50,3)

    print("Vine copula defined via dictionary...")
    iOpts['Copula'] = {
        'Type': 'CVine',
        'Families': ['t', 'Frank', 'Independence'],
        'Parameters': [[.4, 2], .5, []],
        'Structure': [2,1,3]
    }
    myInput2 = uq.createInput(iOpts)
    uq.rng(100, 'twister')
    X2 = uq.getSample(myInput2,50)
    assert X2.shape == (50,3)  

    assert np.allclose(X, X2, atol=5e-5)
    print("All good...")

    print("Truncation...")
    iOpts['Copula']['Truncation'] = 1
    myInput_Truncated = uq.createInput(iOpts)
    assert myInput_Truncated['Options']['Copula']['Truncation'] == 1
    print("All good...")

def test_PairCopulaParameterRange(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')   

    families = ['Gaussian', 'Clayton', 'Gumbel', 'Frank', 't'] # 'Independent', and 'Independence' families have no parameters
    for family in families:
        R = uq.PairCopulaParameterRange(family)
        assert R     

def test_sampleU(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    U = uq.sampleU(30,3)
    assert U.shape == (30,3)
    options = {'Method': 'LHS','LHSiterations': 2}
    U = uq.sampleU(30,3,options)
    assert U.shape == (30,3)
    options = {'Method': 'Sobol'}
    U = uq.sampleU(30,3,options)
    assert U.shape == (30,3)

def test_setDefaultSampling(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    Input = {"Marginals": uq.Marginals(2, "Gaussian", [0,1])}
    myInput = uq.createInput(Input)    
    success = uq.setDefaultSampling(myInput, 'LHS')
    assert success == 1
    assert uq.getInput(myInput['Name'])['Sampling']['DefaultMethod'] == 'LHS'

def test_subsample(request,helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    Input = {
        "Marginals": [uq.Marginals(1, "Gaussian", [2, .5])]+
                     [uq.Marginals(1, "Gaussian", [1, 1])]}
    myInput = uq.createInput(Input)
    X = uq.getSample(myInput, N = 100, Method = 'LHS')
    Xnew, idx = uq.subsample(X, NK=20, Method='kmeans', Name='Distance_nn', Value='euclidean', nargout=2)
    assert Xnew.shape == (20,2)
    assert idx.shape == (20,1)

def test_KernelMarginals(request,helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    X = np.random.rand(10,3)
    Input = {"Marginals": uq.KernelMarginals(X)}
    myInput = uq.createInput(Input)
    X = uq.getSample(myInput, N=20)
    assert X.shape == (20,3)

def test_test_block_independence(request,helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    InputOpts = {'Marginals': [{'Type':'Gaussian', 'Parameters': [0,1]} for i in np.arange(8)]}
    InputOpts['Copula'] = [
        uq.VineCopula(
            'CVine', 
            [1,2,3],
            ['Clayton', 'Gumbel', 'Gaussian'], 
            [1.4, 2.0, 0.3], 
            [0, 0, 0] ),
        uq.GaussianCopula([[1, -.5],[ -.5, 1]]),
        uq.PairCopula('t', [.5, 2], 0)
    ]
    InputOpts['Copula'][0]['Variables'] = [[1, 4, 6]]
    InputOpts['Copula'][1]['Variables'] = [[3, 7]]
    InputOpts['Copula'][2]['Variables'] = [[2, 8]]
    myInput = uq.createInput(InputOpts)
    uq.print(myInput)
    X = uq.getSample(myInput,1000,'Sobol')

    (BlocksHat, PVs, History, Message) = uq.test_block_independence(X.tolist(), 0.05)
    print(Message)
    assert BlocksHat == [5, [3, 7], [2, 8], [1, 4, 6]]
    print("All good...")

def test_listInputs(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    InputOpts = {
        'Marginals': {
            'Type': 'Uniform',
            'Parameters': [0, 15],
        }
    }    
    myInput = uq.createInput(InputOpts)

    InputOpts2 = {'Marginals': uq.Marginals(M=3, Type='Uniform', Parameters=[-np.pi, np.pi])}
    myInput2 = uq.createInput(InputOpts2)

    print("Check if listInputs() works, redirecting stdout and checking the output...")

    # Redirect stdout to a StringIO object
    captured_output = StringIO()
    sys.stdout = captured_output

    # Now anything printed will be captured
    uq.listInputs()

    # Access the captured output
    captured_output.seek(0)  # Move cursor to the start of StringIO object
    captured_text = captured_output.read()

    # Restore sys.stdout to its original value
    sys.stdout = sys.__stdout__

    assert all(input['Name'] in captured_text for input in [myInput, myInput2])
    print("Looks fine...")
    mySession.quit()
    time.sleep(1)    




