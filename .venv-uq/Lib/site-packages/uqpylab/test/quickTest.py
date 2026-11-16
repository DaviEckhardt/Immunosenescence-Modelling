import pytest
import numpy as np

# def test_analysis(request,helpers):
#     mySession = helpers.init_session(request)
#     uq = mySession.cli
#     uq.rng(100, 'twister')
#     mySession.timeout = 400 
#     # Prepare the experimental design
#     # & validation set from a simple model
#     InputOpts = {
#         'Marginals': [
#             {
#             'Name':'X1',
#             'Type':'Gaussian',
#             'Parameters' : [0.25, 1]
#             },
#             {
#             'Name':'X2',
#             'Type':'Gaussian',
#             'Parameters' : [0.25, 1]
#             }]
#         }
#     print("Creating an input...")
#     uq.createInput(InputOpts)
#     print("Done.")
#     print("Creating a true model object...")
#     ModelOpts = {'Type' : 'Model',
#         'ModelFun': 'uqpylab.test.true_models.hat2d'}
#     #MCSOpts.Type = 'Reliability'
#     myModel = uq.createModel(ModelOpts)
#     print("Done.")
#     # do an AK-MCS analysis
#     AKMCSOpts = {
#         'Type': 'Reliability',
#         'Method' : 'AKMCS',
#         'Simulation': {
#             'MaxSampleSize': 1e4},
#         'AKMCS': {
#             'MaxAddedED' : 20,
#             'IExpDesign':{
#                 'N': 100,
#                 'Sampling': 'LHS',
#             },
#             'Kriging':{
#                 'Corr':{
#                     'Family': 'Gaussian'
#                 }
#             },
#         'Convergence': 'stopPf',
#         'LearningFunction': 'EFF'
#         },
#     }
#     print("Starting an AKMCS analysis...")
#     myAKMCSAnalysis = uq.createAnalysis(AKMCSOpts)
#     print("Done.")
#     assert myAKMCSAnalysis['Results']['Pf'] < 0.001
#     print("Done.")

def R_S_input_and_model(uq):
    print("Creating a true model object...")
    ModelOpts = {'Type': 'Model', 'mString': 'X(:,1) - X(:,2)', 'isVectorized': 1}
    myModel = uq.createModel(ModelOpts)
    print("Done.")

    print("Creating the input...")
    InputOpts = {
        "Marginals": [
            {"Name": "R", "Type": "Gaussian", "Moments": [5.0 , 0.8]},
            {"Name": "S", "Type": "Gaussian", "Moments": [2.0 , 0.6]}
        ]
    }
    myInput = uq.createInput(InputOpts)
    print("Done.")
    return myModel, myInput

# def R_S_run_Analysis_print_display(uq, myOpts):
#     Analysis = uq.createAnalysis(myOpts)
#     print("Done.")
#     print(f"Checking print functionality...")       
#     uq.print(Analysis)
#     print("Done.")    
#     print(f"Checking display functionality...")
#     uq.display(Analysis,test_mode=True);
#     print("Done.")    
#     return Analysis

def test_sse_reliability(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')
    print("Creating an input...")
    InputOpts = {
            "Marginals": [
                {"Name": "R",               # Resistance
                "Type": "Gaussian",
                "Moments": [5.0 , 0.8]
                },
                {"Name": "S",               # Stress
                "Type": "Gaussian",
                "Moments": [2.0 , 0.6]
                }
            ]
        }
    myInput = uq.createInput(InputOpts)
    print("Done.")
    print("Creating a model...")
    ModelOpts = { 
        'Type': 'Model', 
        'mString': 'X(:,1) - X(:,2)',
        'isVectorized': 1
    }
    myModel = uq.createModel(ModelOpts)
    print("Done.")
    SSEROpts = {
    "Type": "Reliability",
    "Method": "SSER"
    }
    print("Performing SSE Reliability...")
    SSERAnalysis = uq.createAnalysis(SSEROpts)
    print("Done.")
    print("Testing print...")
    uq.print(SSERAnalysis)
    print("Testing display...")
    uq.display(SSERAnalysis,test_mode=True)
    print("All good.")


# def test_MCS_Reliability(request, helpers):
#     mySession = helpers.init_session(request)
#     uq = mySession.cli
#     uq.rng(100, 'twister')   

#     R_S_input_and_model(uq)
#     print("Running Monte Carlo simulation...")    
#     MCSOpts = {
#         "Type": "Reliability",
#         "Method":"MCS",
#         "Simulation": {"MaxSampleSize": 1e4, "BatchSize": 1e3, "TargetCoV": 5e-2}
#     }
#     MCSAnalysis = R_S_run_Analysis_print_display(uq, MCSOpts)
#     assert abs(MCSAnalysis['Results']['Pf'] - 0.001349898031630) < 0.001

# def test_FORM_Reliability(request, helpers):
#     mySession = helpers.init_session(request)
#     uq = mySession.cli
#     uq.rng(100, 'twister')   

#     R_S_input_and_model(uq)
#     print("Running FORM...") 
#     FORMOpts = {
#         "Type": "Reliability",
#         "Method":"FORM"
#     }
#     FORMAnalysis=R_S_run_Analysis_print_display(uq, FORMOpts)
#     assert abs(FORMAnalysis['Results']['Pf'] - 0.001349898031630) < 0.001

# def test_IS_Reliability(request, helpers):
#     mySession = helpers.init_session(request)
#     uq = mySession.cli
#     uq.rng(100, 'twister')   

#     R_S_input_and_model(uq)
#     print("Running Inportance sampling...")    
#     ISOpts = {
#         "Type": "Reliability",
#         "Method":"IS",
#         "Simulation": {"MaxSampleSize": 1e4, "BatchSize": 1e3, "TargetCoV": 5e-2}
#     }
#     ISAnalysis=R_S_run_Analysis_print_display(uq, ISOpts)
#     assert abs(ISAnalysis['Results']['Pf'] - 0.001349898031630) < 0.001

# def test_Subset_Reliability(request, helpers):
#     mySession = helpers.init_session(request)
#     uq = mySession.cli
#     uq.rng(100, 'twister')   

#     R_S_input_and_model(uq)
#     print("Running Subset simulation...")    
#     SubsetSimOpts = {
#         "Type": "Reliability",
#         "Method":"Subset",
#         "Simulation": {"MaxSampleSize": 1e4, "BatchSize": 1e3, "TargetCoV": 5e-2}
#     }
#     SubsetSimAnalysis=R_S_run_Analysis_print_display(uq, SubsetSimOpts)
#     assert abs(SubsetSimAnalysis['Results']['Pf'] - 0.001349898031630) < 0.001

# def test_APCKMCS_Reliability(request, helpers):
#     mySession = helpers.init_session(request)
#     uq = mySession.cli
#     uq.rng(100, 'twister')   

#     R_S_input_and_model(uq)
#     print("Running Adaptive-Polynomial-Chaos-Kriging-Monte-Carlo-Simulation...")    
#     APCKOpts = {
#         "Type": "Reliability",
#         "Method": "AKMCS",
#         "Simulation": {"MaxSampleSize": 1.0E+6},
#         "AKMCS": {
#             "MaxAddedED": 20,
#             "MetaModel": "PCK",
#             "PCK": {
#                 "Kriging": {
#                     "Corr": {
#                         "Family": "Gaussian"
#                     }
#                 }
#             },
#             "IExpDesign": {
#                 "N": 5,
#                 "Sampling": "LHS"
#             },
#             "Convergenge": "stopPf",
#             "LearningFunction": "EFF"
#         },

#     }
#     APCKAnalysis=R_S_run_Analysis_print_display(uq, APCKOpts)
#     assert abs(APCKAnalysis['Results']['Pf'] - 0.001349898031630) < 0.001

# def test_selectModel(request, helpers):
#     mySession = helpers.init_session(request)
#     uq = mySession.cli
#     uq.rng(2, 'twister')

#     ModelFullOpts = { 
#         'Type': 'Model', 
#         'mString': 'max(X(:,1) - X(:,2), 1.2*X(:,1) - 0.9*X(:,2))',
#         'isVectorized': 1
#     }
#     myLimitStateFull = uq.createModel(ModelFullOpts)

#     Model1Opts = { 
#         'Type': 'Model', 
#         'mString': '1.2*X(:,1) - 0.9*X(:,2)',
#         'isVectorized': 1
#     }
#     myLimitState1 = uq.createModel(Model1Opts)

#     InputOpts = {
#         "Marginals": [
#             {"Name": "R", "Type": "Gaussian", "Moments": [3.0 , 0.3]},
#             {"Name": "S", "Type": "Gaussian", "Moments": [2.0 , 0.4]}
#         ]
#     }
#     myInput = uq.createInput(InputOpts)

#     MCSOpts = {
#         "Type": "Reliability",
#         "Method":"MCS",
#         "Simulation": {"MaxSampleSize": 2e6}
#     }

#     uq.selectModel(myLimitStateFull['Name'])
#     myMCSAnalysis = uq.createAnalysis(MCSOpts)

#     assert np.abs(myMCSAnalysis['Results']['Pf'] - 0.000203) < 1e-4
    
# def test_ALR(request, helpers):
#     mySession = helpers.init_session(request)
#     uq = mySession.cli
#     uq.rng(100, 'twister')

#     print("Running Active Learning...")    
#     ModelOpts = { 
#         'Type': 'Model', 
#         'mString': '20 - (X(:,1)-X(:,2)).^2 - 8*(X(:,1)+X(:,2)-4).^3',
#         'isVectorized': 1
#     }
#     myModel = uq.createModel(ModelOpts)
#     InputOpts = {
#         "Marginals": [
#             {"Name": "X1", "Type": "Gaussian", "Parameters": [0.25, 1]},
#             {"Name": "X2", "Type": "Gaussian", "Parameters": [0.25, 1]}
#         ]
#     } 
#     myInput = uq.createInput(InputOpts)
#     ALROptions = {
#         "Type": "Reliability",
#         "Method": "ALR",
#         "Simulation": {
#             "BatchSize": 1e4,
#             "MaxSampleSize": 1e6
#         },
#         "ALR": {
#             "NumOfPoints": 2,
#             "Convergence": ["StopBetaBound", "StopBetaStab"],
#             "ConvThres": [[0.01, 0.05]],
#             "MaxAddedED": 20,
#             "Metamodel": "Kriging",
#             "Kriging": {
#                 "Corr": {
#                     "Family": "Gaussian",
#                     "Nugget": 1e-10
#                 },
#                 "EstimMethod": "ML",
#                 "Optim": {
#                     "Method": "HGA"
#                 }
#             }
#         },
#         "Subset": {
#             "p0": 0.2
#         }
#     }
#     myALRAnalysis=R_S_run_Analysis_print_display(uq, ALROptions)
#     assert abs(myALRAnalysis['Results']['Pf'] - 0.00039888000000000006) < 1e-5

# def test_AsynchronousALR(request, helpers):
#     mySession = helpers.init_session(request)
#     uq = mySession.cli
#     uq.rng(100, 'twister')

#     InputOpts = {"Marginals": uq.StdNormalMarginals(2)}
#     myInput = uq.createInput(InputOpts)

#     ALROptions = {
#         "Type": "Reliability",
#         "Method": "ALR",
#         "ALR": {
#             "IExpDesign": {"N": 4}
#         },
#         "Async": {
#             "Enable": True,
#             "InitED": True
#         }
#     }    
#     myALRAnalysis = uq.createAnalysis(ALROptions)

#     Xnext = np.array(myALRAnalysis["Results"]["NextSample"])
#     Ynext = 2.5 - (Xnext[:,0]+ Xnext[:,1])/np.sqrt(2) + 0.1 * (Xnext[:,0] - Xnext[:,1])**2 
#     myALRAnalysis = uq.resumeAnalysis(Ynext.tolist())

#     Xnext = np.array(myALRAnalysis["Results"]["NextSample"], ndmin=2)
#     Ynext = 2.5 - (Xnext[:,0]+ Xnext[:,1])/np.sqrt(2) + 0.1 * (Xnext[:,0] - Xnext[:,1])**2 
#     myALRAnalysis = uq.resumeAnalysis(Ynext.tolist())

#     uq.print(myALRAnalysis)
#     uq.display(myALRAnalysis, test_mode=True);
        
# def test_sensitivity(request, helpers):
#     mySession = helpers.init_session(request)
#     uq = mySession.cli
#     uq.rng(100, 'twister')

#     ModelOpts = {
#         'Type': 'Model', 
#         'ModelFun':'uqpylab.test.true_models.borehole'
#     }
#     myModel = uq.createModel(ModelOpts)

#     InputOpts = {
#         'Marginals': [
#             {'Name': 'rw', 'Type': 'Gaussian',  'Parameters': [0.10, 0.0161812]},  # Radius of the borehole (m)
#             {'Name': 'r',  'Type': 'Lognormal', 'Parameters': [7.71, 1.0056]},     # Radius of influence (m)
#             {'Name': 'Tu', 'Type': 'Uniform',   'Parameters': [63070, 115600]},    # Transmissivity, upper aquifer (m^2/yr)
#             {'Name': 'Hu', 'Type': 'Uniform',   'Parameters': [990, 1110]},        # Potentiometric head, upper aquifer (m)
#             {'Name': 'Tl', 'Type': 'Uniform',   'Parameters': [63.1, 116]},        # Transmissivity, lower aquifer (m^2/yr)
#             {'Name': 'Hl', 'Type': 'Uniform',   'Parameters': [700, 820]},         # Potentiometric head , lower aquifer (m)
#             {'Name': 'L',  'Type': 'Uniform',   'Parameters': [1120, 1680]},       # Length of the borehole (m)
#             {'Name': 'Kw', 'Type': 'Uniform',   'Parameters': [9855, 12045]},      # Borehole hydraulic conductivity (m/yr)
#         ]
#     }
#     myInput = uq.createInput(InputOpts)

#     print("Sensitivity: Correlation analysis...")
#     CorrSensOpts = {
#         "Type": "Sensitivity",
#         "Method": "Correlation",
#         "Correlation": {"SampleSize": 1e2}
#     }
#     CorrAnalysis = uq.createAnalysis(CorrSensOpts)
#     uq.print(CorrAnalysis)
#     uq.display(CorrAnalysis, test_mode=True);
#     print("Done...")

#     print("Sensitivity: Standard Regression Coefficients (SRC)...")
#     SRCSensOpts = {
#         "Type": "Sensitivity",
#         "Method": "SRC",
#         "SRC": {"SampleSize": 1e2}
#     }
#     SRCAnalysis = uq.createAnalysis(SRCSensOpts)
#     uq.print(SRCAnalysis)
#     uq.display(SRCAnalysis, test_mode=True);
#     print("Done...")

#     print("Sensitivity: Perturbation-based indices...")
#     PerturbationSensOpts = {
#         "Type": "Sensitivity",
#         "Method": "Perturbation"
#     }
#     PerturbationAnalysis = uq.createAnalysis(PerturbationSensOpts)
#     uq.print(PerturbationAnalysis)
#     uq.display(PerturbationAnalysis, test_mode=True);
#     print("Done...")

#     print("Sensitivity: Cotter sensitivity indices...")
#     CotterSensOpts = {
#         "Type": "Sensitivity",
#         "Method": "Cotter",
#         "Factors": {"Boundaries": 0.5}
#     }
#     CotterAnalysis = uq.createAnalysis(CotterSensOpts)
#     uq.print(CotterAnalysis)
#     uq.display(CotterAnalysis,test_mode=True);
#     print("Done...")

#     print("Sensitivity: Morris' elementary effects...")
#     MorrisSensOpts = {
#         "Type": "Sensitivity",
#         "Method": "Morris",
#         "Factors": {"Boundaries": 0.5},
#         "Morris": {"Cost": 1e2}
#     }
#     MorrisAnalysis = uq.createAnalysis(MorrisSensOpts)
#     uq.print(MorrisAnalysis)
#     uq.display(MorrisAnalysis,test_mode=True);
#     print("Done...")

#     print("Sensitivity: Sobol' indices...")
#     SobolOpts = {
#         "Type": "Sensitivity",
#         "Method": "Sobol",
#         "Sobol": {
#             "Order": 1,
#             "SampleSize": 1e2
#         }
#     }
#     SobolAnalysis = uq.createAnalysis(SobolOpts)
#     uq.print(SobolAnalysis)
#     uq.display(SobolAnalysis,test_mode=True);
#     print("Done...")

#     print("Sensitivity: PCE-based Sobol' indices...")
#     PCEOpts = {
#         'Type': 'Metamodel',
#         'MetaType': 'PCE',
#         'Method': 'LARS',
#         'FullModel': myModel['Name'],
#         'Degree': 5,
#         'ExpDesign': {'NSamples': 200},
#     }
#     myPCE = uq.createModel(PCEOpts)
#     mySobolAnalysisPCE = uq.createAnalysis(SobolOpts)
#     print("Done...")

#     print("Sensitivity: PCE-based Sobol' indices with PCE truncation scheme...")
#     PCEOpts = {
#         'Type': 'Metamodel',
#         'MetaType': 'PCE',
#         'FullModel': myModel['Name'],
#         'Degree': np.arange(1,5).tolist(),
#         'TruncOptions': {
#             'qNorm': 0.7,
#             'MaxInteraction': 2
#         },
#         'ExpDesign': {
#             'NSamples': 100,
#             'Sampling': 'LHS'
#         }
#     }
#     myPCE = uq.createModel(PCEOpts)
#     SobolOpts = {
#         'Type': 'Sensitivity',
#         'Method': 'Sobol',
#         'Sobol': {
#             'Order': 2
#         }
#     }
#     mySobolAnalysisPCE = uq.createAnalysis(SobolOpts)
#     print("Done...")

#     print("Sensitivity: LRA-based Sobol' indices...")
#     LRAOpts = {
#         'Type': 'Metamodel',
#         'MetaType': 'LRA',
#         'FullModel': myModel['Name'],
#         'Rank': np.arange(1,20).tolist(),
#         'Degree': np.arange(1,20).tolist(),
#         'ExpDesign': {'NSamples': 200}
#     }
#     mySobolAnalysisLRA = uq.createAnalysis(SobolOpts)
#     print("Done...")

#     print("Sensitivity: Borgonovo indices...")
#     BorgonovoOpts = {
#         "Type": "Sensitivity",
#         "Method": "Borgonovo",
#         "Borgonovo": {
#             "SampleSize": 1e2,
#             "NClasses": 20    
#         }
#     }
#     BorgonovoAnalysis = uq.createAnalysis(BorgonovoOpts)
#     uq.print(BorgonovoAnalysis)
#     uq.display(BorgonovoAnalysis,test_mode=True);
#     uq.display(BorgonovoAnalysis, outidx=1, Joint_PDF=True, inidx=1, test_mode=True);
#     print("Done...")

# def test_sensitivity_multiple_outputs_model(request, helpers):
#     mySession = helpers.init_session(request)
#     uq = mySession.cli
#     uq.rng(50, 'twister')

#     ModelOpts = {
#         'Type': 'Model',
#         'ModelFun': 'uqpylab.test.true_models.simply_supported_beam_9points',
#         'isVectorized': 'true'
#     }
#     myModel = uq.createModel(ModelOpts)
#     InputOpts = {'Marginals': [
#             {'Name': 'b', 'Type': 'Lognormal', 'Moments': [0.15, 0.0075]},   # beam width (m)
#             {'Name': 'h', 'Type': 'Lognormal', 'Moments': [0.3, 0.015]},     # beam height (m)
#             {'Name': 'L', 'Type': 'Lognormal', 'Moments': [5, 0.05]},        # beam length (m)
#             {'Name': 'E', 'Type': 'Lognormal', 'Moments': [3e10, 4.5e9]},    # Young's modulus (Pa)
#             {'Name': 'p', 'Type': 'Lognormal', 'Moments': [1e4, 2e3]}        # uniform load (N/m)
#             ]}
#     myInput = uq.createInput(InputOpts)
#     SobolOpts = {
#         "Type": "Sensitivity",
#         "Method": "Sobol",
#         "Sobol": {
#             "SampleSize": 1e2
#         }
#     }
#     print("Sensitivity: multiple outputs model...")
#     mySobolAnalysis = uq.createAnalysis(SobolOpts)
#     print("Done...")

# def test_sensitivity_dependent_inputs(request, helpers):
#     mySession = helpers.init_session(request)
#     uq = mySession.cli
#     uq.rng(101, 'twister')

#     ModelOpts = {
#         'Type': 'Model',
#         'ModelFun': 'uqpylab.test.true_models.shortcol',
#     }
#     myModel = uq.createModel(ModelOpts)
#     InputOpts = {'Marginals': [
#             {'Name': 'Y', 'Type': 'Lognormal', 'Moments': [5, 0.5]}, # yield stress (MPa)
#             {'Name': 'M', 'Type': 'Gaussian','Moments': [2000, 400]}, # bending moment (N.mm)
#             {'Name': 'P', 'Type': 'Gaussian','Moments': [500, 100]} # axial force (N)
#     ]}
#     InputOpts['Copula'] = {
#         'Type': 'Gaussian',
#         'RankCorr': [[1, 0, 0], [0, 1, 0.72], [0, 0.72, 1]]
#     }
#     myInput = uq.createInput(InputOpts)

#     print("Sensitivity: Kucherenko indices...")
#     KucherenkoOpts = {
#         'Type': 'Sensitivity',
#         'Method': 'Kucherenko',
#         'Kucherenko': {'SampleSize': 5e2}
#     }
#     myKucherenkoAnalysis = uq.createAnalysis(KucherenkoOpts)
#     uq.print(myKucherenkoAnalysis)
#     uq.display(myKucherenkoAnalysis, test_mode=True);
#     print("Done...")

#     print("Sensitivity: ANCOVA...")
#     ANCOVAOpts = {
#         'Type': 'Sensitivity',
#         'Method': 'ANCOVA',
#         'Ancova': {'SampleSize': 150}
#     }
#     myANCOVAAnalysis = uq.createAnalysis(ANCOVAOpts)
#     uq.print(myANCOVAAnalysis)
#     uq.display(myANCOVAAnalysis, test_mode=True);
#     print("Done...")

