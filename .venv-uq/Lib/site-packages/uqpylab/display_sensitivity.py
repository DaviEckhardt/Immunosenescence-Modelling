""" Plotting utility imitating the uq_display_uq_sensitivity functionality of UQLab """
from uqpylab import sessions
import uqpylab.display_util as display_util
import uqpylab.display_general as display_general
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

## Figure template
pio.templates.default = 'simple_white'

## Template modifications
# Size
pio.templates[pio.templates.default].layout.height = 600
pio.templates[pio.templates.default].layout.width = 600

# Grid
pio.templates[pio.templates.default].layout.xaxis.showgrid = True
pio.templates[pio.templates.default].layout.yaxis.showgrid = True
pio.templates[pio.templates.default].layout.autosize = False

# Font
pio.templates[pio.templates.default].layout.font.size = 16

# Markers
pio.templates[pio.templates.default].data.histogram[0].marker.line.width = 0

def sensitivity(obj, outidx=None, cobweb=None, Joint_PDF=None, inidx=None, **kwargs):
    # check for input arguments
    if outidx is None:
        outidx = [1]
    if type(outidx)==int:
        outidx = [outidx]
    outidx = [i-1 for i in outidx]

    method = obj['Options']['Method'].lower()

    # check the residual command line for cobweb
    cobweb_flag = cobweb

    if cobweb_flag:
        print("Not yet implemented!")
    else:
        if method == 'correlation':
            return sensitivity_correlation(obj, outidx)
        elif method == "perturbation":
            return sensitivity_perturbation(obj, outidx)
        elif method == "cotter":
            return sensitivity_cotter(obj, outidx)
        elif method == "src":
            return sensitivity_src(obj, outidx)
        elif method == "morris":
            return sensitivity_morris(obj, outidx)
        elif method == "sobol":
            return sensitivity_sobol(obj, outidx, **kwargs)
        elif method == "borgonovo":
            if Joint_PDF is None:
                Joint_PDF = False
            if inidx is None:
                inidx = 'all'
            return sensitivity_borgonovo(obj, outidx, Joint_PDF, inidx)
        elif method == "ancova":
            return sensitivity_ancova(obj, outidx)
        elif method == "kucherenko":
            return sensitivity_kucherenko(obj, outidx)
        else:
            print("Select from 'correlation', 'perturbation', 'cotter', 'src', 'morris', 'sobol', 'borgonovo', 'ancova', or 'kucherenko' method!")


def sensitivity_correlation(obj, outidx):
    # Collect the relevant information 
    if type(obj['Results']) == dict: 
        CorrIDX = np.array(obj['Results']['CorrIndices'])
        RankCorrIDX = np.array(obj['Results']['RankCorrIndices'])
        if 'VariableNames' in obj['Results']:
            VarNames = obj['Results']['VariableNames'] 
        else:
            VarNames = [f'X{i}' for i in len(CorrIDX)] 
    elif type(obj['Results']) == list: 
        CorrIDX = np.array(obj['Results'][-1]['CorrIndices'])
        RankCorrIDX = np.array(obj['Results'][-1]['RankCorrIndices'])
        if 'VariableNames' in obj['Results'][-1]:
            VarNames = obj['Results'][-1]['VariableNames']
        else:
            VarNames = [f'X{i}' for i in len(CorrIDX)] 
    else:
        msg = "obj['Results'] must be either a list or a dictionary"
        raise(msg)
    
    if CorrIDX.ndim < 2: # single output case
        CorrIDX = np.expand_dims(CorrIDX,axis=1)
        RankCorrIDX = np.expand_dims(RankCorrIDX,axis=1)

    if len(outidx) > 1:
        fig = []

    for oo in outidx:
        # Plot the sensitivity
        data = {'linear': CorrIDX[:,oo], 'rank': RankCorrIDX[:,oo]}
        if len(outidx) > 1:
            title= f'Correlation indices, output #{oo+1}' 
        else:
            title=f'Correlation-based indices'           
        fig_ = display_general.display_bar(data, VarNames=VarNames, xaxis_title='Input Variable', 
                                           yaxis_title='ρ', title=title)

        if len(outidx) == 1:
            fig = fig_    
        else:
            fig.append(fig_)  

    return fig


def sensitivity_perturbation(obj, outidx):
    # Collect the relevant information
    if type(obj['Results']) == dict: 
        Sensitivity = np.array(obj['Results']['Sensitivity'])
        VarNames = obj['Results']['VariableNames'] 
    elif type(obj['Results']) == list: 
        Sensitivity = np.array(obj['Results'][-1]['Sensitivity'])
        VarNames = obj['Results'][-1]['VariableNames'] 
    else:
        msg = "obj['Results'] must be either a list or a dictionary"
        raise(msg)
        
    if Sensitivity.ndim < 2: # single output case
        Sensitivity = np.expand_dims(Sensitivity,axis=1)

    if len(outidx) > 1:
        fig = []

    for oo in outidx:
    # Plot the sensitivity
        data = {'Sensitivity': Sensitivity[:,oo]}
        if len(outidx) > 1:
            title= f'Perturbation-based indices, output #{oo+1}' 
        else:
            title= f'Perturbation-based indices'         
        fig_ = display_general.display_bar(data, VarNames=VarNames, xaxis_title='Input Variable', 
                                           yaxis_title='Sensitivity', yaxis_ticks=[0,1,0.1], 
                                           title=title)          

        if len(outidx) == 1:
            fig = fig_
        else:
            fig.append(fig_)

    return fig  


def sensitivity_cotter(obj, outidx):
    if type(obj['Results']) == dict:
        CotterIndices = np.array(obj['Results']['CotterIndices'])
        VarNames = obj['Results']['VariableNames']
    elif type(obj['Results']) == list: 
        CotterIndices = np.array(obj['Results'][-1]['CotterIndices'])
        VarNames = obj['Results'][-1]['VariableNames']
    else:
        msg = "obj['Results'] must be either a list or a dictionary"
        raise(msg) 

    if CotterIndices.ndim < 2: # single output case
        CotterIndices = np.expand_dims(CotterIndices, axis=1)         

    if len(outidx) > 1:
        fig = []

    for oo in outidx:
        # Plot the sensitivity    
        data = {'CotterIndices': CotterIndices[:,oo]}
        if len(outidx) > 1:
            title= f'Cotter indices, output #{oo+1}'
        else:
            title= f'Cotter indices'          
        fig_ = display_general.display_bar(data, VarNames=VarNames, xaxis_title='Input Variable', 
                                           yaxis_title='Cotter Index',title=title)

         
        
        if len(outidx) == 1:
            fig = fig_    
        else:
            fig.append(fig_)  

    return fig


def sensitivity_src(obj, outidx):
    if type(obj['Results']) == dict:
        SRCIDX = np.array(obj['Results']['SRCIndices'])
        SRRCIDX = np.array(obj['Results']['SRRCIndices'])
        VarNames = obj['Results']['VariableNames']
    elif type(obj['Results']) == list: 
        SRCIDX = np.array(obj['Results'][-1]['SRCIndices'])
        SRRCIDX = np.array(obj['Results'][-1]['SRRCIndices'])
        VarNames = obj['Results'][-1]['VariableNames']
    else:
        msg = "obj['Results'] must be either a list or a dictionary"
        raise(msg)

    if SRCIDX.ndim < 2: # single output case
        SRCIDX = np.expand_dims(SRCIDX, axis=1)
        SRRCIDX = np.expand_dims(SRRCIDX, axis=1)

    if len(outidx) > 1:
        fig = []

    for oo in outidx:
    # Plot the sensitivity    
        data = {'SRC': SRCIDX[:,oo], 'SRRC': SRRCIDX[:,oo]}
        if len(outidx) > 1:
            title= f'SRC results, output #{oo+1}'
        else:
            title= f'SRC results'         
        fig_ = display_general.display_bar(data, VarNames=VarNames, xaxis_title='Input Variable', 
                                    yaxis_title='Sensitivity', showlegend=True, title=title)              
         
        if len(outidx) == 1:
            fig = fig_    
        else:
            fig.append(fig_)  

    return fig     


def sensitivity_morris(obj, outidx):
    # Collect the relevant information
    if type(obj['Results']) == dict:
        MU = np.array(obj['Results']['Mu'])
        MUStar = np.array(obj['Results']['MuStar'])
        MSTD = np.array(obj['Results']['Std'])
        VarNames = obj['Results']['VariableNames']
    elif type(obj['Results']) == list: 
        MU = np.array(obj['Results'][-1]['Mu'])
        MUStar = np.array(obj['Results'][-1]['MuStar'])
        MSTD = np.array(obj['Results'][-1]['Std'])
        VarNames = obj['Results'][-1]['VariableNames']
    else:
        msg = "obj['Results'] must be either a list or a dictionary"
        raise(msg)    

    if MU.ndim < 2:
        MU = np.expand_dims(MU, axis=1)
        MUStar = np.expand_dims(MUStar, axis=1)
        MSTD = np.expand_dims(MSTD, axis=1)    

    fig = []

    for oo in outidx:

        ## Plot MU
        minMU = np.min(MU[:,oo])
        maxMU = np.max(MU[:,oo])
        minMU = np.amin([minMU - 0.15*np.abs(maxMU), -0.15*np.abs(maxMU)])
        maxMU = np.amax([maxMU + 0.15*np.abs(maxMU),  0.15*np.abs(maxMU)])

        maxMSTD = np.max(MSTD[:,oo])
        minMSTD = -0.15*np.abs(maxMSTD)
        maxMSTD = maxMSTD + 0.15*np.abs(maxMSTD) 

        data = {'minMU': minMU, 'maxMU': maxMU, 'minMSTD': minMSTD, 'maxMSTD': maxMSTD,
                'MU': MU[:,oo], 'MSTD': MSTD[:,oo]}  

        fig_ = display_general.morris_plot(data, VarNames=VarNames, xaxis_title='μ', yaxis_title='σ')
        if len(outidx) > 1:
            fig_.update_layout(title= f'Elementary effects, output #{oo+1}') 
        else:
            fig_.update_layout(title= f'Elementary effects')                
        
        fig.append(fig_)  

        ## Plot MU*
        minMUStar = np.min(MUStar[:,oo])
        maxMUStar = np.max(MUStar[:,oo])
        minMUStar = np.amin([minMUStar - 0.15*np.abs(maxMUStar), -0.15*np.abs(maxMUStar)])
        maxMUStar = np.amax([maxMUStar + 0.15*np.abs(maxMUStar),  0.15*np.abs(maxMUStar)])

        maxMSTD = np.max(MSTD[:,oo])
        minMSTD = -0.15*np.abs(maxMSTD)
        maxMSTD = maxMSTD + 0.15*np.abs(maxMSTD)   

        data = {'minMU': minMUStar, 'maxMU': maxMUStar, 'minMSTD': minMSTD, 'maxMSTD': maxMSTD,
                'MU': MUStar[:,oo], 'MSTD': MSTD[:,oo]}  

        fig_ = display_general.morris_plot(data, VarNames=VarNames, xaxis_title='μ<sup>*', yaxis_title='σ')
        if len(outidx) > 1:
            fig_.update_layout(title= f'Elementary effects, output #{oo+1}') 
        else:
            fig_.update_layout(title= f'Elementary effects')                
        
        fig.append(fig_)  

    return fig
           
def sensitivity_borgonovo(obj, outidx, Joint_PDF=False, inidx='all'):
    # Collect the relevant information 
    if type(obj['Results']) == dict:
        Delta = np.array(obj['Results']['Delta'])
        if 'nonCosts' in obj['Internal']['Input']: 
            VarNames = [obj['Results']['VariableNames'][i-1] for i in obj['Internal']['Input']['nonCosts']]
        else:
            VarNames = obj['Results']['VariableNames']
        if Joint_PDF:
            JointPDF = obj['Results']['JointPDF']
    elif type(obj['Results']) == list: 
        Delta = np.array(obj['Results'][-1]['Delta'])
        if 'nonCosts' in obj['Internal']['Input']: 
            VarNames = [obj['Results'][-1]['VariableNames'][i-1] for i in obj['Internal']['Input']['nonCosts']]
        else:
            VarNames = obj['Results'][-1]['VariableNames']       
        if Joint_PDF: 
            JointPDF = obj['Results'][-1]['JointPDF']
    else:
        msg = "obj['Results'] must be either a list or a dictionary"
        raise(msg)

    if Delta.ndim < 2: # single output case
        Delta = np.expand_dims(Delta,axis=1)
    
    if len(outidx) > 1:
        fig = []

    if not Joint_PDF:
        for oo in outidx:
            # Plot the sensitivity 
            data = {'Delta': Delta[:,oo]}
            if len(outidx) > 1:
                title= f'Borgonovo indices, output #{oo+1}'
            else:
                title= f'Borgonovo indices'           

            fig_ = display_general.display_bar(data, VarNames=VarNames, xaxis_title='Input Variable', 
            yaxis_title='𝛿<sub>i', yaxis_ticks=[0, 1, .2], title=title)              
            
            if len(outidx) == 1:
                fig = fig_    
            else:
                fig.append(fig_)
    else: 
        for oo in outidx:
            if type(inidx)==list:
                inidx = np.array(inidx)
                if np.max(inidx) > len(Delta[:,oo]):
                    msg = IndexError('The requested variable index does not exist in the model!')
                    raise(msg)
                inidx = [i-1 for i in inidx]
            if type(inidx)==int:
                if inidx > len(Delta[:,oo]):
                    msg = IndexError('The requested variable index does not exist in the model!')
                    raise(msg)                    
                inidx = [inidx-1]
            if inidx=='all':
                inidx = list(range(len(Delta[:,oo])))

            if len(outidx) == 1 and len(inidx) > 1:  
                fig = []

            for ii in inidx:
                fig_ = px.imshow(np.array(JointPDF[ii]).T, origin='lower')

                fig_.update_layout(
                    xaxis_title= VarNames[ii],
                )   
                if len(outidx) > 1:
                    fig_.update_layout(
                        yaxis_title= f'Y<sub>{oo+1}',
                        title=f'Joint Pdf Y<sub>{oo+1}</sub>|{VarNames[ii]}'
                    )   
                else:
                    fig_.update_layout(
                        yaxis_title= 'Y',
                        title=f'Joint Pdf Y|{VarNames[ii]}'
                    ) 
                     
                if len(inidx) > 1:
                    fig.append(fig_)
                else:
                    fig = fig_    
    return fig

def sensitivity_sobol(obj, outidx, pie=None, hist=None, **kwargs):
    # Universal options for the Sobol Indices plots    
    NIndices = 10                       # max. number of Indices to plot (total and 1st order)
    NIndicesHigh = 5                    # max number of indices to plot (higher orders)
    NumOfColors = 6
    MyColors = display_util.colorOrder(NumOfColors)

    # initialization:
    pie_flag, hist_flag = pie, hist
    if pie_flag is None and hist_flag is None: # default is hist_flag
        pie_flag, hist_flag = False, True
        if 'Bootstrap' in obj['Results'] and obj['Results']['Bootstrap'] is not None:
            Bootstrap = True
        else:
            Bootstrap = False
    else:
        Bootstrap = False
    
    VarNames = obj['Results']['VariableNames']

    # Collect the relevant information
    if 'Total' in obj['Results']:
        Total = np.array(obj['Results']['Total'])
        if Total.ndim < 2:
            Total = np.expand_dims(Total, axis=1)

    fig = []

    for oo in outidx:
        ## --------------- TOTAL INDICES ----------------- ##
        # HISTOGRAM
        if hist_flag:
            if Bootstrap:
                # Collect the relevant information
                MEAN = np.array(obj['Results']['Bootstrap']['Total']['Mean'])
                CI = np.array(obj['Results']['Bootstrap']['Total']['CI'])
                if MEAN.ndim < 2:
                    MEAN = np.expand_dims(MEAN, axis=1)
                if CI.ndim < 3:
                    CI = np.expand_dims(CI, axis=1)
                lb = MEAN[:,oo] - CI[:,oo,0]
                ub = CI[:,oo,1] - MEAN[:,oo]

                # Plot a histogram in case the 'AllOrders' data is not present in ob['Results'], otherwise store the data for a grouped bar plot
                data = {'x_bar': [VarNames], 'y_bar': [Total[:,oo]], 'lb': [lb], 'ub': [ub], 'trace_name': ["Total Sobol' Indices"]}
                if 'AllOrders' not in obj['Results']:
                    fig_ = display_general.display_bar_errors(data, xaxis_title='Input Variable', 
                            yaxis_title='S<sub>i</sub><sup>Tot</sup>',  yaxis_ticks=[0, 1, .2])

            else:
                # Plot a histogram in case the 'AllOrders' data is not present in ob['Results'], otherwise store the data for a grouped bar plot
                data = {"Total Sobol' indices": Total[:,oo]}
                if 'AllOrders' not in obj['Results']:
                    fig_ = display_general.display_bar(data, VarNames=VarNames, xaxis_title='Input Variable', 
                                yaxis_title='S<sub>i</sub><sup>Tot</sup>', yaxis_ticks=[0, 1, .2])

            if 'AllOrders' not in obj['Results']:
                if len(outidx) > 1:
                    fig_.update_layout(title= dict(text=f"Total Sobol' indices, output #{oo+1}",y=0.95)) 
                else:
                    fig_.update_layout(title= dict(text=f"Total Sobol' indices",y=0.95))  
                fig.append(fig_)

        # PIE CHART
        if pie_flag:
            data = {'Values': Total[:,oo]}
            fig_ = display_general.pie_chart(data, VarNames)
            
            if len(outidx) > 1:
                fig_.update_layout(title= f"Total Sobol' indices (pie), output #{oo+1}") 
            else:
                fig_.update_layout(title= f"Total Sobol' indices (pie)")  
            fig.append(fig_)

        # ---------------  i-th ORDER INDICES: only print the first few --------------- #
        if 'AllOrders' in obj['Results']:
            coloridx = 1
            AllOrders = obj['Results']['AllOrders']
            if obj['Internal']['Sobol']['Order'] == 1:  
                AllOrders = [AllOrders]
            for ii in range(len(AllOrders)):
                # Collect the relevant information
                AllOrders_ii = np.array(AllOrders[ii])
                if AllOrders_ii.ndim < 2:
                    AllOrders_ii = np.expand_dims(AllOrders_ii, axis=1)
                coloridx = coloridx % len(MyColors)
                TickNames = VarNames
                if ii>0:
                    # Higher-order Indices: plot at most NIndicesHigh indices                   
                    NumOfIndices = min([len(AllOrders_ii[:,oo]),NIndicesHigh])
                    idx = np.argsort(-AllOrders_ii[:,oo])
                    CurIndices = AllOrders_ii[idx,oo]
                    idx = idx[:NumOfIndices]
                    CurIndices = CurIndices[:NumOfIndices]
                    CurrentVarIdx = np.array(obj['Results']['VarIdx'][ii])[idx,:] - 1
                    TickNames = [f'{VarNames[CurrentVarIdx[i,0]]}{VarNames[CurrentVarIdx[i,1]]}' for i in range(CurrentVarIdx.shape[0])]

                else:
                    # First-order Indices: plot all indices
                    NumOfIndices = len(AllOrders_ii[:,oo])
                    CurIndices = AllOrders_ii[:,oo]
                    idx = list(range(len(CurIndices)))

                # HISTOGRAM
                if hist_flag:
                    if Bootstrap:
                        # Collect the relevant information
                        if len(AllOrders)==1: # for order=1, obj['Results']['Bootstrap']['AllOrders'] is a dict
                            MEAN = np.array(obj['Results']['Bootstrap']['AllOrders']['Mean'])
                            CI = np.array(obj['Results']['Bootstrap']['AllOrders']['CI'])
                        else: # otherwise it's a list of dicts
                            MEAN = np.array(obj['Results']['Bootstrap']['AllOrders'][ii]['Mean'])
                            CI = np.array(obj['Results']['Bootstrap']['AllOrders'][ii]['CI'])

                        if MEAN.ndim < 2:
                            MEAN = np.expand_dims(MEAN, axis=1)
                        if CI.ndim < 3:
                            CI = np.expand_dims(CI, axis=1)
                        lb = MEAN[idx,oo] - CI[idx,oo,0]
                        ub = CI[idx,oo,1] - MEAN[idx,oo]

                        # append data for the First-order Indices to the data of the Total Indices for the grouped bar plot
                        if ii==0:
                            data['x_bar'].append(VarNames)
                            data['y_bar'].append(CurIndices) 
                            data['lb'].append(lb)
                            data['ub'].append(ub)
                            data['trace_name'].append(f"Sobol' indices Order {ii+1}")
                            fig_ = display_general.display_bar_errors(data, xaxis_title='Input Variable', 
                                    yaxis_title='Sensitivity',  yaxis_ticks=[0, 1], showlegend=True)

                        # create a new dict for Higher-order Indices and plot a bar plot
                        else:
                            data = {'x_bar': TickNames, 'y_bar': CurIndices, 'lb': lb, 'ub': ub} 
                            fig_ = display_general.display_bar_errors(data, xaxis_title='Input Variable', 
                                    yaxis_title=f'S<sub>u</sub><sup>({ii+1})</sup>')

                    else:
                        if ii==0:
                            data[f"Sobol' indices Order {ii+1}"] = CurIndices
                            # grouped bar plot for Total and First-order indices
                            fig_ = display_general.display_bar(data, VarNames=TickNames, xaxis_title='Input Variable', 
                                            yaxis_title=f'Sensitivity',  yaxis_ticks=[0, 1, .2], showlegend=True)  
                        else:
                            data = {f"Sobol' indices Order {ii+1}": CurIndices}
                                # single bar plot for Higher-order Indices
                            fig_ = display_general.display_bar(data, VarNames=TickNames, xaxis_title='Input Variable', 
                                            yaxis_title=f'S<sub>u</sub><sup>({ii+1})</sup>')                           

                    if len(outidx) > 1:
                        if ii==0: # grouped bar plot for Total and First-order indices
                            fig_.update_layout(title= dict(text=f"Sobol' indices, output #{oo+1}",y=.95)) 
                        else: # single bar plot for Higher-order Indices
                            fig_.update_layout(title= dict(text=f"Sobol' indices Order {ii+1}, output #{oo+1}",y=0.95)) 
                    else:
                        if ii==0: # grouped bar plot for Total and First-order indices
                            fig_.update_layout(title= dict(text=f"Sobol' indices",y=0.95))  
                        else: # single bar plot for Higher-order Indices
                            fig_.update_layout(title= dict(text=f"Sobol' indices Order {ii+1}",y=0.95))  
                    fig.append(fig_)
        
                # PIE CHART
        if pie_flag and ii == 0:
            data = {'Values': CurIndices/np.sum(CurIndices)}
            fig_ = display_general.pie_chart(data, VarNames)
            
            if len(outidx) > 1:
                fig_.update_layout(title= f"Sobol' indices Order {ii+1} (pie), output #{oo+1}") 
            else:
                fig_.update_layout(title= f"Sobol' indices Order {ii+1} (pie)")
            fig.append(fig_)

    return fig

def sensitivity_ancova(obj, outidx):
    # Universal options for the ANCOVA Indices Plots
    NIndices = 100
    # Collect the relevant information
    if type(obj['Results']) == dict: 
        Uncorrelated = np.array(obj['Results']['Uncorrelated'])
        Interactive = np.array(obj['Results']['Interactive'])
        Correlated = np.array(obj['Results']['Correlated'])
        FirstOrder = np.array(obj['Results']['FirstOrder'])
        VarNames = obj['Results']['VariableNames'] 
    elif type(obj['Results']) == list: 
        Uncorrelated = np.array(obj['Results'][-1]['Uncorrelated'])
        Interactive = np.array(obj['Results'][-1]['Interactive'])
        Correlated = np.array(obj['Results'][-1]['Correlated'])
        FirstOrder = np.array(obj['Results'][-1]['FirstOrder'])
        VarNames = obj['Results'][-1]['VariableNames'] 
    else:
        msg = "obj['Results'] must be either a list or a dictionary"
        raise(msg)

    if Uncorrelated.ndim < 2:
        Uncorrelated = np.expand_dims(Uncorrelated, axis=1)
        Interactive = np.expand_dims(Interactive, axis=1)
        Correlated = np.expand_dims(Correlated, axis=1)   
        FirstOrder = np.expand_dims(FirstOrder, axis=1)   

    fig = []

    myColorOrder = display_util.colorOrder(4)

    for oo in outidx:
        # Get the number of indices
        NumOfIndices = len(FirstOrder[:,oo])
        
        Xticks = np.ceil(np.linspace(1,NumOfIndices,np.amin([NumOfIndices, NIndices]))).astype(int)
        TickNames = [VarNames[i-1] for i in Xticks]

        # Set the y-axis limits and make sure to have two different values
        round_limit = 0.15
        ylimitunc = [
            np.floor(np.amin(Uncorrelated[:,oo])/round_limit)*round_limit,
            np.ceil(np.amax(Uncorrelated[:,oo])/round_limit)*round_limit
        ]
        if ylimitunc[0] == ylimitunc[1]:
            ylimitunc[1] = ylimitunc[0]+0.3
        ylimitint = [
            np.floor(np.amin(Interactive[:,oo])/round_limit)*round_limit,
            np.ceil(np.amax(Interactive[:,oo])/round_limit)*round_limit
        ]
        if ylimitint[0] == ylimitint[1]:
            ylimitint[1] = ylimitint[0] + 0.3
        ylimitcor = [
            np.floor(np.amin(Correlated[:,oo])/round_limit)*round_limit,
            np.ceil(np.amax(Correlated[:,oo])/round_limit)*round_limit
        ] 
        if ylimitcor[0] == ylimitcor[1]:
            ylimitcor[1] = ylimitcor[0] + 0.3
        ylimitfis = [
            np.floor(np.amin(FirstOrder[:,oo])/round_limit)*round_limit,
            np.ceil(np.amax(FirstOrder[:,oo])/round_limit)*round_limit
        ] 
        if ylimitfis[0] == ylimitfis[1]:
            ylimitfis[1] = ylimitfis[0] + 0.3        

        # --- UNCORRELATED INDICES --- #
        data = {'Uncorrelated': Uncorrelated[:,oo]}
        if len(outidx) > 1:
            title= f'Uncorrelated ANCOVA indices, output #{oo+1}'
        else:
            title= f'Uncorrelated ANCOVA indices'   
        fig_ = display_general.display_bar(data, VarNames=TickNames, xaxis_title='Input Variable', 
                    yaxis_title='S<sub>i</sub><sup>U</sup>', yaxis_ticks=ylimitunc, 
                    color=myColorOrder[0], title=title)      

        fig.append(fig_)

        # --- INTERACTIVE indices --- #
        data = {'Interactive': Interactive[:,oo]}
        if len(outidx) > 1:
            title= f'Interactive ANCOVA indices, output #{oo+1}'
        else:
            title= f'Interactive ANCOVA indices'  
        fig_ = display_general.display_bar(data, VarNames=TickNames, xaxis_title='Input Variable', 
                    yaxis_title='S<sub>i</sub><sup>I</sup>', yaxis_ticks=ylimitint, 
                    color=myColorOrder[1], title=title)      

        fig.append(fig_)    

        # --- CORRELATED INDICES --- #
        data = {'Correlated': Correlated[:,oo]}
        if len(outidx) > 1:
            title= f'Correlated ANCOVA indices, output #{oo+1}'
        else:
            title= f'Correlated ANCOVA indices'      
        fig_ = display_general.display_bar(data, VarNames=TickNames, xaxis_title='Input Variable', 
                    yaxis_title='S<sub>i</sub><sup>C</sup>', yaxis_ticks=ylimitcor, 
                    color=myColorOrder[2],title=title)
 
        fig.append(fig_)         

        # --- SUMMED UP FIRST ORDER INDICES --- #
        data = {'FirstOrder': FirstOrder[:,oo]}
        if len(outidx) > 1:
            title= f'First-order ANCOVA indices, output #{oo+1}'
        else:
            title= f'First-order ANCOVA indices'
        fig_ = display_general.display_bar(data, VarNames=TickNames, xaxis_title='Input Variable', 
                    yaxis_title='S<sub>i</sub>', yaxis_ticks=ylimitfis, 
                    color=myColorOrder[3],title=title)

        fig.append(fig_)             

    return fig

def sensitivity_kucherenko(obj, outidx):
    # Collect the relevant information
    if type(obj['Results']) == dict:
        Total = np.array(obj['Results']['Total'])
        FirstOrder = np.array(obj['Results']['FirstOrder'])
        VarNames = obj['Results']['VariableNames']
    elif type(obj['Results']) == list: 
        Total = np.array(obj['Results'][-1]['Total'])
        FirstOrder = np.array(obj['Results'][-1]['FirstOrder'])
        VarNames = obj['Results'][-1]['VariableNames']
    else:
        msg = "obj['Results'] must be either a list or a dictionary"
        raise(msg)    

    if Total.ndim < 2:
        Total = np.expand_dims(Total, axis=1)
        FirstOrder = np.expand_dims(FirstOrder, axis=1)

    # Universal options for the Kucherenko Indices plots
    NIndices = 10

    if len(outidx) > 1:
        fig = []

    for oo in outidx:    
        NumOfIndices = len(Total[:,oo])
        NumOfIndicesPrinted = min([NIndices, NumOfIndices])

        # ---- TOTAL INDICES ---- #
        data = {'Total Indices': Total[0:NumOfIndicesPrinted,oo]}
        TickNames=[VarNames[i] for i in range(NumOfIndicesPrinted)]

        # ---- FIRST ORDER INDICES ---- #
        data['First Order Indices'] = FirstOrder[0:NumOfIndicesPrinted,oo]
        if len(outidx) > 1:
            title= f'Kucherenko indices, output #{oo+1}'
        else:
            title= f'Kucherenko indices'      
        fig_ = display_general.display_bar(data, VarNames=TickNames, 
                xaxis_title='Input Variable', yaxis_title='Sensitivity',showlegend=True, title=title)
        
        if len(outidx) == 1:
            fig = fig_    
        else:
            fig.append(fig_)

    return fig     
