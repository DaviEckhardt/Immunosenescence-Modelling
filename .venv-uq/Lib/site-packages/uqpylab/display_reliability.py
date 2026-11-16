""" Plotting utility imitating the uq_display_uq_reliablity functionality of UQLab """
from uqpylab import sessions
import uqpylab.display_util as display_util
import uqpylab.display_general as display_general
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from scipy.stats import norm
from plotly.subplots import make_subplots


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

# Title
pio.templates[pio.templates.default].layout.title.xanchor = 'center'
pio.templates[pio.templates.default].layout.title.yanchor = 'top'
pio.templates[pio.templates.default].layout.title.x = 0.5
pio.templates[pio.templates.default].layout.title.y = 0.9

def reliability(obj, theInterface, outidx=None, **kwargs):
    # check for input arguments
    if outidx is None:
        outidx = [1]
    if type(outidx)==int:
        outidx = [outidx]
    outidx = [i-1 for i in outidx]

    method = obj['Options']['Method'].lower()

    if method == 'mcs':
        return mc_display(obj, outidx)
    elif method in ['form', 'sorm']:
        return sorm_display(obj, outidx)
    elif method == 'is':
        return importancesampling_display(obj, outidx)
    elif method == 'subset':
        return subsetsim_display(obj, theInterface, outidx)
    elif method == 'akmcs':
        return akmcs_display(obj, theInterface, outidx)
    elif method in ['activelearning', 'alr']:
        return activelearning_display(obj, theInterface, outidx)
    elif method == 'sser':
        return sser_display(obj, theInterface, outidx, **kwargs)
    else:
        print("Select from 'mcs', 'sorm', 'form', 'is', 'subset', 'akmcs', 'activelearning', 'alr', or 'sser' method!")

def mc_display(obj, outidx, **kwargs):

    # collect the relevant information
    BatchSize = obj['Internal']['Simulation']['BatchSize']
    History = obj['Results']['History']

    # consistency checks
    if type(History) == dict: 
        History = [History]

    # initialize a list to store the figures
    fig = []

    # for each response quantity # TODO: test for more than one response quantity
    for oo in outidx:
        
        # prepare data to be plotted
        Pf = np.array(History[oo]['Pf'])
        Conf = np.array(History[oo]['Conf'])
        iter = np.size(Pf)

        # only plot when more than one iteration has been done in MCS   
        if iter != 1: 
            # Plot the convergence curve for the failure probability estimate 
            fig_ = display_general.plotConfidence(yvalue=Pf, LB=Pf-Conf, UB=Pf+Conf, BatchSize=BatchSize, xaxis_title='N', yaxis_title='P<sub>f')
            fig_.update_layout(title= f'MCS - Convergence of P<sub>f') 
            fig.append(fig_) 

            # Plot the convergence curve for the reliability index (beta)
            fig_ = display_general.plotConfidence(yvalue=-norm.ppf(Pf), LB=-norm.ppf(Pf-Conf), UB=-norm.ppf(Pf+Conf), BatchSize=BatchSize, xaxis_title='N', yaxis_title='β<sub>MC')
            fig_.update_layout(title= f'MCS - Convergence of β<sub>MC')  
            fig.append(fig_) 

        # Plot safe/fail sample points only for 2-dimensional problem
        if np.array(History[-1]['X']).shape[1] == 2:
            if 'SaveEvaluations' in obj['Internal'] and obj['Internal']['SaveEvaluations']:
                nplot = int(np.amin([len(np.array(History[-1]['G'])), 1e4]))
                LSF = np.expand_dims(np.array(History[-1]['G']),axis=1)[:nplot,oo]
                XSamples = np.array(History[-1]['X'])[:nplot,:]
                fig_ = display_general.sample_points(Samples=XSamples, LSF=LSF, title='MCS - Samples')
                fig.append(fig_)
    
    return fig

def sorm_display(obj, outidx, **kwargs):
    # collect the relevant information
    History = obj['Results']['History']
    MethodName = obj['Internal']['Method'].upper()
    # consistency checks
    if type(History) == dict:
        History = [History]

    # colors
    UQBlue = display_util.colorOrder(1)[0]

    # initialize a list to store figures
    fig = []

    for oo in outidx:
        # prepare data to plot
        BetaHL = History[oo]['BetaHL']
        iter = np.size(BetaHL)
        N = np.arange(1,iter+1)

        # Display the evolution of the reliability index
        fig_=go.Figure(
            data=go.Scatter(
                x=N,
                y=BetaHL,
                mode='markers+lines',
                marker_color=UQBlue,
                marker_symbol='square'
            )
        )

        # update layout
        fig_.update_layout(
            title=f'{MethodName} - Convergence of β<sub>HL',
            xaxis_title_text = 'number of iterations',
            yaxis_title_text = 'β<sub>HL',
            xaxis_dtick=1,
        )

        fig.append(fig_)

        # Display iterations, design point and FORM plane
        if np.array(History[-1]['X']).shape[1] == 2:
            # Plot the algorithm steps
            UstarValues = np.array(History[oo]['U']) # FORM steps
            traces = [
                # FORM steps
                go.Scatter(
                    x=UstarValues[:,0],
                    y=UstarValues[:,1],
                    mode='markers+lines',
                    marker_symbol='triangle-right',
                    marker_color=UQBlue,
                    marker_size=20,
                    name='Iterations'
                ),
                # highlight in black the starting point
                go.Scatter(
                    x=[UstarValues[0,0]],
                    y=[UstarValues[0,1]],
                    mode='markers',
                    marker_color='black',
                    marker_size=20,
                    marker_symbol='triangle-right',
                    showlegend=False,
                    name='Starting point'
                ),
                # highlight in green the ending point
                go.Scatter(
                    x=[UstarValues[-1,0]],
                    y=[UstarValues[-1,1]],
                    mode='markers',
                    marker_color='green',
                    marker_size=20,
                    marker_symbol='triangle-right',
                    showlegend=False,
                    name='Ending point'
                ),
                # plot the FORM limit state surface in black
                go.Scatter(
                    x=UstarValues[-1,0]+np.array([UstarValues[-1,1], -UstarValues[-1,1]]),
                    y=UstarValues[-1,1]+np.array([-UstarValues[-1,0], +UstarValues[-1,0]]),
                    mode='lines',
                    marker_color='black',
                    name='FORM limit state surface'
                )
            ]

            # layout
            layout = go.Layout(
                title=f'{MethodName} - Design point, failure plane',
                xaxis_title_text='u<sub>1',
                yaxis_title_text='u<sub>2',
                yaxis_dtick=1,
                # legend_x=0.99,
                # legend_xanchor='right',
                # legend_y=0.99,
                # legend_yanchor='top'  
            )

            # create a figure
            fig_=go.Figure(data=traces, layout=layout)

            # update axis limits
            yaxis_min = np.amin(np.array([UstarValues[-1,1]-UstarValues[-1,0], UstarValues[-1,1]+UstarValues[-1,0], UstarValues[0,1]]))
            yaxis_max = np.amax(np.array([UstarValues[-1,1]-UstarValues[-1,0], UstarValues[-1,1]+UstarValues[-1,0], UstarValues[0,1]]))
            fig_.update_yaxes(
                scaleanchor="x",
                scaleratio=1,
                range=[yaxis_min, yaxis_max],
                constrain="domain",
            )

            fig.append(fig_)

    return fig

def importancesampling_display(obj, outidx, **kwargs):

    # collect the relevant information
    BatchSize = obj['Internal']['Simulation']['BatchSize']
    History = obj['Results']['History']

    # consistency checks
    if type(History) == dict:
        History = [History]
    if 'SaveEvaluations' in obj['Internal']:
        History_FORM = obj['Results']['FORM']['History']
        if type(History_FORM) == dict:
            History_FORM = [History_FORM]

    # initialize a list to store figures
    fig = []

    for oo in outidx:
        # prepare data to be plotted
        Pf = np.array(History[oo]['Pf'])
        Conf = np.array(History[oo]['Conf'])
        iter = np.size(Pf)
    
        # only plot when more than one iteration has been done in MCS   
        if iter != 1: 
            fig_ = display_general.plotConfidence(yvalue=Pf, LB=Pf-Conf, UB=Pf+Conf, BatchSize=BatchSize, xaxis_title='N', yaxis_title='P<sub>f')
            fig_.update_layout(title= f'IS - Convergence of Pf') 
            fig.append(fig_) 

        # display design point, cloud of points in 2 dimensions
        if np.array(History[-1]['X']).shape[1] == 2:
            if 'SaveEvaluations' in obj['Internal'] and obj['Internal']['SaveEvaluations']:
                USamples = np.array(History[oo]['U'])
                LSF = np.array(History[oo]['G'])
                fig_ = display_general.sample_points(Samples=USamples, LSF=LSF, title='IS - FORM design point and failure plane')

                # Plot the algorithm steps
                UstarValues = np.array(History_FORM[oo]['U']) # FORM steps
                fig_.add_scatter(
                    x=UstarValues[:,0],
                    y=UstarValues[:,1],
                    mode='markers+lines',
                    marker_symbol='triangle-right',
                    marker_color='red',
                    marker_size=20,
                    name='FORM iterations'
                )

                # highlight in black the starting point
                fig_.add_scatter(
                    x=[UstarValues[0,0]],
                    y=[UstarValues[0,1]],
                    mode='markers',
                    marker_color='black',
                    marker_size=20,
                    marker_symbol='triangle-right',
                    showlegend=False,
                    name='Starting point'
                ),
                # highlight in green the ending point
                fig_.add_scatter(
                    x=[UstarValues[-1,0]],
                    y=[UstarValues[-1,1]],
                    mode='markers',
                    marker_color='green',
                    marker_size=20,
                    marker_symbol='triangle-right',
                    showlegend=False,
                    name='Ending point'
                ),
                # plot the FORM limit state surface in black
                fig_.add_scatter(
                    x=UstarValues[-1,0]+np.array([UstarValues[-1,1], -UstarValues[-1,1]]),
                    y=UstarValues[-1,1]+np.array([-UstarValues[-1,0], +UstarValues[-1,0]]),
                    mode='lines',
                    marker_color='black',
                    name='FORM limit state surface'
                )
                # update layout
                fig_.update_layout(
                    xaxis_title_text='u<sub>1',
                    yaxis_title_text='u<sub>2',
                    # legend_x=0.99,
                    # legend_xanchor='right',
                    # legend_y=0.99,
                    # legend_yanchor='top' 
                )

                fig.append(fig_)
    
    
    return fig

def subsetsim_display(obj, theInterface, outidx, **kwargs):

    # collect the relevant information
    History = obj['Results']['History']

    # consistency checks
    if type(History) == dict:
        History = [History]    
    if type(theInterface.getInput(obj['Internal']['Input'])['Marginals']) == list:
        numMarginals = len(theInterface.getInput(obj['Internal']['Input'])['Marginals'])
    elif type(theInterface.getInput(obj['Internal']['Input'])['Marginals']) == dict:
        numMarginals = 1
    else:
        raise ValueError('Unsupported type of Marginals.')

    # initialize a list to store figures
    fig = []

    # check whether the model evaluations were stored
    if 'SaveEvaluations' in obj['Internal'] and obj['Internal']['SaveEvaluations']:
        for oo in outidx:
            # display the samples of each subset in 1- and 2-dimensional cases
            # Scatter plots of the subset sample for 2-dimensional problems
            if numMarginals == 2:
                colorOrder = display_util.colorOrder(obj['Results']['NumberSubsets'])
                traces = [] # one trace for each subset
                # for each subset
                for ii in range(len(History[oo]['q'])):
                    traces.append(
                        go.Scatter(
                            x=np.array(History[oo]['X'][ii])[:,0], 
                            y=np.array(History[oo]['X'][ii])[:,1],
                            mode='markers',
                            marker_symbol='circle',
                            marker_color=colorOrder[ii],
                            name=f'Subset {ii}'
                            ),
                    )

                # layout
                layout = go.Layout(
                   title='SubsetSim - Samples in each subset',
                   xaxis_title_text='x<sub>1',
                   yaxis_title_text='x<sub>2',
                   showlegend=False
                )

                if len(outidx) > 1:
                    fig.append(go.Figure(data=traces, layout=layout))
                else:
                    fig = go.Figure(data=traces, layout=layout)

            # Histogram plots of the subset sample for 1-dimensional problems
            elif numMarginals == 1:
                data = {}
                for ii in range(len(History[oo]['q'])):
                    data[f'Subset_{ii+1}'] = History[oo]['X'][ii]
                fig_ = display_general.display_histogram(data, title='SubsetSim - Samples in each subset', xaxis_title='x', yaxis_title='count')

                if len(outidx) > 1:
                    fig.append(fig_)
                else:
                    fig = fig_

            # Plot nothing for more than 2 dimensions
    
    return fig

def akmcs_display(obj, theInterface, outidx, **kwargs):
    # collect the relevant information
    History = obj['Results']['History']

    # consistency checks
    if type(History) == dict:
        History = [History]
    CoV = obj['Results']['CoV']
    if type(CoV) != list:
        CoV = [CoV]
    
    # initialize a list to store figures
    fig = []

    for oo in outidx:
        # prepare data
        Nstart = History[oo]['NInit']
        xvalue = np.array(History[oo]['NSamples'])+Nstart
        yvalue = np.array(History[oo]['Pf'])
        LB = np.array(History[oo]['PfLower'])
        UB = np.array(History[oo]['PfUpper'])

        # Plot confidence interval of Pf estimate
        fig_ = display_general.plotConfidence(
            yvalue=yvalue, LB=LB, UB=UB, xvalue=xvalue, 
            xaxis_title='Number of samples', yaxis_title='P<sub>f')
        # change legend name for confidence interval of Pf estimate
        fig_['data'][1]['name'] = 'P<sub>f</sub><sup>+</sup>, P<sub>f</sub><sup>-' 

        # Plot error bar for MCS confindence interval
        bound = CoV[oo]*norm.ppf(1-obj['Internal']['Simulation']['Alpha']/2,0,1)*History[oo]['Pf'][-1]
        fig_.add_trace(
            go.Scatter(
                x=[History[oo]['NSamples'][-1]+Nstart],
                y=[History[oo]['Pf'][-1]],
                marker_color='black',
                name='CI MCS',
                legendrank=3,
                error_y=dict(
                    type='data',
                    array=[bound],
                    color='black', 
                    visible=True
                    )
            )
        )

        # update layout
        fig_.update_layout(
            title= f'AK-MCS - Convergence of Pf',
            # legend_x=0.99,
            # legend_xanchor='right',
            # legend_y=0.01,
            # legend_yanchor='bottom'
        )
        
        fig.append(fig_)

        ## for the 2-dimensional case, plot the safe and failed samples of the experimental design
        if 'SaveEvaluations' in obj['Internal'] and obj['Internal']['SaveEvaluations']:
            # let's handle constants
            myInput = theInterface.getInput(obj['Internal']['Input'])
            NCidx = np.array(myInput['nonConst'],dtype=int)-1
            if len(NCidx)==2:
                fig_ = display_general.limitState(obj, outidx, theInterface)
                fig.append(fig_)
                
    return fig

def sser_display(obj, theInterface, outidx, history=True, limitState=True, displaysse=False, **kwargs):
    ## consistency checks
    # currently only single outidx is supported for SSER
    if len(outidx) != 1 and outidx[0] != 0:
       raise RuntimeError('SSER currently only supports single outputs') 
    else:
        outidx = 0
    
    # collect the relevant information
    mySSE = theInterface.extractFromAnalysis(parentName=obj["Name"],objPath="Results.SSER")
    myInput = theInterface.getInput(obj['Internal']['Input'])

    ## Check for input arguments
    # set optional arguments
    # 'densityplot' plots an mDim parameter density plot
    history_flag = history
    # 'limitstate' displays the limit-state function approximation
    if len(myInput['nonConst']) != 2:
        limitState=False 
    displayLimitState_flag = limitState
    # 'displaysse' forward the SSLE object to the SSE display function
    displaySSE_flag = displaysse

    # # initialize a list to store figures
    fig = []

    ## Convergence plots
    if history_flag:
        # init
        colorOrder = display_util.colorOrder(2)

        # compute experimental design sizes
        maxRef = mySSE['SSE']['currRef']
        sampleRef = mySSE['SSE']['ExpDesign']['ref']
        ref = np.arange(0,maxRef,dtype=int)
        NED = np.empty((maxRef,))
        NED[:] = np.nan

        for rr in ref:
            currNED = np.sum(sampleRef==rr)
            if rr==0:
                NED[rr] = currNED
            else:
                NED[rr] = NED[rr-1]+currNED
        
        ## extract history results
        HistoryContainer=obj['Results']['History']
        beta=np.array(HistoryContainer['Beta'])[:,0]
        betaCI=np.array(HistoryContainer['Beta'])[:,1:]
        Pf=np.array(HistoryContainer['Pf'])[:,0]
        PfCI=np.array(HistoryContainer['Pf'])[:,1:]

        ## plot beta history
        fig_temp = display_general.plotConfidence(beta, betaCI[:,0], betaCI[:,1], xvalue=ref, yaxis_title='SSER')
        fig_ = make_subplots(rows=2, cols=1,
                             shared_xaxes=True,
                             vertical_spacing=0.1)
        # lower bound
        fig_.add_trace(fig_temp.data[0], row=1, col=1)  # lower bound
        # confidence interval
        fig_temp.data[1]['showlegend'] = False
        fig_.add_trace(fig_temp.data[1], row=1, col=1)
        # upper bound
        fig_.add_trace(fig_temp.data[2], row=1, col=1)
        # ssePlot
        fig_.add_trace(fig_temp.data[3], row=1, col=1)
        # y-axis label
        fig_.update_yaxes(title_text='β', row=1, col=1)

        fig_temp = display_general.plot_line(ref, NED)
        fig_.add_trace(fig_temp.data[0], row=2, col=1)
        fig_.update_xaxes(title_text='Refinement steps', 
                          row=2, col=1,
                          tickmode = 'linear',
                          dtick = 1)
        fig_.update_yaxes(title_text='N<sub>X', row=2, col=1)
        fig_.update_layout(showlegend=True,
                           title='SSER - Convergence of β')
        
        fig.append(fig_)

        ## plot Pf history
        fig_temp = display_general.plotConfidence(Pf, PfCI[:,0], PfCI[:,1], xvalue=ref, yaxis_title='SSER')
        fig_ = make_subplots(rows=2, cols=1,
                             shared_xaxes=True,
                             vertical_spacing=0.1)
        # lower bound
        fig_.add_trace(fig_temp.data[0], row=1, col=1)  # lower bound
        # confidence interval
        fig_temp.data[1]['showlegend'] = False
        fig_.add_trace(fig_temp.data[1], row=1, col=1)
        # upper bound
        fig_.add_trace(fig_temp.data[2], row=1, col=1)
        # ssePlot
        fig_.add_trace(fig_temp.data[3], row=1, col=1)
        # y-axis label
        fig_.update_yaxes(title_text='P<sub>f', row=1, col=1)

        fig_temp = display_general.plot_line(ref, NED)
        fig_.add_trace(fig_temp.data[0], row=2, col=1)
        fig_.update_xaxes(title_text='Refinement steps', 
                          row=2, col=1,
                          tickmode = 'linear',
                          dtick = 1)
        fig_.update_yaxes(title_text='N<sub>X', row=2, col=1)
        fig_.update_layout(showlegend=True,
                           title='SSER - Convergence of Pf')
        
        fig.append(fig_)

        if displayLimitState_flag and ('SaveEvaluations' in obj['Internal'] and obj['Internal']['SaveEvaluations']):
            ## Limit-state funtion approximation
            fig_ = display_general.limitState(obj, outidx, theInterface, myInput, mySSE)
            fig.append(fig_)

        ## Forward to SSE display function
        if displaySSE_flag:
            print('displaySSE not implemented yet')
            # run the standard SSE display command
            # TODO: display SSE meta-model
            # fig_ = theInterface.display(mySSE)
            # fig.append(fig_)

    return fig

def activelearning_display(obj, theInterface, outidx, **kwargs):
    # collect the relevant information
    History = obj['Results']['History']

    # consistency checks
    if type(History) == dict:
        History = [History]
    # CoV = obj['Results']['CoV']
    # if type(CoV) != list:
    #     CoV = [CoV]
    
    # initialize a list to store figures
    fig = []

    for oo in outidx:
        # prepare data
        Nstart = History[oo]['NInit']
        xvalue = np.array(History[oo]['NCurrent'])
        yvalue = np.array(History[oo]['Pf'])

        if 'PfLower' in History[oo]:
            LB = np.array(History[oo]['PfLower'])
            UB = np.array(History[oo]['PfUpper'])
            # Plot confidence interval of Pf estimate
            fig_ = display_general.plotConfidence(
                yvalue=yvalue, LB=LB, UB=UB, xvalue=xvalue, 
                xaxis_title='Number of samples', yaxis_title='P<sub>f')
            # change legend name for confidence interval of Pf estimate
            fig_['data'][1]['name'] = 'P<sub>f</sub><sup>+</sup>, P<sub>f</sub><sup>-' 
        else:
            fig_ = display_general.plot_line(
                xvalue=xvalue, yvalue=yvalue, xaxis_title='Number of samples', yaxis_title='P<sub>f')

        if 'CoV' in obj['Results']:
            CoV = obj['Results']['CoV']
            if type(CoV) != list:
                CoV = [CoV]

            # Plot error bar for MCS confindence interval
            bound = CoV[oo]*norm.ppf(1-obj['Internal']['Simulation']['Alpha']/2,0,1)*History[oo]['Pf'][-1]
            fig_.add_trace(
                go.Scatter(
                    x=[History[oo]['NCurrent'][-1]],
                    y=[History[oo]['Pf'][-1]],
                    marker_color='black',
                    name='CI SIM',
                    legendrank=3,
                    error_y=dict(
                        type='data',
                        array=[bound],
                        color='black', 
                        visible=True
                        )
                )
            )

        # update layout
        fig_.update_layout(
            title= f'Active learning - Convergence of Pf',
            # legend_x=0.5,
            # legend_xanchor='center',
            # legend_y=0.99,
            # legend_yanchor='top'
        )
        
        fig.append(fig_)

        ## for the 2-dimensional case, plot the safe and failed samples of the experimental design
        if 'SaveEvaluations' in obj['Internal'] and obj['Internal']['SaveEvaluations']:
            # let's handle constants
            myInput = theInterface.getInput(obj['Internal']['Input'])
            NCidx = np.array(myInput['nonConst'],dtype=int)-1
            if len(NCidx)==2:
                fig_ = display_general.limitState(obj, outidx, theInterface, myInput)
                fig.append(fig_)
                
    return fig











