""" Plotting utility for the abstract functionalities """
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import uqpylab.display_util as display_util
import uqpylab.helpers as helpers
from collections import OrderedDict
from plotly.subplots import make_subplots
import warnings
from scipy import stats

def display_bar(data, VarNames=None, xaxis_title='', yaxis_title='', yaxis_ticks=None, title=None, legend_title='', showlegend=False, grid=True, color=None, width=None):

    # Process data
    if type(data) == dict:
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        msg = "Accepted data type is dict or pandas DataFrame"
        raise RuntimeError(msg)
    if VarNames is not None:
        df.index = VarNames

    # Prepare color scheme
    subset = list(data.keys())
    if color is None:
        color = display_util.colorOrder(len(data))
    if type(color) == str:
        color = [color]
    
    # Bar plot
    if len(data)==1:
        fig = px.bar(
            df,
            color_discrete_map=dict(zip(subset, color)))
    else:
        fig = px.bar(
            df, 
            barmode='group', 
            color_discrete_map=dict(zip(subset, color))
        )

    if grid:
        fig.update_xaxes(mirror=True)
        fig.update_yaxes(mirror=True)

    fig.update_layout(
        xaxis_title= xaxis_title,
        yaxis_title= yaxis_title,
        legend_title= legend_title,
        showlegend=showlegend,    
    ) 

    if width is None and showlegend==True:
        fig.update_layout(
            width=800
        )
    else:
        fig.update_layout(
            width=width
        )

    if yaxis_ticks is not None:
        yaxis_range = yaxis_ticks[:2]
        fig.update_layout(
            yaxis_range=yaxis_range,
        )

        if len(yaxis_ticks) == 3:
            dtick = yaxis_ticks[2]
            fig.update_layout(
                yaxis = dict(
                    tickmode = 'linear',
                    dtick = dtick
            )
        ) 

    if title is not None:
        fig.update_layout(title=dict(text=title,y=0.95))  

    return fig


def morris_plot(data, VarNames=None, xaxis_title='', yaxis_title='', legend_title='', showlegend=False, grid=True):


    cm = display_util.colorOrder(len(data['MU']))

    traces = [
        go.Scatter(
            x=[2*data['minMU'], 2*data['maxMU']], 
            y=[0, 0], 
            mode='lines',
            showlegend=False,
            line=dict(
                color='black',
                width=2
            )
        ),
        go.Scatter(
            x=[0, 0],
            y=[2*data['minMSTD'], 2*data['maxMSTD']],
            mode='lines',
            showlegend=False,            
            line=dict(
                color='black',
                width=2
            )
        ),
    ]

    for i in range(len(data['MU'])):
        traces.append(
            go.Scatter(
                x=[data['MU'][i]], 
                y=[data['MSTD'][i]],
                text=[VarNames[i]],
                customdata= [[[xaxis_title], [yaxis_title]]],
                mode='markers', 
                marker_color=cm[i], 
                textposition='top center',
                textfont_color=cm[i],
                name=VarNames[i],
                hovertemplate =
                '<b>Input Variable: %{text}</b>'+
                '<br><b>%{customdata[0]}</b>: %{x:.5g}'+
                '<br><b>%{customdata[1]}</b>: %{y:.5g}<br>',
            )
        )

    fig = go.Figure(data=traces)

    if grid:
        fig.update_xaxes(mirror=True)
        fig.update_yaxes(mirror=True)

    # fig.update_xaxes(showspikes=True)
    # fig.update_yaxes(showspikes=True)

    if showlegend is None:
        showlegend=True if len(data['MU'])<10 else False

    fig.update_layout(
        xaxis_title= xaxis_title,
        yaxis_title= yaxis_title,
        legend_title= legend_title,
        showlegend=showlegend,
        xaxis_range=[np.amin([data['minMU'], 0-0.1*data['minMU']]), data['maxMU']],
        yaxis_range=[data['minMSTD'], data['maxMSTD']]
    ) 

    return fig


def display_bar_errors(data, xaxis_title='', yaxis_title='',  yaxis_ticks=None, legend_title='', showlegend=False, grid=True, color=None, width=None):

    if not any(isinstance(i, list) for i in data['x_bar']):
        data['x_bar'] = [data['x_bar']]
        data['y_bar'] = [data['y_bar']] 
        data['lb'] = [data['lb']]
        data['ub'] = [data['ub']]    
        if 'trace_name' in data:
            data['trace_name'] = [data['trace_name']]

    numOutputs = len(data['x_bar'])

    if color is None:
        color = display_util.colorOrder(numOutputs)

    if 'trace_name' not in data:
        data['trace_name'] = []
        for i in range(numOutputs):
            data['trace_name'].append(f'dataset #{i}')

    traces = []

    for i in range(numOutputs):
        traces.append(
                go.Bar(
                    x=data['x_bar'][i], 
                    y=data['y_bar'][i],
                    marker={'color': color[i]},
                    name=data['trace_name'][i],
                error_y=dict(
                    type='data',
                    symmetric=False,
                        array=data['ub'][i],
                        arrayminus=data['lb'][i],
                    color='black'
                )
            )
        )

    fig = go.Figure(data=traces)

    if grid:
        fig.update_xaxes(mirror=True)
        fig.update_yaxes(mirror=True)

    fig.update_layout(
        xaxis_title= xaxis_title,
        yaxis_title= yaxis_title,
        legend_title= legend_title,
        showlegend=showlegend,
    ) 

    if showlegend==True and width is None:
        fig.update_layout(
            width=800
        )
    else:
        fig.update_layout(
            width=width
        )

    if yaxis_ticks is not None:
        yaxis_range = yaxis_ticks[:2]
        fig.update_layout(
            yaxis_range=yaxis_range,
        )

        if len(yaxis_ticks) == 3:
            dtick = yaxis_ticks[2]
            fig.update_layout(
                yaxis = dict(
                    tickmode = 'linear',
                    dtick = dtick
            )
        )         

    return fig

def pie_chart(data, VarNames): 
    # Process data
    df = pd.DataFrame(data)
    if VarNames is not None:
        df.index = VarNames 


    fig = px.pie(
        df, 
        values='Values', 
        names=VarNames,
    )

    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        showlegend=False,
    )

    return fig


def plotConfidence(yvalue, LB, UB, xvalue=None, BatchSize=None, xaxis_title=None, yaxis_title=None,**kwargs):

    # prepare data
    if xvalue is None:
        iter = np.size(yvalue)
        xvalue = np.arange(1,iter+1)
    if BatchSize is None:
        BatchSize = 1
    xvalue *= BatchSize

    lineColor = kwargs.get('lineColor', display_util.colorOrder(1)[0])
    lineWidth = kwargs.get('lineWidth', .5)
    boundColor = kwargs.get('boundColor', 'rgba(0,0,0,0.1)')
    legend_entries = kwargs.get('legend', [yaxis_title, 'CI'])
    # alphaN = kwargs.get('alphaN', 0.3)    
    # blue_color = display_util.colorOrder(1)[0]

    fig = go.Figure()

    # lower bound
    fig.add_trace(go.Scatter(
        x=xvalue, 
        y=LB, 
        fill=None,
        mode='lines', 
        line_width = lineWidth,
        line_color=boundColor,
        showlegend=False,
        name=legend_entries[1],
        )
    )

    # confidence interval
    fig.add_trace(go.Scatter(
        x=xvalue, 
        y=UB, 
        fill='tonexty',
        fillcolor=boundColor,
        mode='lines', 
        line_width = lineWidth,
        line_color='white',
        name=legend_entries[1],
        legendrank=2,
        )
    )

    # upper bound
    fig.add_trace(go.Scatter(
        x=xvalue, 
        y=UB, 
        fill=None,
        fillcolor=boundColor,
        mode='lines', 
        line_width = lineWidth,
        line_color=boundColor,
        showlegend=False,
        name=legend_entries[1],
        )
    )

    # mean as a line
    fig.add_trace(go.Scatter(
        x=xvalue, 
        y=yvalue,
        fill=None,
        mode='lines',
        line_color=lineColor,
        name=legend_entries[0],
        legendrank=1,
        )
    )

    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_x=0.99,
        legend_xanchor='right',
        legend_y=0.99,
        legend_yanchor='top'
    )

    return fig


def sample_points(Samples, LSF, title=None):

    colorOrder = display_util.colorOrder(2)

    traces = [
        go.Scatter(
            x=Samples[LSF<=0,0] if len(Samples[LSF<=0,0])!=0 else [None], 
            y=Samples[LSF<=0,1] if len(Samples[LSF<=0,1])!=0 else [None],
            mode='markers',
            marker_symbol='circle',
            marker_color=colorOrder[0],
            name='g(X) ≤ 0'
            ),
        go.Scatter(
            x=Samples[LSF>0,0] if len(Samples[LSF>0,0])!=0 else [None],
            y=Samples[LSF>0,1] if len(Samples[LSF>0,1])!=0 else [None],
            mode='markers',
            marker_symbol='circle',
            marker_color=colorOrder[1],
            name='g(X) > 0'
            ),                       
    ]

    layout = go.Layout(
        xaxis_title_text='x<sub>1',
        yaxis_title_text='x<sub>2',
        showlegend=True,
        legend_x=0.99,
        legend_xanchor='right',
        legend_y=0.99,
        legend_yanchor='top'
    )

    fig = go.Figure(data=traces, layout=layout)

    if title:
        fig.update_layout(
            title=title
        )
    
    return fig


def display_histogram(data, VarNames=None, xaxis_title='', yaxis_title='', yaxis_ticks=None, title=None, legend_title='', showlegend=False, grid=True, color=None, width=None):

    # Histogram properties
    nbins=15
    bin_size = (max(data[list(data.keys())[0]])-min(data[list(data.keys())[0]]))/nbins

    # Prepare color scheme
    subset = list(data.keys())
    if color is None:
        color = display_util.colorOrder(len(data))
    if type(color) == str:
        color = [color]
    
    # Histogram plot    
    fig = go.Figure()
    count = 0
    for key, value in data.items():
        fig.add_trace(
            go.Histogram(
                x=value, 
                name=key, 
                marker_color=color[count],
                xbins_size=bin_size
            )
        )
        count+=1

    if grid:
        fig.update_xaxes(mirror=True)
        fig.update_yaxes(mirror=True)

    fig.update_layout(
        xaxis_title= xaxis_title,
        yaxis_title= yaxis_title,
        legend_title= legend_title,
        showlegend=showlegend,    
    ) 

    if len(data) > 1:
        fig.update_layout(barmode='overlay')
        fig.update_traces(opacity=0.8)

    if width is not None:
        fig.update_layout(
            width=900
        )

    if yaxis_ticks is not None:
        yaxis_range = yaxis_ticks[:2]
        fig.update_layout(
            yaxis_range=yaxis_range,
        )

        if len(yaxis_ticks) == 3:
            dtick = yaxis_ticks[2]
            fig.update_layout(
                yaxis = dict(
                    tickmode = 'linear',
                    dtick = dtick
            )
        ) 

    # revert order of histograms
    # fig.data = fig.data[::-1]

    if title is not None:
        fig.update_layout(title=title)   

    return fig


def plot_line(xvalue, yvalue, line_color=None, xaxis_title=None, yaxis_title=None):

    # prepare data
    if xvalue is None:
        iter = np.size(yvalue)
        xvalue = np.arange(1,iter+1)

    if line_color is None:
        line_color = display_util.colorOrder(1)[0]

    fig = go.Figure()

    # plot line
    fig.add_trace(go.Scatter(
        x=xvalue, 
        y=yvalue,
        mode='lines',
        line_color=line_color,
        showlegend=False
        )
    )

    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )

    return fig
    

def limitState(obj, outIdx, theInterface, myInput=None, myMetamodel=None):
    # validate input
    myAnalysis=obj
    myAnalysisMethodName = myAnalysis['Options']['Method']

    if myAnalysis['Class'] != 'uq_analysis':
        raise RuntimeError('Input needs to be a UQ_ANALYSIS object.')

    if myAnalysis['Type'] != 'uq_reliability':
        raise RuntimeError('Analysis needs to be a reliability analysis.')

    if not myAnalysis['Options']['Method'].lower() in ['sser','activelearning','alr','akmcs']:
        raise RuntimeError('Reliability analysis needs to be metamodel-based.')

    if type(myAnalysis['Internal']['Input']) == str:
        myInput = theInterface.getInput(obj['Internal']['Input'])
    else:
        myInput = obj['Internal']['Input']
    if len(myInput['nonConst']) != 2:
        raise RuntimeError('Works only for 2-dimensional problems.')

    # extract results
    Results = obj['Results']

    # extract metamodel and history
    if myAnalysis['Options']['Method'].lower() == 'sser':
        if type(obj['Results']['SSER']) == str:
            myMetamodel = theInterface.getModel(obj['Results']['SSER'])
            MetaModelName = myMetamodel['Options']['MetaType']
        else:
            raise ValueError('Unsupported type of Model!')

    elif myAnalysis['Options']['Method'].lower() == 'akmcs':
        MetaModelName = obj['Internal']['AKMCS']['MetaModel']
        parentName = obj["Name"]
        objPath = "Results." + MetaModelName 
        myMetamodel = theInterface.extractFromAnalysis(parentName=parentName,objPath=objPath)
        if type(myMetamodel) == list:
            myMetamodel = myMetamodel[outIdx]
    elif myAnalysis['Options']['Method'].lower() == 'alr':
        myAnalysisMethodName = "Active learning"
        MetaModelName = obj['Internal']['ALR']['MetaModel']
        parentName = obj["Name"]
        objPath = "Results.Metamodel" 
        myMetamodel = theInterface.extractFromAnalysis(parentName=parentName,objPath=objPath)
        if type(myMetamodel) == list:
            myMetamodel = myMetamodel[outIdx]

    else:
        raise RuntimeError("Not yet implemented!")

    if myAnalysis['Options']['Method'].lower() == 'sser':
        # get extremes
        minX = []
        maxX = []
        # UQLab uses a table with 11 columns to store data for Nodes in myMetamodel['SSE']['Graph']['Nodes'], namely: 
        # neighbours, bounds, inputMass, ref, level, idx, expansions, refineScore, Pf, History, and PfRepl
        numnodes = int(len(myMetamodel['SSE']['Graph']['Nodes'])/11)
        for dd in range(numnodes):
            Ucurr = np.array(myMetamodel['SSE']['Graph']['Nodes'][9*numnodes+dd]['U'])
            if Ucurr.size > 0:
                myInput1 = theInterface.getInput(myMetamodel['SSE']['Input']['Original'])
                Xcurr = theInterface.invRosenblattTransform(Ucurr, myInput1['Marginals'], myInput1['Copula'])
                if type(minX) == list:
                    minX = np.amin(Xcurr, axis=0, keepdims=True)
                    maxX = np.amax(Xcurr, axis=0, keepdims=True)
                else:
                    minX = np.amin(np.concatenate((minX, Xcurr), axis=0), axis=0, keepdims=True)
                    maxX = np.amax(np.concatenate((maxX, Xcurr), axis=0), axis=0, keepdims=True)

    elif myAnalysis['Options']['Method'].lower() == 'akmcs':
        minX = np.amin(np.array(Results['History']['MCSample']), axis=0, keepdims=True)  # from uq_akmcs_display
        maxX = np.amax(np.array(Results['History']['MCSample']), axis=0, keepdims=True)  # from uq_akmcs_display
    
    elif myAnalysis['Options']['Method'].lower() == 'alr':
        minX = np.amin(np.array(Results['History']['ReliabilitySample']), axis=0, keepdims=True)
        maxX = np.amax(np.array(Results['History']['ReliabilitySample']), axis=0, keepdims=True)

    else:
        raise RuntimeError("Not yet implemented!")

    # extract experimental design
    X = np.array(myMetamodel['ExpDesign']['X'])
    G = np.array(myMetamodel['ExpDesign']['Y'])

    # init
    colorOrder = display_util.colorOrder(2)

    # compute grid
    NGrid = 200
    [xx, yy] = np.meshgrid(np.linspace(minX[0,0], maxX[0,0], NGrid),
                           np.linspace(minX[0,1], maxX[0,1], NGrid))
    XGrid = np.stack((xx.flatten('F'), yy.flatten('F'))).T
    zz = theInterface.evalModel(myMetamodel, XGrid)
    zz = np.reshape(zz, xx.shape, order='F')

    traces = [
            go.Contour(
                x=xx.flatten(),
                y=yy.flatten(),
                z=zz.flatten(),
                contours_coloring='lines',
                line_width=2,
                contours_start=0,
                contours_end=0,
                colorscale=[[0, 'rgb(0,0,0)'], [1, 'rgb(0,0,0)']],
                name='g(X) = 0',
                showlegend=True,
                showscale=False,
                coloraxis=None
            )
    ]

    if np.any(X[G<=0,0]):
        traces.append(go.Scatter(
            x=X[G<=0,0] if len(X[G<=0,0])!=0 else [None], 
            y=X[G<=0,1] if len(X[G<=0,1])!=0 else [None],
            mode='markers',
            marker_symbol='square',
            marker_color=colorOrder[0],
            name='g(X) ≤ 0'
            ))

    if np.any(X[G>0,0]):
        traces.append(go.Scatter(
            x=X[G>0,0] if len(X[G>0,0])!=0 else [None], 
            y=X[G>0,1] if len(X[G>0,1])!=0 else [None],
            mode='markers',
            marker_symbol='cross',
            marker_color=colorOrder[1],
            name='g(X) > 0'
            ))

    layout = go.Layout(
        xaxis_title_text=myInput['Marginals'][0]['Name'],
        yaxis_title_text=myInput['Marginals'][1]['Name'],
        showlegend=True,
        legend_x=0.99,
        legend_xanchor='right',
        legend_y=0.99,
        legend_yanchor='top',
        title=f"{myAnalysisMethodName} - Limit state approximation"
    )

    fig = go.Figure(data=traces, layout=layout)

    fig.update_xaxes(range=[minX[0,0], maxX[0,0]])
    fig.update_yaxes(range=[minX[0,1], maxX[0,1]])
    return fig

def scatterDensity(X, paramLabelsScatter, Color='rgb(128, 179, 255)',Limits=None,Title='',Points=None):

    # Data shape and checking dimensions
    nPoints, nDim = X.shape    

    # Points option
    if Points:
        plotPoints_flag = True
        # if type(Points) is dict:
        #     Points = [Points]

        if helpers.isnumeric(Points):
            if isinstance(Points,list):
                Points = np.array(Points)
            if nDim != Points.shape[1]:
                raise('Additional plot points do not have a compatible size')  
            # store in a dictionary
            plotPoints = {}
            plotPoints['X'] = [Points]
            plotPoints['Type'] = ['custom']
            plotPoints['Collection'] = Points
        elif isinstance(Points,dict):
            plotPointsCollection = []
            #Check dimensions
            for ii in range(len(Points['X'])):
                if nDim != np.array(Points['X'][ii]).shape[1]:
                    raise('Additional plot points do not have a compatible size')
                # store in collection
                for item in Points['X'][ii]:
                    plotPointsCollection.append(item)
                # plotPointsCollection.append(Points['X'][ii])
            plotPoints = Points
            plotPoints['Collection'] = np.array(plotPointsCollection)
            if isinstance(plotPoints['Type'],str):
                plotPoints['Type'] = [plotPoints['Type']]
        else:
            raise("Supplied points don't have required format.")
    else:
        plotPoints_flag = False

    # limits option
    if Limits is not None:
        plotLimits = Limits
        if plotLimits.shape[1] != nDim:
            raise('Supplied limits do not match supplied points')
    else:
        # infer limits from data
        # get smallest and largest sample values in each dimension
        if plotPoints_flag:
            minSample = np.amin(np.vstack((X,plotPoints['Collection'])),axis=0)
            maxSample = np.amax(np.vstack((X,plotPoints['Collection'])),axis=0)
        else:
            minSample = np.amin(X,axis=0)
            maxSample = np.amax(X,axis=0) 
        plotLimits = np.vstack((minSample, maxSample))
    
    ## Setting of plots and subplots
    # nPlots = nDim  # number of plots in each row and column
    # cutOffExp = 4 # number of digits of precision for ticks for %g format

    ## Compute the font size scaling factor (Scale font size based on number of plots linearly between 1 (up to 2 subplots) and 0.4 (more than 10 subplots))
    lowerPlots = 2
    upperPlots = 10
    lowerScale = 1
    upperScale = 0.5#0.27

    # Perform linear interpolation within the provided data range
    if nDim <= lowerPlots:
        fontScale = lowerScale
    elif nDim >= upperPlots:
        fontScale = upperScale
    else:
        fontScale = np.interp(nDim, [1, lowerPlots, upperPlots], [lowerScale, lowerScale, upperScale])

    # Subplots
    fig = make_subplots(
        rows=nDim, 
        cols=nDim, 
    )

    # data = pd.DataFrame(X, columns=paramLabelsScatter)

    # Loop through each variable and create scatter plots
    for ii in range(nDim):
        for jj in range(ii+1): # plot only the lower diagonal of the subplots
            var1 = paramLabelsScatter[ii]
            var2 = paramLabelsScatter[jj]
            if ii == jj:
                # Diagonal: Histogram
                counts, bins = np.histogram(X[:,ii], bins=BinWidth_ScottsRule(X[:,ii]))
                bins = 0.5 * (bins[:-1] + bins[1:])
                fig.add_trace(
                    # go.Histogram(
                    #     x=X[:,ii],
                    #     nbinsx=10,
                    #     showlegend=False,
                    #     marker_color=Color,
                    # ),
                    go.Bar(
                        x=bins, 
                        y=counts,
                        marker_color=Color,
                        marker_line_color=Color,                        
                        showlegend=False,
                    ),
                    row=ii+1, col=jj+1
                )

            elif ii > jj:
                # Non-diagonal: Scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=X[:,jj],
                        y=X[:,ii],
                        mode='markers',
                        marker_color=Color,
                        marker_size=1,
                        marker_opacity=0.5, 
                        showlegend=False,
                    ),
                    row=ii+1, col=jj+1
                )

            if plotPoints_flag:
                nPointGroups = len(plotPoints['X'])
                # get colors
                pointColors = display_util.colorOrder(nPointGroups+1)
                pointColors.pop(0) # discard the blue color that is used for the post sample data
                for pp in range(nPointGroups):
                    plotPointsCurr = np.array(plotPoints['X'][pp])
                    if ii==jj: 
                        # Diagonal: Histogram
                        for xx in range(plotPointsCurr.shape[0]):
                            fig.add_trace(
                                go.Scatter(
                                    x=[plotPointsCurr[xx,ii], plotPointsCurr[xx,ii]],
                                    y=[0,np.amax(counts)*1.05],
                                    mode='lines',
                                    line_width=1,
                                    line_color=pointColors[pp],
                                    showlegend=False,
                                ),row=ii+1, col=jj+1
                            )
                            if xx==0 and ii==0 and jj==0:
                                fig.update_traces(name=plotPoints['Type'][pp], 
                                                  showlegend=True,
                                                #   selector = {'row': ii+1, 'col': jj+1}
                                                  selector=dict(mode='lines', type='scatter', line_color=pointColors[pp]) ,
                                )
                    else:
                        # Scatter plot
                        for xx in range(plotPointsCurr.shape[0]):
                            fig.add_trace(
                                go.Scatter(
                                x=plotPointsCurr[:,jj],
                                y=plotPointsCurr[:,ii],
                                marker_symbol='cross',
                                mode='markers',
                                marker_size=5,
                                marker_line_width=0,
                                marker_color=pointColors[pp],
                                showlegend=False,
                                ),
                                row=ii+1, col=jj+1
                            )

            # update all x axes ticks (range)
            tickformat='8.2e'
            fig.update_xaxes(
                range=[plotLimits[0,jj],plotLimits[1,jj]],
                tickvals=[plotLimits[0,jj],plotLimits[1,jj]],
                tickformat=tickformat,
                tickfont_size=fontScale*16*0.7,
                tickangle=45,             
                row=ii+1, 
                col=jj+1,                
            )
            # update y axes ticks for scatter plots (non-diagonal subplots)
            if ii != jj:
                fig.update_yaxes(
                    range=[plotLimits[0,ii],plotLimits[1,ii]],
                    tickvals=[plotLimits[0,ii],plotLimits[1,ii]],
                    tickformat=tickformat,
                    tickfont_size=fontScale*16*0.7,
                    tickangle=45,                    
                    row=ii+1, 
                    col=jj+1,
                )
            # update y axes ticks for histograms (diagonal subplots)
            else:
                fig.update_yaxes(
                    range=[0,np.amax(counts)*1.05],
                    tickvals=[0,np.amax(counts)*1.05],
                    tickformat=tickformat,
                    tickfont_size=fontScale*16*0.7,
                    tickangle=45,                         
                    row=ii+1, 
                    col=jj+1,                          
                )

            # Remove x-axis tick labels for all but the last row
            if ii != nDim-1:
                fig.update_xaxes(
                    # tickvals=[np.min(X[:,jj]), np.max(X[:,jj])],
                    title_text='', 
                    showticklabels=False, 
                    row=ii+1, 
                    col=jj+1,
                )

            # Remove y-axis tick labels for all but the first column and for the first plot
            if jj != 0 or (jj == 0 and ii == 0):
                fig.update_yaxes(                     
                    title_text='', 
                    showticklabels=False, 
                    row=ii+1, 
                    col=jj+1,
                )

    # Customizing the axis lines for each subplot
    fig.update_xaxes(showline=True, linewidth=.5, linecolor='black',mirror=True) 
    fig.update_yaxes(showline=True, linewidth=.5, linecolor='black',mirror=True)
    # Customizing the gaps between bars
    fig.update_layout(bargap=0)

    # Set the x-axis labels in the last row
    LB = int(nDim*(nDim-1)+1)
    UB = int((nDim**2)+1)
    myRange = list(range(LB,UB))
    for ii,val in enumerate(myRange):
        xaxis = f"xaxis{val}"
        fig['layout'][xaxis]['title'] = paramLabelsScatter[ii]

    # Set the y-axis labels in the first column
    myRange = np.arange(1,nDim**2,nDim)
    for ii,val in enumerate(myRange):
        yaxis = f"yaxis{val}"
        fig['layout'][yaxis]['title'] = paramLabelsScatter[ii] 

    # Update layout
    fig.update_layout(
        font_size=16*fontScale,
        title=Title,
        showlegend=True,
        legend_x=1,
        legend_y=1,
    )

    # fig.show()

    return fig 

def plotSinglePred(Sample, Data, pointEstimatePred=None):
    '''
    creates a plot of single prediction
    '''
    
    # Use histogram for simple data (True: with point estimate)
    pointEstimatePredFlag = bool(pointEstimatePred)

    # Use custom colors
    priorColor = 'rgb(128,179,255)'
    postColor = 'rgb(0,51,153)'

    # nbins = 20

    # Determine plotType
    plotType = predPlotType(Sample)

    if plotType == 'priorPost':
        # Create histogram plots for both prior & posterior predictive runs
        priorRuns = np.array(Sample['PriorPred'])
        postRuns = np.array(Sample['PostPred'])

        counts, bins = np.histogram(priorRuns, bins=BinWidth_ScottsRule(priorRuns))
        xData = 0.5 * (bins[:-1] + bins[1:])
        # normalize the data
        normfac = 1/np.sum(counts*np.mean(np.diff(xData)))
        yData = counts*normfac
        fig = go.Figure(go.Bar(x=xData, y=yData, marker_color=priorColor, marker_line_color=priorColor, name='prior predictive'))

        # Posterior predictive runs
        # Compute the center of the bins of the histogram
        barSpacing = xData[1] - xData[0]
        # If postRuns outside xData, extend xData - TODO: fix this!!!!
        if np.max(postRuns) > np.max(xData):
            xData = np.hstack([xData[:-1],np.arange(xData[-1], np.max(postRuns), barSpacing)])
            bins = xData-barSpacing*.5
            lastElem = np.max(postRuns) if np.max(postRuns) > bins[-1]+barSpacing else bins[-1]+barSpacing
            bins = np.append(bins, lastElem)
        if np.min(postRuns) < np.min(xData):
            pref = np.arange(xData[0], np.min(postRuns), -barSpacing)
            pref = np.append(pref, pref[-1]-barSpacing)
            pref = np.flipud(pref)
            xData = np.hstack([pref[:-1], xData])
            # xData = np.hstack([np.arange(np.min(postRuns), xData[0], barSpacing), xData])
            bins = xData-barSpacing*.5
            bins = np.append(bins, bins[-1]+barSpacing)
        counts, bins = np.histogram(postRuns, bins=bins)
        xData = 0.5 * (bins[:-1] + bins[1:])
        # normalize the data
        normfac = 1/np.sum(counts*np.mean(np.diff(xData)))
        yData = counts*normfac        
        fig.add_trace(go.Bar(x=xData, y=yData, marker_color=postColor, marker_line_color=postColor, name='posterior predictive'))
        fig.update_layout(barmode='overlay') 
        fig.update_layout(bargap=0,bargroupgap=0)

    elif plotType == 'prior':
        priorRuns = np.array(Sample['PriorPred'])
        counts, bins = np.histogram(priorRuns, bins=BinWidth_ScottsRule(priorRuns))
        xData = 0.5 * (bins[:-1] + bins[1:])
        # normalize the data
        normfac = 1/np.sum(counts*np.mean(np.diff(xData)))
        yData = counts*normfac        
        fig = go.Figure(go.Bar(x=xData, y=yData, 
                               marker_color=priorColor, 
                               marker_line_color=priorColor, 
                               name='prior predictive'))
        fig.update_layout(bargap=0)

    elif plotType == 'post':
        postRuns = np.array(Sample['PostPred'])
        counts, bins = np.histogram(postRuns, bins=BinWidth_ScottsRule(postRuns))
        xData = 0.5 * (bins[:-1] + bins[1:])
        # normalize the data
        normfac = 1/np.sum(counts*np.mean(np.diff(xData)))
        yData = counts*normfac        
        fig = go.Figure(go.Bar(x=xData, y=yData, 
                               marker_color=postColor, 
                               marker_line_color=postColor, 
                               name='posterior predictive'))
        fig.update_layout(bargap=0)

    else:
        raise('Unknown plot type.')

    if pointEstimatePredFlag:   
        currModel = Data['MOMap'][0]
        currOut = Data['MOMap'][1]
        ylimmax = np.max(yData)*1.1


        if isinstance(pointEstimatePred, dict):
            pointEstimatePredCurr = np.array(pointEstimatePred['Out'],ndmin=2)[:,currOut-1]
            legendName = 'model at ' + pointEstimatePred['Type'].lower()
            ## plot
            fig.add_trace(go.Scatter(
                x=pointEstimatePredCurr * np.ones((1,2)).squeeze(), 
                y=np.array([0, ylimmax]),
                mode='lines',
                name=legendName))
        elif isinstance(pointEstimatePred, list):
            for pp in range(len(pointEstimatePred)):
                if isinstance(pointEstimatePred[pp], list):
                    pointEstimatePredCurr = np.array(pointEstimatePred[pp][currModel]['Out'],ndmin=2)[:,currOut-1] 
                    legendName = 'model at ' + pointEstimatePred[pp][1]['Type'].lower()
                elif isinstance(pointEstimatePred[pp], dict):
                    pointEstimatePredCurr = np.array(pointEstimatePred[pp]['Out'],ndmin=2)[:,currOut-1]
                    legendName = 'model at ' + pointEstimatePred[pp]['Type'].lower()
                ## plot
                fig.add_trace(go.Scatter(
                    x=pointEstimatePredCurr * np.ones((1,2)).squeeze(), 
                    y=np.array([0, ylimmax]),
                    mode='lines',
                    name=legendName))                  
        else:
            raise("Unexpected data type")


            
    ## Plot the histogram of the data
    DataY = np.array(Data['y'])
    xx = np.vstack((DataY, DataY)).T
    yy = np.vstack((np.zeros(DataY.shape), np.ones(DataY.shape))).T * .15 * ylimmax
    for pp in range(xx.shape[0]):
        if pp != 0:
            fig.add_trace(go.Scatter(x=xx[pp,:], y=yy[pp,:], mode='lines', marker_color='rgb(0,255,0)', name='data', showlegend=False))
        else:
            fig.add_trace(go.Scatter(x=xx[pp,:], y=yy[pp,:], mode='lines', marker_color='rgb(0,255,0)', name='data'))

    # fig.update_xaxes(title=r'$\mathcal{Y}$')
    fig.update_xaxes(title='Y')
    fig.update_yaxes(range=[0, ylimmax])
    return fig

def plotSeriesPred(Sample, Data, pointEstimatePred=None):
    '''
    creates a plot of serial (non-scalar) predictions.    
    '''
    
    pointEstimatePredFlag = bool(pointEstimatePred)

    # Use custom colors
    priorColor = 'rgb(128,179,255)'
    postColor = 'rgb(0,51,153)'

    nbins = 20

    # Determine plotType
    plotType = predPlotType(Sample)

    ## Create the plot
    # Use violin plot for histogram of data series
    if plotType == 'priorPost':
        # Both prior and posterior predictive runs
        priorRuns = np.array(Sample['PriorPred'])
        postRuns = np.array(Sample['PostPred'])
        # Plot prior predictive runs
        priorPlot = violinplot(priorRuns, FaceColor=priorColor, name='prior predictive')
        # Plot posterior predictive runs
        postPlot = violinplot(postRuns, FaceColor=postColor, name='posterior predictive')
        # Merge both plots together
        fig = go.Figure(
            data=priorPlot.data+postPlot.data, 
            layout=priorPlot.layout,
        )
    elif plotType == 'prior':
        # Only prior prior predictive runs
        priorRuns = np.array(Sample['PriorPred'])
        # Plot prior predictive runs
        priorPlot = violinplot(priorRuns, FaceColor=priorColor, name='prior predictive')
        fig = go.Figure(data=priorPlot.data, layout=priorPlot.layout)
    elif plotType == 'post':
        # Only posterior predictive runs
        postRuns = np.array(Sample['PostPred'])
        # Plot posterior predictive runs
        postPlot = violinplot(postRuns, FaceColor=postColor, name='posterior predictive')
        fig = go.Figure(data=postPlot.data, layout=postPlot.layout)
    else:
        raise('Unknown plot type.')

    # Use scatter plot for observed data
    MOMap = np.array(Data['MOMap'],ndmin=2)
    DataY = np.array(Data['y'],ndmin=2)
    if MOMap.shape[0] == 1:
        MOMap = MOMap.T    
        DataY = DataY.T    

    xDummy = np.arange(1,DataY.shape[1]+1)
    for ii in range(xDummy.size):
        yCurr = DataY[:,ii]
        for jj in range(yCurr.size):
            trace = go.Scatter(
                x=np.array(ii+1),
                y=np.array(yCurr[jj]),
                marker_size=5,
                marker_line_color='rgb(0,255,0)',
                # marker_color='rgb(0,255,0)',
                marker_line_width=1,
                marker_symbol='x-thin',
                mode='markers',
                name='data',
                showlegend=False,
            )
            fig.add_trace(trace)
            if ii == 0 and  jj == 0:
                fig.update_traces(
                    selector={'name': 'data'},
                    showlegend=True,
                )

    if pointEstimatePredFlag:
        # assign model runs to current Data based on MOMap
        # define plot color order   
        if isinstance(pointEstimatePred, dict):
            numPointEstPred = 1
        elif isinstance(pointEstimatePred, list):
            numPointEstPred = len(pointEstimatePred)
        else:
            raise('pointEstimatePred should be a list or dict.')
        plotColors = display_util.colorOrder(numPointEstPred+1)

        for pp in range(numPointEstPred):
            # if pp is 1, pointEstimatePred is not iterable
            if numPointEstPred == 1:
                pointEstimatePredCurr = np.zeros((np.array(pointEstimatePred['Out'], ndmin=2).shape[0],DataY.shape[1]))   
                name=f"model at {pointEstimatePred['Type'].lower()}"      
            else:
                pointEstimatePredCurr = np.zeros((np.array(pointEstimatePred[pp][1]['Out'], ndmin=2).shape[0],DataY.shape[1]))  
                name=f"model at {pointEstimatePred[pp][0]['Type'].lower()}"         
            # loop over data points
            for ii in range(pointEstimatePredCurr.shape[1]):
                currModel = MOMap[0,ii]
                currOut = MOMap[1,ii]
                if numPointEstPred == 1:
                    pointEstimatePredCurr[:,ii] = np.array(pointEstimatePred['Out'],ndmin=2)[:,currOut-1]
                else:
                    pointEstimatePredCurr[:,ii] = np.array(pointEstimatePred[pp][currModel]['Out'],ndmin=2)[:,currOut-1]
            for jj in range(pointEstimatePredCurr.shape[0]):
                pointEstimatePlot = go.Scatter(
                    x=xDummy,
                    y=pointEstimatePredCurr[jj,:],
                    marker_size=5,
                    marker_line_color=plotColors[pp+1],
                    marker_line_width=1,
                    marker_symbol='cross-thin',
                    mode='markers',
                    showlegend=True,     
                    name=name,            
                )

            fig.add_trace(pointEstimatePlot)
            


        nTicks = 10
        if len(xDummy) >= nTicks:
            xDummy = np.ceil(np.linspace(1, len(xDummy), nTicks))
        
        labels = [f'y<sub>{int(xDummy[ii])}' for ii in range(len(xDummy))]

        fig.update_xaxes(
            title='Data index (-)',
            tickmode='array',
            tickvals=xDummy,
            ticktext=labels,    
            # tickformat='y_%d'
        ) 
        fig.update_yaxes(title='Y')
        fig.update_layout(title='Predictive distribution')


    return fig


def predPlotType(Sample):
    '''
    determines predictive plot type based on contents of Sample
    '''
    if 'PriorPred' in Sample and 'PostPred' in Sample:
        plotType='priorPost'
    elif 'PriorPred' in Sample:
        plotType='prior'
    elif 'PostPred' in Sample:
        plotType='post'
    else:
        raise ValueError('Sample does not have the correct fields')
    return plotType

def traceplot(X, labels=None):
    '''
    creates trace plots of the series of points with the kernel density-based marginal estimates.
    '''

    if X is None:
        raise('Not enough input arguments.')
    X = np.array(X,ndmin=3)

    [N,M,_] = X.shape

    if not labels:
        # detault labels
        labels = [f'X<sub>{ii}' for ii in range(M)]

    # normal labels
    paramLabels = labels

    # pdf labels
    # paramPDFLabels = [fr"$\pi(\mathrm{{{paramLabels[ii]}}})$" for ii in range(M)]
    paramPDFLabels = [f"π({paramLabels[ii]})" for ii in range(M)]

    UQblue = display_util.colorOrder(1)[0]

    fig = []
    for ii in range(M):
        fig_ = make_subplots(rows=1, cols=2)
        # Get current X, per dimension.
        Xcurr = X[:,ii,:].squeeze()
        # Kernel smoothing density
        kde = stats.gaussian_kde(Xcurr.flatten())
        xi = np.linspace(Xcurr.min(), Xcurr.max(), 100)
        f = kde(xi)  

        # FIRST subplot for trace
        for jj in range(Xcurr.shape[1]):
            fig_.add_trace(go.Scatter(
                    x=np.arange(1,N+1),
                    y=Xcurr[:,jj],
                    mode='lines',
                    line_width=1,
                    line_color='rgb(179,179,179)',
                    showlegend=False),
                row=1, col=1)
        
        # fig_.update_xaxes(title=r'$\mathrm{Steps}$', row=1, col=1,mirror=True)
        fig_.update_xaxes(title='Steps', row=1, col=1,mirror=True)
        # fig_.update_yaxes(title=fr'$\mathrm{{{paramLabels[ii]}}}$', row=1, col=1,mirror=True)
        fig_.update_yaxes(title=f'{paramLabels[ii]}', row=1, col=1,mirror=True)
        
        # SECOND subplot for KDE
        fig_.add_trace(go.Scatter(
                x=f,
                y=xi,
                line_width=2,
                line_color=UQblue,
                showlegend=False),
            row=1,col=2)
        
        fig_.update_xaxes(title=paramPDFLabels[ii], row=1, col=2,mirror=True)
        fig_.update_yaxes(showticklabels=False, row=1, col=2,mirror=True)        

        # Appending images 
        fig.append(fig_)

    return fig
    

def violinplot(Y, X=None, FaceColor=None, name=None):

    # set color(s):
    plotColor = FaceColor if FaceColor else display_util.colorOrder(1)[0]

    # Verify inputs
    if Y is None:
        raise('Not enough arguments. Violin plot needs data to plot.')
    if isinstance(Y, list):
        Y = np.array(Y, ndmin=2)

    if X:
        if isinstance(X, list):
            X = np.array(X, ndmin=1)
        if Y.shape[1] != X.size:
            raise('Dimension mismatch. The number of columns in Y have to be the same as the length of X.')
    
    ## Create the violin plot
    # raise warning if only single point for kernel smoothing
    if Y.shape[0] == 1:
        warnings.warn('Only one point provided for violinplot, increase number of supplied points for accurate PDF estimate.')
    
    fig = go.Figure()
    for ii in range(Y.shape[1]):
        ## Create the violin plot
        # TODO: finish this code and test it with LinearRegression and PredPrey examples!!!
        trace = go.Violin(
            x=ii*np.ones(Y[:,ii].shape)+1, 
            y=Y[:,ii], 
            fillcolor=plotColor, 
            line_color=plotColor,
            line_width=0.5,
            points=False, 
            showlegend=False,
            opacity=.8,
            name=name,
            )
        fig.add_trace(trace)
        if ii==0:
            fig.update_traces(
                # selector={name=ii},
                showlegend=True,


            )        

    if Y.shape[1]<10:
        figureWidth = 800
    elif Y.shape[1]<40:
        figureWidth = Y.shape[1]*40
    else:
        figureWidth = 1600
    # figureWidth = Y.shape[1]*40 if 10<Y.shape[1]<40 else 1600

    fig.update_layout(
        autosize=False,
        width=figureWidth,
        showlegend=True,
    )
    # fig.show()
    return fig
    
def BinWidth_ScottsRule(Y):
    w = 3.49*np.std(Y.flatten(),ddof=1)*Y.size**(-1/3) # Width of a histogram element
    nBins = int(np.max([np.ceil(np.ptp(Y)/w),1])) # Number of bins
    return nBins


        
        

        




