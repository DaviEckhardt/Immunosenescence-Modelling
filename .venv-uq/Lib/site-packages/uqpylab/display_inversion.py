""" Plotting utility imitating the uq_display_uq_inversion functionality of UQLab """
from uqpylab import sessions
import uqpylab.display_util as display_util
import uqpylab.display_general as display_general
import uqpylab.helpers as helpers
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from scipy.stats import norm
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import copy


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

# Axes
pio.templates[pio.templates.default].layout.xaxis.exponentformat = 'e'
pio.templates[pio.templates.default].layout.yaxis.exponentformat = 'e'

def inversion(obj, theInterface, outidx=None, **kwargs):
    # check for input arguments
    if outidx is None:
        outidx = [1]
    if type(outidx)==int:
        outidx = [outidx]
    outidx = [i-1 for i in outidx]

    method = obj['Internal']['Solver']['Type'].lower()

    if method == 'mcmc':
        return inversion_mcmc(obj, theInterface, outidx, **kwargs)
    elif method in 'sle':
        # return inversion_sle(obj, outidx)
        print("SLE display not implemented yet.")
    elif method == 'ssle':
        # return inversion_ssle(obj, outidx)
        print("SSLE display not implemented yet.")
    else:
        print("Select from 'mcmc', 'sle', or 'ssle' method!")

def inversion_mcmc(obj, theInterface, outidx, acceptance=None, scatterplot=None, predDist=None, meanConvergence=None, trace=None, **kwargs):
    '''
        Name               VALUE
        'acceptance'       Plots the acceptance rate per chain 
                            - Boolean
                            default : False
        'scatterplot'      Plots a multi dimensional parameter scatter plot
                            of the parameters in the Results key-value pair of MODULE 
                            - Integer or 'all'
                            default : 'all'
        'predDist'         Plots the predictive distributions if available
                            from post processing (uq.postProcessInversionMCMC)
                            - Boolean
                            default : True (if available) 
        'meanConvergence'  Plots the convergence of the marginal mean for
                            the specified parameter averaged over all chains 
                            - Integer or 'all'
                            default : False
        'trace'            Plots the trace plot of the marginal sample
                            points for the specified parameter 
                            - Integer or 'all'
                            default : 
                            False
    '''

    if 'Type' in obj and obj['Type'] != 'uq_inversion':
        print('uq_display_uq_inversion only operates on objects of type ''Inversion''') 
    if ('Type' in obj['Internal']['Solver'] and obj['Internal']['Solver']['Type'] != 'MCMC') or not obj['Results']:
        raise('Only works on MCMC-based results')
    
    # switch if custom likelihood
    CUSTOM_LIKELI = 'customLikeli' in obj['Internal'] and bool(obj['Internal']['customLikeli'])

    ## ----------------------------------------------------- ##
    ## INITIALIZE
    # Check which post processed arrays are available
    if 'PostProc' in obj['Results']:
        priorPred_avail = 'PriorPredSample' in obj['Results']['PostProc']
        postPred_avail = 'PostPredSample' in obj['Results']['PostProc']
        procPostSample_avail = 'PostSample' in obj['Results']['PostProc']
        # determine sample size
        if procPostSample_avail:
            PriorDist = theInterface.extractFromAnalysis(parentName=obj['Name'], objPath='PriorDist')
            if isinstance(PriorDist['Marginals'], dict):
                nDim = 1
                nIter,nChains = np.array(obj['Results']['PostProc']['PostSample']).shape
            else:
                nIter,nDim,nChains = np.array(obj['Results']['PostProc']['PostSample']).shape
        else:
            if isinstance(PriorDist['Marginals'], dict):
                nDim = 1
                nIter,nChains = np.array(obj['Results']['Sample']).shape
            else:
                nIter,nDim,nChains = np.array(obj['Results']['Sample']).shape
        procPriorSample_avail = 'PriorSample' in obj['Results']['PostProc']
        pointEstimate = obj['Results']['PostProc'].get('PointEstimate')
        pointEstimate_avail = bool(pointEstimate)
        pointEstimatePred = pointEstimate.get('ForwardRun') if pointEstimate else None
        pointEstimatePred_avail = bool(pointEstimatePred) if pointEstimate else False
    else:
        # nothing is available
        priorPred_avail = postPred_avail = procPostSample_avail = pointEstimate_avail = procPriorSample_avail = False
        nIter,nDim,nChains = np.array(obj['Results']['Sample']).shape # determine sample size

    # get labels for parameters
    paramLabels = []
    FullPrior_Input = theInterface.extractFromAnalysis(parentName=obj['Name'], objPath = "Internal.FullPrior")
    if isinstance(FullPrior_Input['Marginals'],list):
        for ii in range(len(FullPrior_Input['Marginals'])):
            paramLabels.append(FullPrior_Input['Marginals'][ii]['Name'])
    elif isinstance(FullPrior_Input['Marginals'],dict):
         paramLabels = FullPrior_Input['Marginals']['Name']
    else:
        raise("Unexpected format of FullPrior_Input.")

    ## Default behavior
    Default = {}
    Default['plotAcceptance_flag'] = False # acceptance ratio
    Default['scatterplotParams'] = list(range(1,nDim+1)) # scatterplot, all
    # Predictive distributions (True if predictive samples are available plot as well)
    Default['plotPredDist_flag'] = priorPred_avail or postPred_avail
    Default['plotMeanConvergence_flag'] = False # mean convergence
    Default['plotTrace_flag'] = False # trace plots

    if acceptance==scatterplot==predDist==meanConvergence==trace==None:
        acceptance = Default['plotAcceptance_flag']
        scatterplot = Default['scatterplotParams']
        predDist = Default['plotPredDist_flag']
        meanConvergence = Default['plotMeanConvergence_flag']
        trace = Default['plotTrace_flag']

    ## ----------------------------------------------------- ##
    ## Check for input arguments
    # Set optional arguments - in the function input arguments already

    # 'acceptance' plots the acceptance rate for each chain
    if acceptance is None:
        acceptance = False
    plotAcceptance_flag = acceptance

    # 'scatterplot' plots an mDim parameter scatter
    plotScatterplot_flag = True
    if scatterplot == 'all':
        scatterplotParams = list(range(nDim))
    elif isinstance(scatterplot, int):
        scatterplotParams = [scatterplot-1]        
    elif (isinstance(scatterplot,list) and all([isinstance(ii,int) for ii in scatterplot])) or (isinstance(scatterplot,int)):
        scatterplotParams = [scatterplot[ii]-1 for ii in range(len(scatterplot))]
    elif scatterplot is None:
        plotScatterplot_flag = False
    else:
        raise('Wrong value found in scatterplot key-value pair')
    
    # 'predDist' plots the prior and posterior predictive distributions (if available)
    if predDist:
        if not (postPred_avail or priorPred_avail):
            # neither prior nor posterior predictive is available
            raise('Need to provide prior or posterior predictive model evaluations. See uq.postProcessInversionMCMC().')
        if CUSTOM_LIKELI:
            raise('Predictive distributions are not supported with user-specified likelihood functions.')
        plotPredDist_flag = priorPred_avail or postPred_avail
    else:
        plotPredDist_flag = False

    # 'meanConvergence' plots the convergence of all chains
    plotMeanConvergence_flag = True
    if not meanConvergence:
        plotMeanConvergence_flag = False
    elif isinstance(meanConvergence, int):
        plotMeanConvergenceIndex = [meanConvergence-1]        
    elif (isinstance(meanConvergence,list) and all([isinstance(ii,int) for ii in meanConvergence])) or isinstance(meanConvergence,int):
        plotMeanConvergenceIndex = [meanConvergence[ii]-1 for ii in range(len(meanConvergence))]
    elif meanConvergence == 'all':
        plotMeanConvergenceIndex = list(range(nDim))
    else:
        raise('Wrong value found in meanConvergence key-value pair')
    
    # 'plotTrace' plots the chain trace plots
    plotTrace_flag = True
    if not trace:
        plotTrace_flag = False
    elif isinstance(trace, int):
        plotTraceIndex = [trace-1]
    elif (isinstance(trace,list) and all([isinstance(ii,int) for ii in trace])) or isinstance(trace,int):
        plotTraceIndex = [trace[ii]-1 for ii in range(len(trace))]
    elif trace == 'all':
        plotTraceIndex = list(range(nDim))
    else:
        raise('Wrong value found in trace key-value pair')
    
    ## ----------------------------------------------------- ##
    ## Create the plots
    figs = []

    ## ----------------------------------------------------- ##   
    ## Plot the acceptance rate for each chain
    if plotAcceptance_flag:
        # Retrieve acceptance rates
        Acceptance = obj['Results']['Acceptance']

        # plot colors 
        plotColor = display_util.colorOrder(2)
        UQBlue = plotColor[0]
        UQOrange =  plotColor[1]

        # retrieve bad chains
        goodChains = list(range(nChains))
        badChainsFlag = False 
        data = []
        nChainsTotal = nChains

        if 'PostProc' in obj['Results'] and 'ChainsQuality' in obj['Results']['PostProc']:
            badChainsFlag = True
            badChains = obj['Results']['PostProc']['ChainsQuality']['BadChains']
            goodChains = obj['Results']['PostProc']['ChainsQuality']['GoodChains']
            nChainsTotal = len(badChains)+len(goodChains)

            # Bad chains - trace 0
            trace0 = go.Scatter(
                x = badChains,
                y = [Acceptance[ii-1] for ii in badChains],
                name='bad chains',
                mode = 'markers',
                marker = dict(
                    size = 7,
                    line_color = UQOrange, 
                    symbol = 'x-thin', 
                    line = dict(width = 2) 
                )
            )
            data.append(trace0)

        # Good chains - trace 1
        trace1 = go.Scatter(                
            x = goodChains,
            y = [Acceptance[ii-1] for ii in goodChains],
            name='good chains',
            mode = 'markers',
            marker = dict(
                size = 7,
                line_color = UQBlue, 
                symbol = 'x-thin', 
                line = dict(width = 2) 
            )
        )
        data.append(trace1)

        # Layout specs
        layout = go.Layout(
            title='Acceptance Rate per Chain',
            xaxis_title='Chain',
            yaxis_title='Acceptance Rate',
            xaxis_range=[0,nChainsTotal],
            yaxis_range=[0,1],
        )

        # Figure
        fig = go.Figure(data=data,layout=layout)
        figs.append(fig)

    ## ----------------------------------------------------- ##   
    ## Plot the m-dimensional parameter scatterplot
    ## ----------------------------------------------------- ##   

    if plotScatterplot_flag:
        # Define custom colors
        priorColor = 'rgb(128, 179, 255)'
        postColor = 'rgb(0, 51, 153)'

        # Number of maximum plot points
        NMaxPlot = 10_000

        # Update scatterplot labels
        paramLabelsScatter = [paramLabels[ii] for ii in scatterplotParams]

        ## ----------------------------------------------------- ##   
        # If *prior sample* is available, create the prior plot
        if procPriorSample_avail:
            if np.array(obj['Results']['PostProc']['PriorSample']).ndim == 1:
                PostProcPriorSample = np.array(obj['Results']['PostProc']['PriorSample'],ndmin=2).T
            else:
                PostProcPriorSample = np.array(obj['Results']['PostProc']['PriorSample'],ndmin=2)
            
            if PostProcPriorSample.ndim == 2:
                new_shape = (1,0)
            elif PostProcPriorSample.ndim == 3:
                new_shape = (1,0,2)
            else:
                raise("Unexpected data shape: PriorSample should have 3 dims at maximum.")
            
            scatterplotPrior_Sample = np.reshape(np.transpose(PostProcPriorSample,new_shape),(nDim,-1),order='F').T   

            # Get relevant subset of Sample
            if NMaxPlot < scatterplotPrior_Sample.shape[0]:
                PlotId = np.random.permutation(np.arange(scatterplotPrior_Sample.shape[0]))[:NMaxPlot]
            else:
                PlotId = np.arange(scatterplotPrior_Sample.shape[0])
            PriorSample = scatterplotPrior_Sample[np.ix_(PlotId,scatterplotParams)]

            # data = pd.DataFrame(PriorSample, columns=paramLabelsScatter)
            Limits = np.array([np.amin(PriorSample,axis=0),np.amax(PriorSample,axis=0)])
            fig=display_general.scatterDensity(PriorSample, paramLabelsScatter, Color=priorColor,Title='Prior Sample',Limits=Limits)
            figs.append(fig)

        ## ----------------------------------------------------- ##   
        # Check for posterior sample
        if procPostSample_avail:
            if np.array(obj['Results']['PostProc']['PostSample']).ndim == 1:
                PostProcPostSample = np.array(obj['Results']['PostProc']['PostSample'],ndmin=2).T
            else:
                PostProcPostSample = np.array(obj['Results']['PostProc']['PostSample'],ndmin=2)

            if PostProcPostSample.ndim == 2:
                new_shape = (1,0)
            elif PostProcPostSample.ndim == 3:
                new_shape = (1,0,2)  #(2,1,0)
            else:
                raise("Unexpected data shape: PostSample should have 3 dims at maximum.")
            scatterplotPost_Sample = np.reshape(np.transpose(PostProcPostSample,new_shape),(nDim,-1),order='F').T
        else:
            Sample = np.array(obj['Results']['Sample'])
            if Sample.ndim == 1:
                scatterplotPost_Sample = Sample
            elif Sample.ndim == 2:
                new_shape = (1,0)
                scatterplotPost_Sample = np.reshape(np.transpose(Sample,new_shape),(nDim,-1),order='F').T
            elif Sample.ndim == 3:
                new_shape = (1,0,2)
                scatterplotPost_Sample = np.reshape(np.transpose(Sample,new_shape),(nDim,-1),order='F').T
            else:
                raise("Unexpected data shape: Sample should have 3 dims at maximum.")    
            
        
        # Get relevant subset (requested dimension) of the posterior sample
        if NMaxPlot < scatterplotPost_Sample.shape[0]:
            PlotId = np.random.permutation(np.arange(scatterplotPost_Sample.shape[0]))[:NMaxPlot]
        else:
            PlotId = np.arange(scatterplotPost_Sample.shape[0])
        PostSample = scatterplotPost_Sample[np.ix_(PlotId,scatterplotParams)]

        # data = pd.DataFrame(PostSample, columns=paramLabelsScatter)

        if pointEstimate_avail:
            # Extract relevant point estimates:
            plotPointEstimates = {}
            plotPointEstimates['X'] = []
            plotPointEstimates['Type'] = pointEstimate['Type']
            plotPointCollection = []

            # The general form of pointEstimate['X'] is a multidimensional nested list - it has 3 dimensions at maximum. However, during encoding, the dimensions are lost. Therefore, we need to take care of all the cases. 
            nType = len(pointEstimate['Type']) if isinstance(pointEstimate['Type'],list) else 1

            # if pointEstimate['X'] is a simple list - one point
            if not isinstance(pointEstimate['X'], list):
                pointEstimate['X'] = [[[pointEstimate['X']]]]
            elif not any(isinstance(item, list) for item in pointEstimate['X']):
                pointEstimate['X'] = [[pointEstimate['X']]]
            # if pointEstimate['X'] is a nested list, it is most likely a custom case of set of points or nType points
            elif not helpers.hasNDimensionalStructure(pointEstimate['X'], 3):
                if nType == 1: # custom case
                    pointEstimate['X'] = [pointEstimate['X']]
                else: # nType of specific points
                    pointEstimate['X'] = [[item] for item in pointEstimate['X']]
            # if pointEstimate['X'] is a nested list of nType elements where an element is a nested list with custom points or a point (such as mean), it's a mixed general type and there is no need expand the list with dimensions
            # else: # do nothing

            # reorganize pointEstimate['X'] points with scatterplotParams order
            plotPointEstimates['X'] = []
            plotPointCollection = []
            for pp in range(nType):
                temp = np.array(pointEstimate['X'][pp],ndmin=2) # make an array from each set of points
                temp = temp[np.ix_(range(temp.shape[0]),scatterplotParams)] # rearange it with scatterplotParams
                temp = temp.tolist() # make it list again
                plotPointEstimates['X'].append(temp)
                for item in plotPointEstimates['X'][pp]:
                    plotPointCollection.append(item)

            # compute plot limits
            allPlotPoints = np.vstack((PriorSample, np.array(plotPointCollection)))
            plotLimits = np.vstack((np.amin(allPlotPoints,axis=0), np.amax(allPlotPoints,axis=0)))

            fig=display_general.scatterDensity(PostSample, paramLabelsScatter, Color=postColor, Limits=plotLimits,Title='Posterior Sample',Points=plotPointEstimates)
            figs.append(fig)
        else:
            fig=display_general.scatterDensity(data, paramLabelsScatter, Color=postColor, Limits=[np.amin(PriorSample), np.amax(PriorSample)], Title='Posterior Sample')
            figs.append(fig)      

    if plotPredDist_flag:
        # Plot samples from the prior and posterior predictive distribution (if available)
        # check which model evaluations are available  

        nDataGroups = obj['Internal']['nDataGroups']
        # Process the data
        if priorPred_avail:
            PriorPredSample = obj['Results']['PostProc']['PriorPredSample']
            if nDataGroups == 1:
                PriorPredSample = [PriorPredSample]

        if postPred_avail:
            PostPredSample = obj['Results']['PostProc']['PostPredSample']
            if nDataGroups == 1:
                PostPredSample = [PostPredSample]
        
        for ii in range(nDataGroups):
            Sample = {}
            if priorPred_avail:
                # store prior predictive
                Sample['PriorPred'] =  PriorPredSample[ii]['Sample']
            if postPred_avail:
                # store posterior predictive
                Sample['PostPred'] = PostPredSample[ii]['Sample']

            if nDataGroups == 1:
                DataCurr = obj['Data']
            else:
                DataCurr = obj['Data'][ii]

            MOMap = np.array(DataCurr['MOMap'],ndmin=2)
            if MOMap.shape[0] == 1:
                MOMap = MOMap.T
            # if np.array(DataCurr['y'],ndmin=2).shape[0] == 1:
            if MOMap.shape[1] == 1:
                # histogram for scalar model outputs
                if pointEstimatePred_avail:
                    if nDataGroups > 1 and isinstance(pointEstimatePred,list):
                        fig = display_general.plotSinglePred(Sample, DataCurr, pointEstimatePred[ii])
                    else:
                        fig = display_general.plotSinglePred(Sample, DataCurr, pointEstimatePred)
                else:
                    fig = display_general.plotSinglePred(Sample, DataCurr)
                # change label to \cg if multiple data groups
                if nDataGroups > 1:
                    # fig.update_layout(xaxis_title = r'$\mathcal{G}^{{({})}}$'.format(ii))
                    fig.update_layout(yaxis_title = f'G<sup>({ii+1})')
            else: 
                # line plots for vectorized model outputs
                if pointEstimatePred_avail:
                    if nDataGroups > 1 and isinstance(pointEstimatePred, list):
                        fig = display_general.plotSeriesPred(Sample, DataCurr, pointEstimatePred[ii])
                    else:
                        fig = display_general.plotSeriesPred(Sample, DataCurr, pointEstimatePred)
                else:
                    fig = display_general.plotSeriesPred(Sample, DataCurr)
                # change label to \cg if multiple data groups
                if nDataGroups > 1:
                    # fig.update_layout(xaxis_title = r'$\mathcal{G}^{{({})}}$'.format(ii))     
                    fig.update_layout(yaxis_title = f'G<sup>({ii+1})')     
                fig.update_xaxes(range=[0, np.array(DataCurr['y'],ndmin=2).shape[1]+1])
            fig.update_layout(title_text=DataCurr['Name'], title_font_size=22)
            figs.append(fig)

    if plotMeanConvergence_flag:
        # check for posterior sample      
        if procPostSample_avail:
            meanConvergence_Sample = np.array(obj['Results']['PostProc']['PostSample'])
        else:
            meanConvergence_Sample = np.array(obj['Results']['Sample'])
        
        plotIndex = plotMeanConvergenceIndex

        # loop over plot indices
        for ii in plotIndex:
            # plot only a certain amount of steps
            Nplotsteps = np.min(np.array([1000,nIter]))
            plotSteps = np.int_(np.unique(np.floor(np.linspace(1,nIter,Nplotsteps))))

            # compute means
            meanVals = np.zeros((Nplotsteps, nChains))
            for jj in range(plotSteps.size):
                currPlotStep = int(plotSteps[jj])
                # loop over chains
                for kk in range(nChains):
                    meanVals[jj,kk] = np.mean(meanConvergence_Sample[:currPlotStep,ii,kk])
            
            # combine chains
            meanComb = np.mean(meanVals,1)

            fig = go.Figure(go.Scatter(
                    x=plotSteps,
                    y=meanComb,
            ))

            fig.update_layout(
                # xaxis_title=r'Step',
                xaxis_title='Step',
                # yaxis_title=r'$E[' + paramLabels[ii] + r']$',
                yaxis_title=f'E[{paramLabels[ii]}]',
                title='Mean convergence'
            )
            figs.append(fig)

    if plotTrace_flag:
        # update traceplot labels
        paramLabelsScatter = [paramLabels[ii] for ii in plotTraceIndex]

        # get relevant sample
        traceSamples = np.array(obj['Results']['Sample'],ndmin=3)[:,plotTraceIndex,:]

        # call traceplot function
        fig = display_general.traceplot(traceSamples, labels=paramLabelsScatter)
        figs += fig
        
    return figs

# def inversion_sle(obj, outidx):
#     pass

# def inversion_ssle(obj, outidx):
#     pass


