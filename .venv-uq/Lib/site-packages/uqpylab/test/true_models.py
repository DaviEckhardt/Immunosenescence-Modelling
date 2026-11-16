import numpy as np

def XsinX(X):
    return X*np.sin(X)

def hat2d(X):
    # 2D hat function
    # see https://www.uqlab.com/reliability-2d-hat-function
    X = np.array(X,ndmin=2)
    t1 = np.square(X[:,0] - X[:,1])
    t2 = 8*np.power(X[:,0] + X[:,1] - 4, 3)
    return 20 - t1 - t2

def ishigami(X):
    # Ishigami function
    X = np.array(X,ndmin=2)
    a = 7
    b = 0.1
    T1 = np.sin(X[:,0])
    T2 = a* np.power(np.sin(X[:,1]),2)
    T3 = b*np.power(X[:,2],4)*np.sin(X[:,0])
    return  T1 + T2 + T3

def meanX(X):
    X = np.array(X,ndmin=2)
    return np.mean(X, axis=1)

def branin(X, P):

    # Check number of input arguments
    assert X.shape[1] == 2, 'only 2 input variables allowed'

    # Y = P(1)*(X(:,2) - P(2)*X(:,1).^2 + P(3)*X(:,1) - P(4)).^2 + P(5)*(1-P(6))*cos(X(:,1)) + P(5);
    term1 = P['a'] * (X[:, 1] - P['b'] * X[:, 0] ** 2 + P['c'] * X[:, 0] - P['r']) ** 2
    term2 = P['s'] * (1 - P['t']) * np.cos(X[:, 0]) + P['s']
    Y =  term1 + term2

    return Y

def ishigami_parametric(X,P):
    X = np.array(X,ndmin=2)
    T1 = np.sin(X[:,0])
    T2 = P['a']* np.power(np.sin(X[:,1]),2)
    T3 = P['b']*np.power(X[:,2],4)*np.sin(X[:,0])
    return  T1 + T2 + T3

def simply_supported_beam_9points(X):
    """calculates the deflection of a Simply Supported Beam on 9 equally-spaced 
    points along the beam length. X refers to a sample of the input random variables: [b h L E p]. 
    The points for which the deflection is calculated are xi = (1:9)/10*L.
    The vector Y contains the displacement at each of the 9 points.

    Parameters
    ----------
    X: ndarray
        5-column matrix
        x[:,0]: beam width (m)
        x[:,1]: beam height (m)
        x[:,2]: length (m)
        x[:,3]: Young modulus (Pa)
        x[:,4]: uniform load (N)
        
    
    Returns
    -------
    ndarray
        Y[:,0]: deflection at xi=1/10*L
        ...
        Y[:,8]: deflection at xi=9/10*L
    """
    X = np.array(X,ndmin=2)
    b = X[:, 0]; # beam width  (m)
    h = X[:, 1]; # beam height (m)
    L = X[:, 2]; # Length (m)
    E = X[:, 3]; # Young modulus (Pa)
    p = X[:, 4]; # uniform load (N)
    
    # The beam is considered primatic, therefore:
    I = b* np.power(h,3) / 12; # the moment of intertia
    Y = np.empty((X.shape[0],9))
    for j in np.arange(0,9):
        xi = (j+1)/10*L
        Y[:,j] = -p*xi*(np.power(L,3)-2*np.power(xi,2)*L + xi**3)/(24*E*I);
    
    return Y

def borehole(X):

    # Y = uq_borehole(X) returns the value of the water flow Y through a borehole, described by 8 variables given in X = [rw, r, Tu, Hu, Tl, Hl, L, Kw]

    # rw - radius of borehole (m)
    # r  - radius of influence (m)
    # Tu - transmissivity of upper aquifer (m^2/yr)
    # Hu - potentiometric head of upper aquifer (m)
    # Tl - transmissivity of lower aquifer (m^2/yr)
    # Hl - potentiometric head of lower aquifer (m)
    # L  - length of borehole (m)
    # Kw - hydraulic conductivity of borehole (m/yr)

    # For more info, see: http://www.sfu.ca/~ssurjano/borehole.html

    X = np.array(X,ndmin=2)
    
    rw = X[:, 0]
    r  = X[:, 1]
    Tu = X[:, 2]
    Hu = X[:, 3]
    Tl = X[:, 4]
    Hl = X[:, 5]
    L  = X[:, 6]
    Kw = X[:, 7]

    # Precalculate the logarithm:
    Logrrw = np.log(np.divide(r,rw))

    Numerator = 2*np.pi*Tu*(Hu - Hl)
    Denominator = Logrrw*(1 + np.divide((2*L*Tu),(Logrrw*(rw**2)*Kw)) + np.divide(Tu,Tl));

    return np.divide(Numerator, Denominator)

def shortcol(X):
    """returns the value of the limit state function Y of the
    strength of a short, square (5x15) steel column subjected to an axial
    load and a bending moment. The problem is described by 3 variables given
    in X = [Y, M, P].

    Parameters
    ----------
    X: ndarray
        3-column matrix
        x[:,0]: yield stress    (MPa)
        x[:,1]: bending moment  (Nmm)
        x[:,2]: axial force     (N)        
    
    Returns
    -------
    1darray
        Y: strength of a steel column

    For more info, see: http://www.sfu.ca/~ssurjano/shortcol.html
    """
    X = np.array(X,ndmin=2)
    Y = X[:, 0]; # yield stress    (MPa)
    M = X[:, 1]; # bending moment  (Nmm)
    P = X[:, 2]; # axial force     (N)
    
    # define cross-section width and depth
    b = 5
    h = 15

    # Calculate the single terms:
    term1 = -4*M / (b*(h**2)*Y)
    term2 = -(P**2) / ((b**2)*(h**2)*(Y**2))
    Y = 1 + term1 + term2
   
    return Y

def XsinX_multipleOutputModel(X):
    # expanded XsiX function so that it has more outputs (only for testing display functions)
    y = X*np.sin(X)
    Y = np.concatenate((y, y, y, y), axis=1)

    return Y

def hat2d_multipleOutputModel(X):
    # expanded 2D hat function  so that it has more outputs (only for testing display functions)
    X = np.array(X,ndmin=2)
    t1 = np.square(X[:,0] - X[:,1])
    t2 = 8*np.power(X[:,0] + X[:,1] - 4, 3)
    y = 20 - t1 - t2
    Y = np.vstack((y, y-300)).T

    return Y