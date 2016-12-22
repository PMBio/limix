from mvSet import MvSetTest
from mvSetFull import MvSetTestFull
from mvSetInc import MvSetTestInc

def fit_iSet(Y=None, Xr=None, F=None, Rr=None, factr=1e7, Rg=None, Ug=None, Sg=None, Ie=None, n_nulls=10):
    # data
    noneNone = Sg is not None and Ug is not None
    bgRE = Rg is not None or noneNone
    # fixed effect
    msg = 'The current implementation of the full rank iSet'
    msg+= ' does not support covariates.'
    msg+= ' We reccommend to regress out covariates and'
    msg+= ' subsequently quantile normalize the phenotypes'
    msg+= ' to a normal distribution prior to use mtSet/iSet.'
    msg+= ' This can be done within the LIMIX framework using'
    msg+= ' the methods limix.utils.preprocess.regressOut and'
    msg+= ' limix.utils.preprocess.gaussianize'
    assert not (F is not None and bgRE), msg
    # strat
    strat = Ie is not None
    msg = 'iSet for interaction analysis of stratified populations ' 
    msg+= 'using contextual variables does not support random effect '
    msg+= 'correction for confounding. '
    msg+= 'Please use the fixed effects to correct for confounding. '
    assert not (strat and bgRE), msg

    #define mtSet
    if bgRE:        mvset = MvSetTest(Y=Y,Xr=Xr,F=F,Rr=Rr,factr=factr)
    elif strat:     mvset = MvSetTestFull(Y=Y,Xr=Xr,Rg=Rg,Ug=Ug,Sg=Sg,factr=factr)
    else:           mvset = MvSetTestInc(Y=Y,Xr=Xr,F=F,Ie=Ie,factr=factr)

    RV = {} 
    RV['mtSet LLR'] = mvset.assoc()
    RV['iSet LLR'] = mvset.gxe()
    RV['iSet-het LLR'] = mvset.gxehet()
    pdb.set_trace()

    RV0 = {}
    RV0['mtSet LLR0'] = mvset.assoc_null(n_nulls=n_nulls)
    RV0['iSet LLR0'] = mvset.gxe_null(n_nulls=n_nulls)
    RV0['iSet-het LLR0'] = mvset.gxehet_null(n_nulls=n_nulls)


    columns = ['mtSet LLR', 'iSet LLR', 'iSet-het LLR',
                'Persistent Var', 'Rescaling-GxC Var', 'Heterogeneity-GxC var', 'Converged']
