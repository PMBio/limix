#IGE (Indirect genetic effects) = SGE (Social genetic effects). IEE = SEE.

#the potential presence of NAs in input phenotype, covs, cages etc means that in this code we introduce two sets of animals (not discussed in the paper):
#focal animals, defined as having phenotype, covs (if covs are provided), cage and kinship
#all animals = focal animals + cage mates, where cage mates are defined as having cage and kinship and being in subset_IDs, and are referenced to using _cm in this code. Note that in the paper cage mate has a more precise meaning, namely "the other mice in the cage of the focal individual".

import sys
import warnings
import scipy as sp
import scipy.linalg as la
from limix.core.mean.mean_base import MeanBase as lin_mean
from limix.core.covar.dirIndirCov import DirIndirCov
from limix.core.covar.fixed import FixedCov 
from limix.core.covar.combinators import SumCov
from limix.utils.preprocess import covar_rescaling_factor
from limix.utils.preprocess import covar_rescale
from limix.core.gp.gp_base import GP
import pdb

class DirIndirVD():

    #many of inputs below are passed from SocialData object built using social_data.py
    #pheno: missing values should be encoded as -999 (needs to be numeric for HDF5). pheno can be 1xN or Nx1 vector (where N is number of individuals)
    #pheno_ID: IDs corresponding to pheno.
    #covs: missing values should be encoded as -999.  covs can be 1xN or Nx1 vector, or a NxK matrix (K number of covariates)
    #covs_ID: IDs corresponding to covs
    #kinship_cm: GRM for all mice ever considered (focal or cagemate). !!!contrary to what notation says here, this is all mice = focal + cage mates, not only cm. no missing values allowed. symmetric matrix.
    #kinship_cm_ID: rownames and colnames of kinship_cm
    #cage_cm: cages of all mice ever considered (focal or cagemate). missing values should be encoded as NA. NA cage will lead to this animal to be ignored from the analysis. if you have reason to believe this animal was a cage mate (but you don't know its cage), this is an issue and you need to be aware of it.
    #cage_cm_ID: IDs corresponding to cage_cm
    #independent_covs: if True (default) LIMIX will check whether any covariate is a linear combination of the others and if so fix the issue by updating covs to independent columns
    #DGE: should DGE be included in the model?
    #IGE: should IGE be included in the model?
    #IEE: should IEE be included in the model? note that if IGE = True, IEE will be set to True by the program (silently) - see below
    #cageEffect: should cage effects be included in the model?
    #calc_ste: should the standard errors of the variance components be estimated and output? 
    #standardize_pheno: should the phenotype be standardized to variance 1?
    #subset_IDs: subset of IDs to consider *as focal individuals or cage mate*. this means that any animal not in subset_IDs will be ignored completely from analysis. correct only if all animals from a cage are excluded! if instead want to exclude animals only as focal individuals, set their phenotype to NA.
    def __init__(self, pheno = None, pheno_ID = None, covs = None, covs_ID = None, kinship_cm = None, kinship_cm_ID = None, cage_cm = None, cage_cm_ID = None, independent_covs = True, DGE = True, IGE = False, IEE = False, cageEffect = False, calc_ste = False, standardize_pheno = True, subset_IDs = None):
 
        self.parseNmore(pheno, pheno_ID, covs, covs_ID, kinship_cm, kinship_cm_ID, cage_cm, cage_cm_ID, independent_covs,standardize_pheno, subset_IDs)

        # the purpose of the IEE argument is to specify whether IEE should be off when IGE are off. When IGE are on, IEE must be on and therefore IEE will be automatically set to True when IGE is True.
        if IGE:
            IEE = True
        #VD defines the genetic, environmental and cage covariance matrices
        self.VD(DGE = DGE, IGE = IGE, IEE = IEE, cageEffect = cageEffect)
        #optimize estimates the variance components
        self.optimize()
        # get_VCs gets the variance components, standard errors, etc. for output
        # self.output useful to retrieve output using getOutput without having to specify calc_ste, DGE, IGE, etc.
        self.output = self.get_VCs (calc_ste, DGE = DGE, IGE = IGE, IEE = IEE, cageEffect = cageEffect)
            
    def parseNmore(self, pheno, pheno_ID, covs, covs_ID, kinship_cm, kinship_cm_ID, cage_cm, cage_cm_ID, independent_covs,standardize_pheno, subset_IDs):
        """match various inputs"""

        assert pheno is not None, 'Specify pheno!'
        assert pheno_ID is not None, 'Specify pheno IDs!'
        assert pheno.shape[0] == len(pheno_ID), 'Lengths of pheno and pheno IDs do not match!'
        assert kinship_cm is not None, 'Specify kinship!'
        assert kinship_cm_ID is not None, 'Specify kinship IDs!'
        assert kinship_cm.shape[0] == kinship_cm.shape[1], 'Kinship is not a square matrix!'
        assert kinship_cm.shape[0] == len(kinship_cm_ID), 'Dimension of kinship and length of kinship IDs do not match!'
        assert cage_cm is not None, 'No social analysis possible if cage information is unavailable. For DGE analysis, use regular LIMIX https://github.com/PMBio/limix'

####hack to shorten runtime
#        uCage=sp.unique(cage_cm)
#        remove=uCage[0:510]
#        idx_remove = sp.array(sp.concatenate([sp.where(cage_cm==remove[i])[0] for i in range(len(remove))]))
#        cage_cm[idx_remove]='NA'
####hack to shorten runtime

        #1. define set of animals with cage and kinship information (_cm)
        #1.1 _cm animals need to be in subset_IDs
        if subset_IDs is not None:
            Imatch = sp.nonzero(subset_IDs[:,sp.newaxis]==kinship_cm_ID)
            kinship_cm = kinship_cm[Imatch[1],:][:,Imatch[1]]
            kinship_cm_ID=kinship_cm_ID[Imatch[1]]

        #1.2 NA allowed in cage information so first of all exclude missing cage data and corresponding animals
        has_cage = (cage_cm!='NA')
        if sum(has_cage)==0:
            cage_cm = None
            assert cage_cm is not None, 'No social analysis possible if cage information is unavailable. For DGE analysis, use regular LIMIX https://github.com/PMBio/limix'
        cage_cm=cage_cm[has_cage]
        cage_cm_ID=cage_cm_ID[has_cage]

        #1.3 match cages and kinship as we need both for any social analysis. note that if this is a non social analysis and cage is specified (cage_cm not None), then only those individuals with known cage will be included in the analysis. This is useful to keep same sample between social and non social analyses.
        Imatch = sp.nonzero(cage_cm_ID[:,sp.newaxis]==kinship_cm_ID)
        cage_cm_ID = cage_cm_ID[Imatch[0]]
        cage_cm = cage_cm[Imatch[0]]
        kinship_cm = kinship_cm[Imatch[1],:][:,Imatch[1]]
        kinship_cm_ID=kinship_cm_ID[Imatch[1]]
        #(kinship_cm_ID==cage_cm_ID).all()
        #True
        #cage and kinship now have no missing values and are matched - IDs are in cage_ID and kinship_cm_ID
        # put IDs in sampleID_cm
        sampleID_cm = kinship_cm_ID
        assert len(sampleID_cm)!=0, 'No _cm animals'

        #2. define focal animals now: those with non missing phenotype and non missing covs, kinship and cage, and in subset_IDs
        #2.1 remove NAs from pheno
        if len(pheno.shape)==1:
            pheno = pheno[:,sp.newaxis]
        has_pheno = (pheno!=(-999))[:,0]
        pheno=pheno[has_pheno,:]
        pheno_ID=pheno_ID[has_pheno]
        #2.2 add intercept to covs
        #if no covs are provided, make it a vector of 1s for intercept
        if covs is None:
            covs = sp.ones((pheno.shape[0],1))
            covs_ID = pheno_ID
        #if covs are provided, append a vector of 1s for intercept
        else:
            new_col=sp.ones([covs.shape[0],1])
            if len(covs.shape)==1:
                covs = covs[:,sp.newaxis]
            covs=sp.append(new_col,covs,1)
        #2.3 remove NAs from covs
        has_covs = (covs!=(-999)).all(1)
        covs=covs[has_covs,:]
        covs_ID=covs_ID[has_covs]
        #2.4 match pheno and covs
        Imatch = sp.nonzero(pheno_ID[:,sp.newaxis]==covs_ID)
        pheno = pheno[Imatch[0],:]
        pheno_ID=pheno_ID[Imatch[0]]
        covs = covs[Imatch[1],:]
        covs_ID=covs_ID[Imatch[1]]
        #(pheno_ID==covs_ID).all()
        #True
        #pheno and covs now have no missing values and are matched - IDs are in pheno_ID and covs_ID
        #2.5 check which of those are in sampleID_cm (and thus have kinship and cage)
        has_geno = sp.array([pheno_ID[i] in sampleID_cm for i in range(pheno_ID.shape[0])])
        pheno = pheno[has_geno,:]
        covs = covs[has_geno,:]
        #create sampleID that has focal individuals.
        sampleID=pheno_ID[has_geno]
        assert len(sampleID)!=0, 'No focal animals'
        #remember sampleID_cm and sampleID are in different order (and of different length possibly)

        #3. create cage and kinship for focal animals
        idxs = sp.array([sp.where(sampleID_cm==sampleID[i])[0][0] for i in range(sampleID.shape[0])])
        cage=cage_cm[idxs]
        if len(cage.shape)==1:
            cage = cage[:,sp.newaxis]
        kinship=kinship_cm[idxs,:][:,idxs]
        
        #4. create focal x _cm genetic cross-covariance
        kinship_cross = kinship_cm[idxs,:]
        #so sampleID along rows and sampleID_cm along colummns

        #5. now create environmental matrices
        env = sp.eye(kinship.shape[0])
        env_cm = sp.eye(kinship_cm.shape[0])
        env_cross = env_cm[idxs,:]


        # standardize_pheno should be True for CV if want to output 1-mse
        if standardize_pheno:
            pheno -= pheno.mean(0)
            pheno /= pheno.std(0)
            print('Pheno has been standardized')

        if independent_covs:
            tol = 1e-6
            R = la.qr(covs,mode='r')[0][:covs.shape[1],:]
            I = (abs(R.diagonal())>tol)
            if sp.any(~I):
                warnings.warn('Covariate cols '+str(sp.where(~I)[0])+' have been removed because linearly dependent on the others')
            covs = covs[:,I]

        self.sampleID=sampleID
        self.pheno=pheno
        self.covs=covs
        self.cage=cage
        self.kinship=kinship
        self.env=env
        self.sampleID_cm=sampleID_cm
        self.cage_cm=cage_cm
        self.kinship_cm=kinship_cm
        self.env_cm=env_cm
        self.kinship_cross=kinship_cross
        self.env_cross=env_cross

    def VD(self, DGE, IGE, IEE, cageEffect):
        """ defines covariance for variance decomposition."""

        #defines mean
        mean = lin_mean(self.pheno,self.covs)

        #define cagemate assignment - required for SGE, SEE, and cage effects. Z is N focal x N_cm and has 0s in cells Z_i,i (i.e. an animal is not its own cage mate)
        same_cage = 1. * (self.cage==self.cage_cm)
        diff_inds = 1. * (self.sampleID[:,sp.newaxis]!=self.sampleID_cm)
        Z = same_cage * diff_inds

        #define the overall genetic covariance matrix
        if DGE or IGE:
            #scales kinship (DGE component) to sample variance 1
            sf_K = covar_rescaling_factor(self.kinship)
            self.kinship *= sf_K

            #now create and scale SGE and DGE/SGE covariance components
            if IGE:
                #first SGE component: ZKcmZ' in this code (ZKZ' in paper)
                _ZKcmZ = sp.dot(Z,sp.dot(self.kinship_cm,Z.T))
                sf_ZKcmZ = covar_rescaling_factor(_ZKcmZ)
                self.kinship_cm *= sf_ZKcmZ
                 #second DGE/SGE covariance:
                self.kinship_cross *= sp.sqrt(sf_K * sf_ZKcmZ)
        
        if DGE and not IGE:
            self._genoCov = FixedCov(self.kinship)
        elif IGE and not DGE:
            self._genoCov = FixedCov(_ZKcmZ)
        elif DGE and IGE:
            self._genoCov = DirIndirCov(self.kinship,Z,kinship_cm=self.kinship_cm,kinship_cross=self.kinship_cross)
        else:
            self._genoCov = None


        #define the overall environmental covariance matrix
        #there is always DEE
        #env naturally has sample variance 1 so no need to scale it
        if IEE:
            #_ZZ = ZIcmZ'
            _ZZ  = sp.dot(Z,Z.T)
            sf_ZZ = covar_rescaling_factor(_ZZ)
            self.env_cm *= sf_ZZ
            self.env_cross *= sp.sqrt(1 * sf_ZZ)

            self._envCov = DirIndirCov(self.env,Z,kinship_cm=self.env_cm,kinship_cross=self.env_cross)
        else:
            self._envCov = FixedCov(self.env)

        ##define cage effect covariance matrix
        if cageEffect:
            N = self.pheno.shape[0]
            uCage = sp.unique(self.cage)
            #W, the cage design matrix, is N x n_cages (where N is number of focal animals) 
            W = sp.zeros((N,uCage.shape[0]))
            for cv_i, cv in enumerate(uCage):
                W[:,cv_i] = 1.*(self.cage[:,0]==cv)
            #WW, the cage effect covariance matrix, is N x N and has 1s in cells WW_i,i
            WW = sp.dot(W,W.T)
            #this is equivalent to getting covar_rescaling_factor first and then multiplying, as done for other matrices above
            WW = covar_rescale(WW)
            self._cageCov = FixedCov(WW)
        else:
            self._cageCov = None

        # define overall covariance matrix as sum of genetic, environmental and cage covariance matrices
        if self._genoCov is None:
            if self._cageCov is None:
                self.covar = SumCov(self._envCov)
            else:
                self.covar = SumCov(self._envCov,self._cageCov)
        else:
            if self._cageCov is None:
                self.covar = SumCov(self._genoCov,self._envCov)
            else:
                self.covar = SumCov(self._genoCov,self._envCov,self._cageCov)

        ## define gp
        self._gp = GP(covar=self.covar,mean=mean)
        

    def optimize(self):
        """optimises the covariance matrix = estimate variance components"""
        if 0:
            # trial for inizialization it is complicated though
            cov = sp.array([[0.2,1e-4],[1e-4,1e-4]])
            self._genoCov.setCovariance(cov)
            self._envCov.setCovariance(cov)
            self._cageCov.scale = 0.2
        else:
            self._gp.covar.setRandomParams()

        #optimization - keep calc_ste = False
        self.conv, self.info = self._gp.optimize(calc_ste = False)
        
    def get_VCs(self, calc_ste, DGE, IGE, IEE, cageEffect):
        """function to access estimated variance components, standard errors"""
        if calc_ste:
            try:
                STE_output = self.getGenoSte(DGE, IGE, IEE, cageEffect)
                genetic_STEs = STE_output['R']
                #not sure what corr_params are
                corr_params = STE_output['corr_params']
            
            except:
                genetic_STEs = sp.array([[-999,-999],[-999,-999]])
                corr_params = (-999)
        else:
            genetic_STEs = sp.array([[-999,-999],[-999,-999]])
            corr_params = (-999)

        R = {}
        #whether the run converged
        R['conv'] = self.conv
        #should be small (e.g. < 10^-4)
        R['grad'] = self.info['grad']
        #Contrary to what it says, this is -LML
        R['LML']  = self._gp.LML()
        #number of focal animals
        R['sample_size'] = len(self.sampleID)
        #number of _cm animals. note that this doesnt say much as an animal that is not phenotyped nor a cage mate could still be in there
        R['sample_size_cm'] = len(self.sampleID_cm)
        #standard error for DGE
        R['STE_Ad'] = genetic_STEs[0,0]
        #standard error for SGE
        R['STE_As'] = genetic_STEs[1,1]
        #standard error for covariance between DGE and SGE
        R['STE_Ads'] = genetic_STEs[0,1]
        #covariance between DGE and SGE estimates
        R['corr_params'] = corr_params
        #effect sizes of fixed effects in the model
        R['b'] = self._gp.mean.b

        #variance components and total genetic variance below
        if DGE and (not IGE):
            R['var_Ad'] = self._genoCov.scale
            R['var_As'] = (-999)
            R['corr_Ads'] = (-999)
            R['total_gen_var'] = 1/covar_rescaling_factor(self._genoCov.K())
        elif IGE and (not DGE):
            R['var_Ad'] = (-999)
            R['var_As'] = self._genoCov.scale
            R['corr_Ads'] = (-999)
            R['total_gen_var'] = 1/covar_rescaling_factor(self._genoCov.K())
        elif DGE and IGE:
            R['var_Ad'] = self._genoCov.covff.K()[0,0]
            R['var_As'] = self._genoCov.covff.K()[1,1]
            R['corr_Ads'] = self._genoCov.covff.K()[0,1]/(sp.sqrt(R['var_Ad']) * sp.sqrt(R['var_As']))
            R['total_gen_var'] = 1/covar_rescaling_factor(self._genoCov.K())
        else:
            R['var_Ad'] = (-999)
            R['var_As'] = (-999)
            R['corr_Ads'] = (-999)
            R['total_gen_var'] = (-999)

        if not IEE:
            R['var_Ed'] = self._envCov.scale
            R['var_Es'] = (-999)
            R['corr_Eds'] = (-999)
        else:
            R['var_Ed'] = self._envCov.covff.K()[0,0]
            R['var_Es'] = self._envCov.covff.K()[1,1]
            R['corr_Eds'] = self._envCov.covff.K()[0,1] / (sp.sqrt(R['var_Ed']) * sp.sqrt(R['var_Es']))


        if cageEffect:
            R['var_C'] = self._cageCov.scale
            #environmental covariance matrix (fitted)
            envK = self._envCov.K() + self._cageCov.K()
            #overall (total) covariance matrix (fitted)
            if DGE or IGE:
                totK = self._genoCov.K() + self._envCov.K() + self._cageCov.K()
            else:
                totK = self._envCov.K() + self._cageCov.K()
        else:
            R['var_C'] = (-999)
            envK = self._envCov.K()
            if DGE or IGE:
                totK = self._genoCov.K() + self._envCov.K()
            else:
                totK = self._envCov.K()
        #calculate sample variance of environmental matrix and overall covariance matrices
        R['total_env_var'] = 1/covar_rescaling_factor(envK)
        R['total_var'] = 1/covar_rescaling_factor(totK)
        self.total_variance = R['total_var']

        return R

    def getGenoSte(self, DGE, IGE, IEE, cageEffect):

        self._gp.covar.getFisherInf()
        F = self._gp.covar.getFisherInf()

        # scalar in front of each term
        # ordering for geno and env is
        # direct, covar, indirect as in fisher matrix
        aP = []
        vi = []

        if DGE and (not IGE):
            aP.append(self._genoCov.scale)
            vi.append(1. / covar_rescaling_factor(self._genoCov.K0))
        elif IGE and (not DGE):
            aP.append(self._genoCov.scale)
            vi.append(1. / covar_rescaling_factor(self._genoCov.K0))
        elif DGE and IGE:
            aP.append(self._genoCov.covff.K()[0,0])
            aP.append(self._genoCov.covff.K()[0,1])
            aP.append(self._genoCov.covff.K()[1,1])
            vi.append(1. / covar_rescaling_factor(self._genoCov._K))
            vi.append(1. / covar_rescaling_factor(self._genoCov._KZ + self._genoCov._ZK))
            vi.append(1. / covar_rescaling_factor(self._genoCov._ZKZ))
        else:
            pass

        if not IEE:
            aP.append(self._envCov.scale)
            vi.append(1. / covar_rescaling_factor(self._envCov.K0))
        else:
            aP.append(self._envCov.covff.K()[0,0])
            aP.append(self._envCov.covff.K()[0,1])
            aP.append(self._envCov.covff.K()[1,1])
            vi.append(1. / covar_rescaling_factor(self._envCov._K))
            vi.append(1. / covar_rescaling_factor(self._envCov._KZ + self._envCov._ZK))
            vi.append(1. / covar_rescaling_factor(self._envCov._ZKZ))

        if cageEffect:
            aP.append(self._cageCov.scale)
            vi.append(1. / covar_rescaling_factor(self._cageCov.K0))
        else:
            pass

        # make them vectors
        aP = sp.array(aP)
        vi = sp.array(vi)

        # overall variance
        # this should correspond to the one you get from sampling
        v = (aP*vi).sum()

        # fractions of variance exaplined by each term
        # (can be negative)
        h = (aP*vi) / v

        # jacobean
        J = sp.zeros((aP.shape[0], aP.shape[0]))
        J[:, 0] = h / vi
        J[-1, 1:] = -v / vi[-1]
        for i in range(aP.shape[0]-1):
            J[i, i+1] = v / vi[i]

        # transformation of Fisher
        Fnew = sp.dot(J.T, sp.dot(F, J))

        # invert the new Fisher
        S,U = sp.linalg.eigh(Fnew)
        I = S>1e-9
        U = U[:,I]
        S = S[I]
        FI = sp.dot(U,sp.dot(sp.diag(S**(-1)),U.T))
        # reorder to have same ordering as before
        idxs = list(range(1, aP.shape[0]))
        idxs.append(0)
        FI = FI[idxs, :][:, idxs]
        # R is 2x2 matrix: STE_Ad and STE_As on diag, STE_Ads off
        R = sp.zeros((2, 2))
        STE_output = {}

        if DGE and IGE:
            FI_geno = FI[:3,:][:,:3]
            #STEs = sp.sqrt(FI_geno.diagonal()) ( ordered as Ad Ads As)
            #STEs = sqrt of var of VC corr_params 
            #fills diag and 1 off first
            R[sp.tril_indices(2)] = sp.sqrt(FI_geno.diagonal())
            #now fills other off
            R = R + R.T - sp.diag(R.diagonal())
        
            corr_param_Ad_As = FI_geno[0,2]/(sp.sqrt(FI_geno[0,0])*sp.sqrt(FI_geno[2,2]))
        
        elif DGE and (not IGE):
            R[0,0] = sp.sqrt(FI[0,0])
            R[0,1] = -999
            R[1,0] = -999
            R[1,1] = -999
            corr_param_Ad_As = -999
        
        elif (not DGE) and IGE:
            R[0,0] = -999
            R[0,1] = -999
            R[1,0] = -999
            R[1,1] = sp.sqrt(FI[0,0])
            corr_param_Ad_As = -999
       
        else:
            R[0,0] = -999
            R[0,1] = -999
            R[1,0] = -999
            R[1,1] = -999
            corr_param_Ad_As = -999

        
        STE_output['R']=R
        STE_output['corr_params']= corr_param_Ad_As

        return STE_output


    def getOutput(self):
        """to get output without having to specify DGE, SGE, ...."""
        return self.output
    