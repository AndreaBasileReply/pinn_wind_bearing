# ______          _           _     _ _ _     _   _      
# | ___ \        | |         | |   (_) (_)   | | (_)     
# | |_/ / __ ___ | |__   __ _| |__  _| |_ ___| |_ _  ___ 
# |  __/ '__/ _ \| '_ \ / _` | '_ \| | | / __| __| |/ __|
# | |  | | | (_) | |_) | (_| | |_) | | | \__ \ |_| | (__ 
# \_|  |_|  \___/|_.__/ \__,_|_.__/|_|_|_|___/\__|_|\___|
# ___  ___          _                 _                  
# |  \/  |         | |               (_)                 
# | .  . | ___  ___| |__   __ _ _ __  _  ___ ___         
# | |\/| |/ _ \/ __| '_ \ / _` | '_ \| |/ __/ __|        
# | |  | |  __/ (__| | | | (_| | | | | | (__\__ \        
# \_|  |_/\___|\___|_| |_|\__,_|_| |_|_|\___|___/        
#  _           _                     _                   
# | |         | |                   | |                  
# | |     __ _| |__   ___  _ __ __ _| |_ ___  _ __ _   _ 
# | |    / _` | '_ \ / _ \| '__/ _` | __/ _ \| '__| | | |
# | |___| (_| | |_) | (_) | | | (_| | || (_) | |  | |_| |
# \_____/\__,_|_.__/ \___/|_|  \__,_|\__\___/|_|   \__, |
#                                                   __/ |
#                                                  |___/ 
#														  
# MIT License
# 
# Copyright (c) 2019 Probabilistic Mechanics Laboratory
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

import pandas as pd
import numpy as np
import os

from pyDOE import lhs

from case_config import SUFFIX, SEED, JUDGMENT

# =============================================================================
#   RANDOM PLANE GENERATION
# =============================================================================

if __name__ == "__main__":
       
    def deltaGreaseDamagePlane(DOE,coefs):
        """Function to generate delta grease damage outputs for given DOE and plane coefficients
        """
        lowBound = 1e-7
        upBound = 1.3e-4
        delGrsDmg = lowBound + (upBound - lowBound) * (coefs[0]+coefs[1]*np.transpose(DOE)[0]+coefs[2]*np.transpose(DOE)[1]+coefs[3]*np.transpose(DOE)[2])
        return delGrsDmg
    
    parent_dir = os.path.dirname(os.getcwd())
    
    if SEED is not None:
        np.random.seed(SEED)

    npnts = 500
    # pyDOE ignora np.random.seed(): va passato seed= esplicitamente, altrimenti
    # il piano cambia a ogni esecuzione anche a parita' di CASE.
    Xolhs = lhs(n = 3, samples = npnts, criterion = 'maximin', iterations = 10,
                seed = SEED) if SEED is not None else lhs(n = 3, samples = npnts,
                criterion = 'maximin', iterations = 10)
    lowerBounds = np.asarray([[0.0,1/1500,60.0]])
    upperBounds = np.asarray([[1.0,1/500,80.0]])
    
    scaledXolhs =  np.repeat(lowerBounds, npnts, axis = 0) + Xolhs * (upperBounds - lowerBounds)
    
    if JUDGMENT:
        # Section 3.4: "engineering judgment can be used to limit delta-dGRS, which is
        # expected to be on the order of magnitude of the observed dGRS divided by the
        # number of time intervals (i.e., cycles)".  With dGRS ~ 1 over 6*24*30*6 ten-minute
        # steps, delta-dGRS must be able to reach down to ~1e-6 and up to ~1e-4.  The
        # unconstrained draw below puts the plane's LOWER bound at 1.3e-4*coefs[0], which
        # exceeds the early-life increment unless coefs[0] ~ 0, so the offset is bounded
        # and the slopes are scaled to keep the plane inside the expected band.
        # All coefficients stay positive (delta-dGRS grows with temperature, load and damage).
        # Tarato sui dati: dalle 6 ispezioni il Delta d GRS vero sta in [~0, 8.2e-05].
        # a0 fissa il PAVIMENTO del piano (1.3e-4*a0): se supera l'incremento di inizio
        # vita, il danno cumulato non puo' piu' scendere sotto e la sovrastima diventa
        # strutturale, non recuperabile dall'addestramento.  La somma dei coefficienti
        # fissa il TETTO, che deve superare 8.2e-05.  Tutti positivi: il degrado cresce
        # con temperatura, carico e danno gia' accumulato.
        coefrand = np.empty(4)
        coefrand[0] = np.random.uniform(0.0, 0.02)
        slopes = np.random.random(3)
        span = np.random.uniform(0.75, 1.0) - coefrand[0]
        coefrand[1:] = slopes * span / slopes.sum()
    else:
        coefrand = np.random.random(4)
    print(coefrand)
    dfPlane = pd.DataFrame({'dynamicLoads':np.transpose(scaledXolhs)[1],'bearingTemp':np.transpose(scaledXolhs)[2],'Dkappa':np.transpose(scaledXolhs)[0],'delDkappa':deltaGreaseDamagePlane(Xolhs,coefrand)})
    dfPlane.to_csv(parent_dir+'/data/random_plane_set_'+str(npnts)+'_adv'+SUFFIX+'.csv', index = False)
    