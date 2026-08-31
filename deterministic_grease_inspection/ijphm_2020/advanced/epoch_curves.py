# -*- coding: utf-8 -*-
"""Curva della masked MSE su d_BRG epoca per epoca, physics vs no-physics.

L'asse delle epoche del ramo cuscinetto esiste solo senza fisica (la catena SKF non
si addestra), quindi non serve a confrontare i due metodi.  L'asse comune sono le 50
epoche del GRASSO: mentre d_GRS migliora, entrambi i rami convertono un grasso diverso
in un d_BRG diverso, e le due curve si possono sovrapporre.

Ad ogni epoca del grasso si rifa' la previsione per la turbina 8 e la si passa nei due
rami cuscinetto (catena SKF / MLP gia' addestrato del run corrispondente), misurando la
masked MSE sulle 6 ispezioni del danno.

    python epoch_curves.py [--casi 1,2,3,4,10]
"""
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, RNN
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from pinn.layers import CumulativeDamageCell, getScalingDenseLayer
import train as T
from models_and_functions import create_pinn_model, arrange_table

DATA = '../data/'
rd = lambda f: pd.read_csv(DATA+f)


def modello_cuscinetto_mlp(Xb, seed, pesi, Cyc, Pc):
    """Ricostruisce il ramo cuscinetto senza fisica e ne carica i pesi gia' addestrati."""
    inc_nom = Cyc/(10**T.SN_B * Pc**T.SN_A)
    lob, upb = np.asarray([0.0]), np.asarray([10.0*inc_nom.max()])
    tf.keras.utils.set_random_seed(seed)
    mn = pd.Series(np.zeros(4)); rg = pd.Series(np.ones(4))   # sovrascritti da load_weights
    mlpb = Sequential([getScalingDenseLayer(mn, rg),
                       Dense(40, activation='sigmoid'), Dense(20, activation='elu'),
                       Dense(10, activation='elu'), Dense(5, activation='elu'),
                       Dense(1, activation='sigmoid')], name='mlp_cuscinetto')
    ph = Input(shape=(Xb.shape[2]+1,))
    sel = Lambda(lambda z: z[:, 1:])(ph)
    ob = Lambda(lambda z, lo=lob, up=upb: z*(up-lo)+lo)(mlpb(sel))
    cb = CumulativeDamageCell(model=Model(inputs=[ph], outputs=[ob]),
                              batch_input_shape=Xb.shape, dtype='float32',
                              initial_damage=np.zeros((1, 1), dtype='float32'))
    m = Sequential(); m.add(RNN(cell=cb, return_sequences=True, return_state=False,
                                batch_input_shape=Xb.shape, unroll=False))
    m.compile(loss='mse', optimizer='rmsprop')
    m.load_weights(pesi)
    return m


class Traccia(Callback):
    """A fine epoca: grasso -> d_BRG con i due rami -> masked MSE sulle 6 ispezioni."""
    def __init__(self, mv, X8, Cyc, Pc, Tc, Ver, mfis, mmlp, righe, caso):
        super().__init__()
        self.mv, self.X8, self.Cyc, self.Pc, self.Tc = mv, X8, Cyc, Pc, Tc
        self.Ver, self.mfis, self.mmlp, self.righe, self.caso = Ver, mfis, mmlp, righe, caso
        self.Vg = np.asarray(rd('ViscDamage_6Months.csv')[T.TURB_CUSC])

    def misura(self, epoca, loss):
        n = len(self.Ver)
        G = self.mv.predict(self.X8, verbose=0)[0, :, 0][:n]
        Xb = np.dstack((G, self.Cyc, np.log10(self.Pc), self.Tc))
        for nome, mod in (('physics', self.mfis), ('nophysics', self.mmlp)):
            B = mod.predict(Xb, verbose=0)[0, :, 0]
            e = B[T.INSP] - self.Ver[T.INSP]
            self.righe.append(dict(
                caso=self.caso, epoca=epoca, metodo=nome, loss_grasso=loss,
                masked_mse=float(np.mean(e**2)),
                errore_rel_pct=float(np.sqrt(np.mean((e/self.Ver[T.INSP])**2))*100),
                mse_grasso_t8=float(np.mean((G[T.INSP]-self.Vg[T.INSP])**2))))

    def on_train_begin(self, logs=None):
        self.misura(0, None)

    def on_epoch_end(self, epoch, logs=None):
        self.misura(epoch+1, float(logs['loss']))


def esegui_caso(caso, righe):
    seed = 3000 + caso
    t0 = time.time()
    Ptr = np.transpose(np.asarray(rd('DynamicLoad_6Months.csv').dropna()))
    Ttr = np.transpose(np.asarray(rd('BearingTemp_6Months.csv').dropna()))
    Vtr = np.asarray(rd('ViscDamage_6Months.csv').dropna())
    Xtr = np.dstack((1/Ptr, Ttr))
    Ytr = np.transpose(np.asarray([Vtr[T.INSP, :]]))

    # ---- 1-2. piano e MLP, identici a train.py -----------------------------
    piano, _ = T.piano_iniziale(seed)
    lo = np.asarray([piano.delDkappa.min()]); up = np.asarray([piano.delDkappa.max()])
    ins = piano[['Dkappa', 'dynamicLoads', 'bearingTemp']]
    tf.keras.utils.set_random_seed(seed)
    mlp = Sequential([getScalingDenseLayer(ins.min(axis=0), ins.max(axis=0)-ins.min(axis=0)),
                      Dense(40, activation='sigmoid'), Dense(20, activation='elu'),
                      Dense(10, activation='elu'), Dense(5, activation='elu'),
                      Dense(1, activation='sigmoid')], name='mlp_grasso')
    mlp.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.RMSprop(T.MLP_LR))
    o = piano[['delDkappa']]
    mlp.fit(ins, (o-o.min())/(o.max()-o.min()), epochs=T.MLP_EPOCHS, verbose=0,
            callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.85, min_lr=1e-15,
                                         patience=30, mode='min')])
    mlp.trainable = True

    # ---- ramo cuscinetto: dati e due modelli, costruiti una volta sola ------
    Pc = np.asarray(rd('DynamicLoad_6Months.csv')[T.TURB_CUSC])
    Tc = np.asarray(rd('BearingTemp_6Months.csv')[T.TURB_CUSC])
    Cyc = np.asarray(rd('Cycles_6Months_bsc.csv')[T.TURB_CUSC])
    Ver = np.asarray(pd.read_csv(DATA+'True_FatigueDamage_6Months_bsc.csv',
                                 header=None).iloc[:, 0])
    nb = min(len(Pc), len(Ver))
    Pc, Tc, Cyc, Ver = Pc[:nb], Tc[:nb], Cyc[:nb], Ver[:nb]
    forma = (1, nb, 4)
    tabs = {k: arrange_table(pd.read_csv('../tables/%s.csv' % k))
            for k in ('aSKF', 'kappa', 'etac')}
    mfis = create_pinn_model(T.SN_A, T.SN_B, T.PU,
        tabs['aSKF']['data'],  tabs['aSKF']['bounds'],  tabs['aSKF']['table_shape'],
        tabs['kappa']['data'], tabs['kappa']['bounds'], tabs['kappa']['table_shape'],
        tabs['etac']['data'],  tabs['etac']['bounds'],  tabs['etac']['table_shape'],
        np.zeros((1, 1), dtype='float32'), forma, [1], [2], [3], [4], 'float32',
        return_sequences=True)
    idx = sorted(glob.glob('runs/*case%d_nophysics_pct100/models/mlp_cuscinetto.h5py.index' % caso))[-1]
    pesi = idx[:-len('.index')]
    mmlp = modello_cuscinetto_mlp(np.empty(forma), seed, pesi, Cyc, Pc)
    print('   pesi MLP cuscinetto: %s' % pesi)

    # ---- 3. RNN del grasso, con la traccia a ogni epoca --------------------
    mg = T.rnn_grasso(mlp, Xtr.shape, lo, up)
    mv = T.rnn_grasso(mlp, (1,)+Xtr.shape[1:], lo, up)      # stesso mlp: pesi condivisi
    X8 = Xtr[T.IDX_CUSC][None]
    tr = Traccia(mv, X8, Cyc, Pc, Tc, Ver, mfis, mmlp, righe, caso)
    mg.fit(Xtr, Ytr, epochs=T.RNN_EPOCHS, verbose=0, steps_per_epoch=1,
           callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.7, min_lr=1e-15,
                                        patience=5, mode='min'), tr])

    # ---- controllo: l'ultimo punto deve coincidere col run gia' salvato ----
    fin = {r['metodo']: r for r in righe if r['caso'] == caso and r['epoca'] == T.RNN_EPOCHS}
    for m in ('physics', 'nophysics'):
        j = json.load(open(sorted(glob.glob('runs/*case%d_%s_pct100/metrics.json' % (caso, m)))[-1]))
        atteso = j['cuscinetto']['rmse_ispezioni']**2
        print('   %-9s epoca 50: %.4e   run salvato: %.4e   scarto %.2f%%'
              % (m, fin[m]['masked_mse'], atteso, 100*(fin[m]['masked_mse']-atteso)/atteso))
    print('   caso %d in %.0f s' % (caso, time.time()-t0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--casi', default='1,2,3,4,10')
    ap.add_argument('--out', default='runs/epoch_curves.csv')
    A = ap.parse_args()
    righe = []
    for c in [int(x) for x in A.casi.split(',') if x]:
        print('-> caso %d' % c)
        esegui_caso(c, righe)
        pd.DataFrame(righe).to_csv(A.out, index=False)   # salva mano a mano
        tf.keras.backend.clear_session()
    print('\n-> %s' % A.out)


if __name__ == '__main__':
    main()
