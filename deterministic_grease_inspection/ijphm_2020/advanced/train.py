# -*- coding: utf-8 -*-
"""Addestra il modello completo del paper, con o senza fisica sul ramo cuscinetto.

    python train.py --physics    [--case N] [--train-frac F]
    python train.py --no-physics [--case N] [--train-frac F]

La pipeline e' quella dell'articolo (Yucesan & Viana, IJPHM 11(1), 2020):

  1. piano lineare di partenza (eq. 11), scelto dal caso
  2. MLP addestrato sul piano            RMSprop lr 0.01,   500 epoche
  3. rifinitura dentro la RNN di accumulo RMSprop lr 5e-4,   50 epoche
     loss = MSE mascherato sulle 6 ispezioni del grasso (eq. 10)
  4. dal grasso previsto si ricava il danno di fatica del cuscinetto

L'unica differenza fra i due comandi e' il passo 4:

  --physics      catena del catalogo SKF: kappa -> eta_c -> a_SKF -> L10 ->
                 Palmgren-Miner.  Zero parametri, zero dati, nessun addestramento.
  --no-physics   un MLP con gli stessi ingressi calcola l'incremento al posto
                 della catena SKF, addestrato contro la curva di danno reale.

--train-frac riduce i dati di training, per le curve di data efficiency.  Di default
e' 100 (tutti i dati, un run solo).  Piu' percentuali separate da virgola vengono
addestrate in sequenza, un run per percentuale:

    python train.py --physics --train-frac 20,40,60,80,100

La riga in loss_on_percentage.csv viene scritta alla fine di ogni percentuale, quindi
una curva interrotta a meta' conserva i punti gia' passati.  Cosa si riduce:

  passo 3 (grasso)      meno turbine, con tutte e 6 le ispezioni.  La turbina 8 resta
                        sempre dentro: il passo 4 usa il suo grasso previsto.
  passo 4 (cuscinetto)  meno ispezioni della turbina 8, l'unica con la verita' sul
                        danno: scelte spaziate e sempre comprendendo la sesta.  Conta
                        solo con --no-physics, perche' la catena SKF non si addestra.

La frazione finisce in config.json (train_frac, train_pct) e nel nome della cartella
del run (..._case4_physics_pct40), cosi' compare_data_efficiency.py la ritrova.

NOTA SUI DATI: la verita' sul danno del cuscinetto esiste per una sola turbina.
True_FatigueDamage_6Months_bsc.csv e' la TURBINA 8, non la 1: con gli ingressi
della turbina 8 la catena SKF riproduce quel file entro +0.4%, con quelli della
turbina 1 sbaglia del +43%.  Il run02_predict_pinn.py originale usa la turbina 1.
La turbina 8 fa parte del training del grasso, quindi la valutazione del ramo
cuscinetto non e' su dati completamente non visti: e' l'unica possibile.
"""
import argparse, json, os, sys, time, datetime
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, RNN
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from pinn.layers import CumulativeDamageCell, getScalingDenseLayer
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '../basic')
from models_and_functions import create_pinn_model, arrange_table

BLU, ARA, INK, GRID = '#2a78d6', '#eb6834', '#0f141a', '#dce3ea'
INSP = np.asarray([6*24*30*1, 6*24*30*2, 6*24*30*3, 6*24*30*4, 6*24*30*5, 6*24*30*6-1])
A1, C_DYN, PU = 1.0, 6000, 750                      # SKF 230/600 CAW33
SN_A = -10/3
SN_B = (10/3)*np.log10(C_DYN) + np.log10(1e6) + np.log10(A1)
TURB_CUSC = 'Turbine8'                              # unica con verita' sul cuscinetto
IDX_CUSC  = 7                                       # sua posizione nel training set

# valori di default dell'articolo
MLP_EPOCHS, RNN_EPOCHS = 500, 50
MLP_LR, RNN_LR = 0.01, 5e-4


def piano_iniziale(seed, npnts=500):
    """Piano lineare di partenza (eq. 11), coefficienti vincolati come da sez. 3.4.

    Tutti positivi (il degrado cresce con temperatura, carico e danno accumulato) e
    limitati perche' delta_d resti nella banda fisica attesa.  a0 fissa il PAVIMENTO:
    se supera l'incremento di inizio vita (~4e-06) la sovrastima diventa strutturale.
    """
    from pyDOE import lhs
    np.random.seed(seed)
    # pyDOE usa np.random.default_rng() e ignora np.random.seed(): va seminato qui.
    X = lhs(n=3, samples=npnts, criterion='maximin', iterations=10, seed=seed)
    lo_b, up_b = np.asarray([[0.0, 1/1500, 60.0]]), np.asarray([[1.0, 1/500, 80.0]])
    Xs = np.repeat(lo_b, npnts, axis=0) + X*(up_b-lo_b)
    a = np.empty(4)
    a[0] = np.random.uniform(0.0, 0.02)
    sl = np.random.random(3)
    a[1:] = sl*(np.random.uniform(0.75, 1.0)-a[0])/sl.sum()
    lo, up = 1e-7, 1.3e-4
    dd = lo + (up-lo)*(a[0] + a[1]*X.T[0] + a[2]*X.T[1] + a[3]*X.T[2])
    return pd.DataFrame({'dynamicLoads': Xs.T[1], 'bearingTemp': Xs.T[2],
                         'Dkappa': Xs.T[0], 'delDkappa': dd}), a


def masked_mse(idx):
    ii = tf.constant(idx, dtype=tf.int32)
    def f(y_true, y_pred):
        return tf.keras.losses.mean_squared_error(
            y_true, tf.expand_dims(tf.gather(y_pred[:, :, 0], ii, axis=1), -1))
    return f


def rnn_grasso(mlp, shape, lo, up):
    d0 = np.zeros((shape[0], 1), dtype='float32')
    ph = Input(shape=(shape[2]+1,))
    out = Lambda(lambda x, lo=lo, up=up: x*(up-lo)+lo)(mlp(ph))
    cell = CumulativeDamageCell(model=Model(inputs=[ph], outputs=[out]),
                                batch_input_shape=shape, dtype='float32', initial_damage=d0)
    m = Sequential()
    m.add(RNN(cell=cell, return_sequences=True, return_state=False,
              batch_input_shape=shape, unroll=False))
    m.compile(loss=masked_mse(INSP), optimizer=RMSprop(RNN_LR), metrics=[masked_mse(INSP)])
    return m


def sottoinsieme_turbine(n_tot, frac):
    """Turbine di training tenute da --train-frac, in ordine, TURB_CUSC compresa."""
    n = max(1, int(round(frac*n_tot)))
    altre = [i for i in range(n_tot) if i != IDX_CUSC]
    return sorted([IDX_CUSC] + altre[:n-1])


def sottoinsieme_ispezioni(n_tot, frac):
    """Ispezioni del cuscinetto tenute da --train-frac: spaziate, l'ultima sempre.

    I livelli sono ANNIDATI: ogni percentuale aggiunge ispezioni a quella sotto invece
    di ricalcolarle da capo, cosi' salendo lungo la curva cambia solo la taglia del
    campione e non la sua composizione.  Con np.linspace non era vero (l'ispezione 4
    entrava al 60% e usciva all'80%) e il salto falsava il confronto fra i livelli.

    Si parte dall'ultima ispezione, poi dalla prima, e ogni aggiunta successiva spezza
    a meta' il tratto scoperto piu' lungo: la spaziatura resta quella di prima.
    """
    n = max(1, int(round(frac*n_tot)))
    scelti = [n_tot-1] if n == 1 else [0, n_tot-1]
    while len(scelti) < n:
        s = sorted(scelti)
        buchi, prec = [], -1
        for c in s + [n_tot]:
            if c - prec > 1:                          # tratto libero fra prec e c
                buchi.append((prec+1, c-1))
            prec = c
        a, b = max(buchi, key=lambda r: r[1]-r[0])
        scelti.append((a+b)//2)
    return np.unique(np.asarray(scelti, dtype=int))


def aggiorna_loss_on_percentage(percorso, riga):
    """Aggiunge (o rimpiazza) la riga di questo run nel CSV cumulativo.

    Una riga per (metodo, caso, percentuale): rilanciando lo stesso punto il valore
    vecchio viene sostituito, non duplicato.  Il file vive in --outdir, quindi per
    avere fisica e non-fisica nello stesso CSV basta dare lo stesso --outdir ai due
    comandi.  L'ordine e' per metodo e percentuale, cosi' le due curve si leggono
    affiancate senza doverlo riordinare.
    """
    col = ['percentuale', 'metodo', 'caso', 'mse_danno_ispezioni', 'mse_danno_curva',
           'loss_grasso', 'loss_cuscinetto', 'mse_grasso_test', 'turbine',
           'ispezioni_cuscinetto', 'epoche_cuscinetto', 'timestamp', 'run']
    chiave = lambda r: (r['metodo'], r['caso'], r['percentuale'])
    righe = {}
    if os.path.isfile(percorso):
        for r in pd.read_csv(percorso).to_dict('records'):
            righe[chiave(r)] = r
    righe[chiave(riga)] = riga
    pd.DataFrame(list(righe.values()), columns=col).sort_values(
        ['metodo', 'percentuale']).to_csv(percorso, index=False)
    return percorso


def leggi_frazioni(spec):
    """'20,40,60,80,100' oppure '40' oppure '0.4' -> lista di frazioni in (0, 1]."""
    fr = []
    for pezzo in str(spec).replace(' ', '').split(','):
        if not pezzo:
            continue
        try:
            v = float(pezzo)
        except ValueError:
            sys.exit('--train-frac non numerico: %r' % pezzo)
        f = v/100.0 if v > 1 else v
        if not 0 < f <= 1:
            sys.exit('--train-frac fuori intervallo: atteso 0-1 oppure 1-100, ricevuto %g' % v)
        fr.append(f)
    if not fr:
        sys.exit('--train-frac vuoto')
    return sorted(set(fr))


def esegui(A, frac, ts, metodo, seed):
    """Un punto della curva: addestra con la frazione `frac` dei dati e salva tutto.

    La riga in loss_on_percentage.csv viene scritta qui, alla fine di questa
    percentuale: interrompendo una curva a meta\' si tiene quello che e\' gia\' passato.
    """
    pct = int(round(100*frac))
    out = os.path.join(A.outdir, '%s_case%d_%s_pct%d' % (ts, A.case, metodo, pct))
    for sub in ('models', 'plots'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    print('-> %s   (caso #%d, %s, %d%% dei dati)' % (out, A.case, metodo, pct))

    parent = os.path.dirname(os.getcwd())
    rd = lambda f: pd.read_csv(parent+'/data/'+f)
    Ptr = np.transpose(np.asarray(rd('DynamicLoad_6Months.csv').dropna()))
    Ttr = np.transpose(np.asarray(rd('BearingTemp_6Months.csv').dropna()))
    Vtr = np.asarray(rd('ViscDamage_6Months.csv').dropna())
    Pte = np.transpose(np.asarray(rd('DynamicLoad_6Months_Val_adv.csv').dropna()))
    Tte = np.transpose(np.asarray(rd('BearingTemp_6Months_Val_adv.csv').dropna()))
    Vte = np.asarray(rd('ViscDamage_6Months_Val_adv.csv').dropna())
    Xtr, Xte = np.dstack((1/Ptr, Ttr)), np.dstack((1/Pte, Tte))
    Ytr = np.transpose(np.asarray([Vtr[INSP, :]]))

    n_turb_tot = Xtr.shape[0]
    i_turb = sottoinsieme_turbine(n_turb_tot, frac)
    Xtr, Ytr, Vtr = Xtr[i_turb], Ytr[i_turb], Vtr[:, i_turb]
    turbine_id = [i+1 for i in i_turb]               # numerazione del dataset intero
    i_cusc = i_turb.index(IDX_CUSC)                  # TURB_CUSC nel set ridotto
    j_isp = sottoinsieme_ispezioni(len(INSP), frac)
    INSP_B = INSP[j_isp]                             # ispezioni viste dal passo 4
    print('   grasso: %d/%d turbine %s   cuscinetto: %d/%d ispezioni %s'
          % (len(i_turb), n_turb_tot, turbine_id, len(INSP_B), len(INSP),
             [int(k)+1 for k in j_isp]))

    t0 = time.time()
    # ---- 1-2. piano e MLP -------------------------------------------------
    piano, coef = piano_iniziale(seed)
    piano.to_csv(os.path.join(out, 'models', 'piano_iniziale.csv'), index=False)
    lo = np.asarray([piano.delDkappa.min()]); up = np.asarray([piano.delDkappa.max()])
    ins = piano[['Dkappa', 'dynamicLoads', 'bearingTemp']]
    tf.keras.utils.set_random_seed(seed)
    mlp = Sequential([getScalingDenseLayer(ins.min(axis=0), ins.max(axis=0)-ins.min(axis=0)),
                      Dense(40, activation='sigmoid'), Dense(20, activation='elu'),
                      Dense(10, activation='elu'), Dense(5, activation='elu'),
                      Dense(1, activation='sigmoid')], name='mlp_grasso')
    mlp.compile(loss='mean_squared_error', optimizer=RMSprop(MLP_LR))
    o = piano[['delDkappa']]
    h1 = mlp.fit(ins, (o-o.min())/(o.max()-o.min()), epochs=MLP_EPOCHS, verbose=0,
                 callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.85, min_lr=1e-15,
                                              patience=30, mode='min')])
    mlp.save(os.path.join(out, 'models', 'mlp_grasso.h5py')); mlp.trainable = True

    # ---- 3. RNN del grasso ------------------------------------------------
    mg = rnn_grasso(mlp, Xtr.shape, lo, up)
    hg = mg.fit(Xtr, Ytr, epochs=RNN_EPOCHS, verbose=0, steps_per_epoch=1,
                callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.7, min_lr=1e-15,
                                             patience=5, mode='min')])
    mg.save_weights(os.path.join(out, 'models', 'rnn_grasso.h5py'))
    Gtr = mg.predict(Xtr, verbose=0)[:, :, 0]
    mv = rnn_grasso(mlp, Xte.shape, lo, up); mv.set_weights(mg.get_weights())
    Gte = mv.predict(Xte, verbose=0)[:, :, 0]
    t_grasso = time.time()-t0

    # ---- 4. ramo cuscinetto ----------------------------------------------
    t1 = time.time()
    Pc  = np.asarray(rd('DynamicLoad_6Months.csv')[TURB_CUSC])
    Tc  = np.asarray(rd('BearingTemp_6Months.csv')[TURB_CUSC])
    Cyc = np.asarray(rd('Cycles_6Months_bsc.csv')[TURB_CUSC])
    Ver = np.asarray(pd.read_csv(parent+'/data/True_FatigueDamage_6Months_bsc.csv',
                                 header=None).iloc[:, 0])
    nb = min(len(Pc), len(Ver))
    Pc, Tc, Cyc, Ver = Pc[:nb], Tc[:nb], Cyc[:nb], Ver[:nb]
    Gc = Gtr[i_cusc][:nb]                            # grasso PREVISTO dal passo 3
    Xb = np.dstack((Gc, Cyc, np.log10(Pc), Tc))
    d0b = np.zeros((1, 1), dtype='float32')
    hb = None

    if A.physics:
        tabs = {k: arrange_table(pd.read_csv(parent+'/tables/%s.csv' % k))
                for k in ('aSKF', 'kappa', 'etac')}
        mb = create_pinn_model(SN_A, SN_B, PU,
            tabs['aSKF']['data'],  tabs['aSKF']['bounds'],  tabs['aSKF']['table_shape'],
            tabs['kappa']['data'], tabs['kappa']['bounds'], tabs['kappa']['table_shape'],
            tabs['etac']['data'],  tabs['etac']['bounds'],  tabs['etac']['table_shape'],
            d0b, Xb.shape, [1], [2], [3], [4], 'float32', return_sequences=True)
        n_oss_b, ep_b = 0, 0
        desc_b = 'catena SKF (kappa, eta_c, a_SKF, L10), nessun parametro addestrabile'
    else:
        inc_nom = Cyc/(10**SN_B * Pc**SN_A)          # scala nota a priori
        lob, upb = np.asarray([0.0]), np.asarray([10.0*inc_nom.max()])
        tf.keras.utils.set_random_seed(seed)
        fe = np.stack([Gc, Cyc, np.log10(Pc), Tc], 1)
        mn, rg = fe.min(0), fe.max(0)-fe.min(0)
        mlpb = Sequential([getScalingDenseLayer(pd.Series(mn), pd.Series(rg)),
                           Dense(40, activation='sigmoid'), Dense(20, activation='elu'),
                           Dense(10, activation='elu'), Dense(5, activation='elu'),
                           Dense(1, activation='sigmoid')], name='mlp_cuscinetto')
        phb = Input(shape=(Xb.shape[2]+1,))
        sel = Lambda(lambda z: z[:, 1:])(phb)        # scarta lo stato: nessuna retroazione
        ob  = Lambda(lambda z, lo=lob, up=upb: z*(up-lo)+lo)(mlpb(sel))
        cb  = CumulativeDamageCell(model=Model(inputs=[phb], outputs=[ob]),
                                   batch_input_shape=Xb.shape, dtype='float32',
                                   initial_damage=d0b)
        mb = Sequential(); mb.add(RNN(cell=cb, return_sequences=True, return_state=False,
                                      batch_input_shape=Xb.shape, unroll=False))
        mb.compile(loss=masked_mse(INSP_B), optimizer=RMSprop(RNN_LR),
                   metrics=[masked_mse(INSP_B)])
        hb = mb.fit(Xb, Ver[INSP_B].reshape(1, len(INSP_B), 1), epochs=A.bearing_epochs,
                    verbose=0, steps_per_epoch=1,
                    callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.7, min_lr=1e-15,
                                                 patience=10, mode='min')])
        mb.save_weights(os.path.join(out, 'models', 'mlp_cuscinetto.h5py'))
        n_oss_b, ep_b = len(INSP_B), A.bearing_epochs
        desc_b = 'MLP al posto della catena SKF, %d osservazioni' % n_oss_b
    Bpred = mb.predict(Xb, verbose=0)[0, :, 0]
    t_cusc = time.time()-t1

    # ---- salvataggi -------------------------------------------------------
    pd.DataFrame({'fase': 'mlp_piano', 'epoch': h1.epoch, 'loss': h1.history['loss']}).to_csv(
        os.path.join(out, 'loss_mlp_piano.csv'), index=False)
    pd.DataFrame({'fase': 'rnn_grasso', 'epoch': hg.epoch, 'loss': hg.history['loss']}).to_csv(
        os.path.join(out, 'loss_grasso.csv'), index=False)
    if hb is not None:
        pd.DataFrame({'fase': 'cuscinetto', 'epoch': hb.epoch, 'loss': hb.history['loss']}).to_csv(
            os.path.join(out, 'loss_cuscinetto.csv'), index=False)

    righe = []
    ids_te = list(range(1, Vte.shape[1]+1))
    for split, P_, V_, ids in (('train', Gtr, Vtr, turbine_id), ('test', Gte, Vte, ids_te)):
        for i in range(P_.shape[0]):
            for k, idx in enumerate(INSP):
                righe.append(dict(split=split, metodo=metodo, caso=A.case, turbina=ids[i],
                                  ispezione=k+1, previsto=float(P_[i, idx]),
                                  reale=float(V_[idx, i]), errore=float(V_[idx, i]-P_[i, idx])))
    dfg = pd.DataFrame(righe)
    dfg[dfg.split == 'test'].to_csv(os.path.join(out, 'grasso_predictions_test.csv'), index=False)
    dfg[dfg.split == 'train'].to_csv(os.path.join(out, 'grasso_predictions_train.csv'), index=False)

    day = np.arange(143, nb, 144)
    pd.DataFrame({'metodo': metodo, 'caso': A.case, 'giorno': (day+1)//144,
                  'previsto': Bpred[day], 'reale': Ver[day],
                  'errore': Ver[day]-Bpred[day]}).to_csv(
                  os.path.join(out, 'cuscinetto_predictions.csv'), index=False)
    dgt = {'giorno': (day+1)//144, 'metodo': metodo}
    for i in range(Gte.shape[0]):
        dgt['test_T%d_pred' % (i+1)] = Gte[i, day]
        dgt['test_T%d_reale' % (i+1)] = Vte[day, i]
    pd.DataFrame(dgt).to_csv(os.path.join(out, 'grasso_predictions_daily.csv'), index=False)

    te, tr = dfg[dfg.split == 'test'], dfg[dfg.split == 'train']
    mse = lambda d: float(np.mean(d.errore**2))
    rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
    M = dict(
        metodo=metodo, usa_fisica=bool(A.physics), caso=A.case, seed=seed,
        dati=dict(frazione=frac, percentuale=pct,
                  turbine_grasso=len(i_turb), turbine_totali=n_turb_tot,
                  turbine=turbine_id, osservazioni_grasso=int(len(i_turb)*len(INSP)),
                  ispezioni_cuscinetto=int(len(INSP_B)), ispezioni_totali=int(len(INSP))),
        grasso=dict(modello='accumulo danno + MLP (invariato nei due comandi)',
                    parametri=int(sum(np.prod(w.shape) for w in mg.trainable_weights)),
                    epoche_mlp=MLP_EPOCHS, epoche_rnn=RNN_EPOCHS,
                    osservazioni=int(Vtr.shape[1]*6),
                    mse_test=mse(te), rmse_test=float(np.sqrt(mse(te))),
                    mse_train=mse(tr), rmse_train=float(np.sqrt(mse(tr))),
                    bias_test=float(te.errore.mean()),
                    loss_finale=float(hg.history['loss'][-1]), tempo_s=round(t_grasso, 1)),
        cuscinetto=dict(modello=desc_b, turbina=TURB_CUSC,
                        parametri=int(sum(np.prod(w.shape) for w in mb.trainable_weights)),
                        osservazioni=int(n_oss_b), epoche=int(ep_b),
                        rmse_curva=rmse(Bpred, Ver), rmse_ispezioni=rmse(Bpred[INSP], Ver[INSP]),
                        danno_finale_previsto=float(Bpred[-1]), danno_finale_reale=float(Ver[-1]),
                        errore_finale_pct=float(100*(Bpred[-1]-Ver[-1])/Ver[-1]),
                        loss_finale=float(hb.history['loss'][-1]) if hb else None,
                        tempo_s=round(t_cusc, 1)),
        piano=dict(coefficienti=[float(c) for c in coef], min=float(piano.delDkappa.min()),
                   max=float(piano.delDkappa.max())),
        tempo_totale_s=round(t_grasso+t_cusc, 1))
    json.dump(M, open(os.path.join(out, 'metrics.json'), 'w'), indent=2)

    # curva loss/percentuale, cumulativa fra i run
    fcsv = aggiorna_loss_on_percentage(
        os.path.join(A.outdir, 'loss_on_percentage.csv'),
        dict(percentuale=pct, metodo=metodo, caso=A.case,
             mse_danno_ispezioni=M['cuscinetto']['rmse_ispezioni']**2,
             mse_danno_curva=M['cuscinetto']['rmse_curva']**2,
             loss_grasso=M['grasso']['loss_finale'],
             loss_cuscinetto=M['cuscinetto']['loss_finale'],
             mse_grasso_test=M['grasso']['mse_test'],
             turbine=len(i_turb), ispezioni_cuscinetto=int(len(INSP_B)),
             epoche_cuscinetto=int(ep_b), timestamp=ts,
             run=os.path.basename(out)))
    json.dump({**vars(A), 'metodo': metodo, 'seed': seed, 'timestamp': ts,
               'train_frac': frac, 'train_pct': pct,
               'mlp_epochs': MLP_EPOCHS, 'mlp_lr': MLP_LR, 'rnn_epochs': RNN_EPOCHS,
               'rnn_lr': RNN_LR, 'python': sys.version.split()[0],
               'tensorflow': tf.__version__}, open(os.path.join(out, 'config.json'), 'w'), indent=2)

    # ---- grafici ----------------------------------------------------------
    c = BLU if A.physics else ARA
    def base(ax, xl, yl, ti):
        ax.grid(axis='y', color=GRID, lw=.7); ax.set_axisbelow(True)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'): ax.spines[s].set_color(GRID)
        ax.tick_params(colors='#55636f', length=0, labelsize=9)
        ax.set_xlabel(xl, fontsize=10); ax.set_ylabel(yl, fontsize=10)
        ax.set_title(ti, fontsize=12, loc='left', fontweight='bold', color=INK, pad=10)

    f, a = plt.subplots(figsize=(7.5, 4))
    a.plot(hg.epoch, hg.history['loss'], color=BLU, lw=2.2, label='grasso (RNN)')
    if hb is not None:
        a.plot(hb.epoch, hb.history['loss'], color=ARA, lw=2.2, label='cuscinetto (MLP)')
    a.set_yscale('log'); base(a, 'Epoca', 'Loss (MSE)', 'Loss di training — caso #%d, %s' % (A.case, metodo))
    lg = a.legend(frameon=False, fontsize=9.5); [t.set_color(INK) for t in lg.get_texts()]
    f.tight_layout(); f.savefig(os.path.join(out, 'plots', 'loss.png'), dpi=160); plt.close(f)

    f, a = plt.subplots(figsize=(5.2, 5.2))
    lim = [0, max(dfg.reale.max(), dfg.previsto.max())*1.05]
    a.plot(lim, lim, '--', color='#8b97a3', lw=1.2)
    a.scatter(tr.reale, tr.previsto, s=42, color=BLU, alpha=.32, edgecolor='w', lw=.9, label='train')
    a.scatter(te.reale, te.previsto, s=52, color=BLU, alpha=.95, edgecolor='w', lw=1.1, label='test')
    a.set_xlim(lim); a.set_ylim(lim)
    base(a, 'Reale', 'Previsto', 'Grasso: previsto vs reale')
    lg = a.legend(frameon=False, fontsize=9.5); [t.set_color(INK) for t in lg.get_texts()]
    f.tight_layout(); f.savefig(os.path.join(out, 'plots', 'grasso_predicted_vs_actual.png'), dpi=160); plt.close(f)

    f, a = plt.subplots(figsize=(7.5, 4))
    for i in range(Gte.shape[0]):
        a.plot(dgt['giorno'], dgt['test_T%d_reale' % (i+1)], color=INK, lw=1.5, alpha=.7,
               label='reale' if i == 0 else None)
        a.plot(dgt['giorno'], dgt['test_T%d_pred' % (i+1)], color=BLU, lw=1.7, ls='--',
               label='previsto' if i == 0 else None)
    base(a, 'Giorni', 'Danno del grasso', 'Grasso — turbine di test')
    lg = a.legend(frameon=False, fontsize=9.5); [t.set_color(INK) for t in lg.get_texts()]
    f.tight_layout(); f.savefig(os.path.join(out, 'plots', 'grasso_test_curves.png'), dpi=160); plt.close(f)

    f, a = plt.subplots(figsize=(7.5, 4))
    a.plot((day+1)//144, Ver[day], color=INK, lw=2.4, label='reale')
    a.plot((day+1)//144, Bpred[day], color=c, lw=2, ls='--',
           label='previsto (%s)' % ('SKF' if A.physics else 'MLP'))
    titolo_b = 'catena SKF' if A.physics else 'MLP al posto della catena SKF'
    base(a, 'Giorni', 'Danno del cuscinetto', 'Cuscinetto — %s' % titolo_b)
    lg = a.legend(frameon=False, fontsize=9.5); [t.set_color(INK) for t in lg.get_texts()]
    f.tight_layout(); f.savefig(os.path.join(out, 'plots', 'cuscinetto_damage.png'), dpi=160); plt.close(f)

    print(json.dumps({'caso': A.case, 'metodo': metodo, 'dati_pct': pct,
                      'cuscinetto_mse_ispezioni': M['cuscinetto']['rmse_ispezioni']**2,
                      'grasso_rmse_test': M['grasso']['rmse_test'],
                      'cuscinetto_rmse': M['cuscinetto']['rmse_curva'],
                      'cuscinetto_errore_finale_pct': M['cuscinetto']['errore_finale_pct'],
                      'tempo_totale_s': M['tempo_totale_s']}, indent=2))
    print('salvato in %s' % out)
    print('curva loss/percentuale aggiornata: %s' % fcsv)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--physics', action='store_true',
                   help='ramo cuscinetto con la catena SKF (situazione del paper)')
    g.add_argument('--no-physics', dest='nophysics', action='store_true',
                   help='ramo cuscinetto con un MLP al posto della formula')
    ap.add_argument('--case', type=int, default=3, choices=range(1, 11), metavar='N',
                    help='inizializzazione del piano, da 1 a 10 (default: 3)')
    ap.add_argument('--train-frac', default='100', metavar='F',
                    help='percentuale di dati di training (default: 100). Piu\' valori '
                         'separati da virgola vengono addestrati in sequenza, un run '
                         'per percentuale: --train-frac 20,40,60,80,100')
    ap.add_argument('--bearing-epochs', type=int, default=300,
                    help='solo con --no-physics (default: 300)')
    ap.add_argument('--outdir', default='runs')
    ap.add_argument('--nota', default='')
    A = ap.parse_args()

    metodo = 'physics' if A.physics else 'nophysics'
    seed = 3000 + A.case
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    frazioni = leggi_frazioni(A.train_frac)
    if len(frazioni) > 1:
        print('curva su %d percentuali: %s   (--train-frac F per un punto solo)'
              % (len(frazioni), ', '.join('%d%%' % round(100*f) for f in frazioni)))
    for k, frac in enumerate(frazioni):
        if k:
            tf.keras.backend.clear_session()         # i grafi della RNN srotolata pesano
        esegui(A, frac, ts, metodo, seed)



if __name__ == '__main__':
    main()
