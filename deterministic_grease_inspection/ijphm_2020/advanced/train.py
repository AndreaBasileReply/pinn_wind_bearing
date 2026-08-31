# -*- coding: utf-8 -*-
"""Addestra il modello del paper con o senza fisica sul ramo del cuscinetto.

    python train.py --physics    --case 4
    python train.py --no-physics --case 4

Pipeline (Yucesan & Viana, IJPHM 11(1), 2020):

  1. piano lineare di partenza (eq. 11), scelto da --case fra i 10 del paper
  2. MLP addestrato sul piano                      RMSprop lr 1e-2, 500 epoche
  3. rifinitura dell'MLP dentro la RNN di accumulo RMSprop lr 5e-4,  50 epoche
     loss = MSE mascherato sulle 6 ispezioni del grasso (eq. 10)  ->  d_GRS
  4. dal grasso previsto si ricava il danno di fatica del cuscinetto  ->  d_BRG

I due comandi differiscono solo nel passo 4:

  --physics     catena del catalogo SKF: kappa -> eta_c -> a_SKF -> L10 ->
                Palmgren-Miner.  Zero parametri, nessun addestramento.
  --no-physics  un MLP con gli stessi ingressi calcola l'incremento di danno al
                posto della catena SKF, addestrato contro la curva di danno vera.

I passi 1-3 sono identici nei due casi: il grasso non dipende dalla scelta.

SPLIT: le turbine sono 14.  Le prime 10 stanno in *_6Months.csv, le altre quattro
(Turbine11..14) in *_6Months_Val_adv.csv, gia' tenute fuori dal training nel
codice originale.  Di default val = 11,12 e test = 13,14, il resto training;
--turbine-val e --turbine-test accettano qualunque sottoinsieme di 1..14.  Il
danno del grasso e' misurato per tutte, quindi val e test sono valutazioni vere.

NOTA SUI DATI: la verita' sul danno del cuscinetto esiste per una sola turbina,
in True_FatigueDamage_6Months_bsc.csv, e il file non dice quale.  Dando in pasto
alla catena SKF il grasso reale di ognuna delle 10 turbine, il danno finale
combacia solo per la turbina 8 (+0.35%); la piu' vicina dopo di lei e' la 7, che
sbaglia del -8.7%, e la turbina 1 del +43.4%.  Da qui il default --turbina 8.
Il passo 4 quindi non ha uno split: c'e' una sola curva di verita', usata per
addestrare (--no-physics) o solo per valutare (--physics).  Con il default la
turbina 8 e' anche in training del grasso, quindi il suo d_GRS non e' su dati
non visti; passarla a --turbine-test la rende un test pulito anche li'.

FRAZIONI: --frac riduce i dati di training e ripete l'intera pipeline una volta
per percentuale, per le curve di data efficiency:

    python train.py --physics --case 4 --frac 20,40,60,80,100

La percentuale si applica al numero di turbine di training (20% di 10 = 2), che
tengono tutte e sei le ispezioni.  I sottoinsiemi sono annidati -- le 2 turbine
del 20% sono dentro le 4 del 40% -- cosi' la curva misura l'effetto di aggiungere
dati e non il rumore di un ricampionamento; l'ordine dipende da --seed.  Val e
test non cambiano mai, altrimenti i punti non sarebbero confrontabili.

Il ramo cuscinetto non viene toccato da --frac: ha una sola curva di verita', che
resta intera.  Quello che la curva mostra e' come la qualita' del grasso si
propaga al danno del cuscinetto -- con --physics e' l'unico effetto in gioco,
visto che la catena SKF non si addestra.

Ogni run scrive in runs/<timestamp>_case<N>_<physics|nophysics>/:

  config.json              tutti gli argomenti, lo split e le ispezioni usate
  metrics.json             per ogni split e per ogni ramo: rmse, mae, rmse_rel
                           (normalizzato sull'escursione del vero) e r2
  loss_history.csv         fase, epoch, train_loss, val_loss, test_loss
  grease_predictions.csv   per turbina e ispezione: reale, previsto, split
  bearing_predictions.csv  la curva di danno del cuscinetto: reale, previsto
  models/                  modelli salvati (.keras) e pesi della RNN
  plots/                   grafici di riepilogo

Con piu' di una frazione le cartelle finiscono dentro un unico
runs/<timestamp>_case<N>_<metodo>_sweep/pct<P>/, accanto a data_efficiency.csv e
al suo grafico.  Il csv viene riscritto dopo ogni percentuale, quindi uno sweep
interrotto a meta' conserva i punti gia' fatti.
"""
import argparse, datetime, json, os, sys, time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, RNN
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pinn.layers import CumulativeDamageCell, getScalingDenseLayer

QUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(QUI, '..', 'data')
TABLES = os.path.join(QUI, '..', 'tables')
sys.path.insert(0, os.path.join(QUI, '..', 'basic'))
from models_and_functions import create_pinn_model  # noqa: E402

DTYPE = 'float32'
PASSI_AL_MESE = 6 * 24 * 30   # sei mesi campionati ogni 10 minuti
N_FLOTTA = 10                 # turbine in *_6Months.csv; le altre 4 sono _Val_adv


# =============================================================================
#   dati
# =============================================================================

def csv(nome, **kw):
    return pd.read_csv(os.path.join(DATA, nome), index_col=None, **kw).dropna()


def indici_ispezione(n_passi):
    """Le 6 ispezioni del grasso, una al mese, l'ultima sull'ultimo passo."""
    if n_passi >= 6 * PASSI_AL_MESE:
        return np.asarray([PASSI_AL_MESE * k for k in range(1, 6)] +
                          [6 * PASSI_AL_MESE - 1])
    return np.linspace(n_passi / 6.0, n_passi - 1, 6).astype(int)   # --max-passi


def carica_flotta(max_passi):
    """Le 14 turbine come (14, n_passi).  Cicli e verita' solo per le prime 10."""
    def coppia(base, val):
        a = np.transpose(np.asarray(csv(base)))
        b = np.transpose(np.asarray(csv(val)))
        return np.concatenate([a, b], axis=0)

    carico = coppia('DynamicLoad_6Months.csv', 'DynamicLoad_6Months_Val_adv.csv')
    temp = coppia('BearingTemp_6Months.csv', 'BearingTemp_6Months_Val_adv.csv')
    grasso = coppia('ViscDamage_6Months.csv', 'ViscDamage_6Months_Val_adv.csv')
    cicli = np.transpose(np.asarray(csv('Cycles_6Months_bsc.csv')))
    # la verita' sul cuscinetto ha un passo in meno di tutto il resto
    danno_vero = np.asarray(csv('True_FatigueDamage_6Months_bsc.csv',
                                header=None)).flatten()

    n = min(carico.shape[1], danno_vero.shape[0])
    if max_passi:
        n = min(n, max_passi)
    return dict(carico=carico[:, :n], temp=temp[:, :n], grasso=grasso[:, :n],
                cicli=cicli[:, :n], danno_vero=danno_vero[:n], n_passi=n)


def elenco_turbine(testo, nome):
    if not testo.strip():
        return []
    try:
        v = sorted({int(x) for x in testo.split(',')})
    except ValueError:
        raise SystemExit('%s: attese turbine separate da virgola, non %r' % (nome, testo))
    fuori = [t for t in v if not 1 <= t <= 14]
    if fuori:
        raise SystemExit('%s: turbine fuori da 1..14: %s' % (nome, fuori))
    return v


def elenco_frazioni(testo):
    try:
        v = [float(x) for x in testo.split(',') if x.strip()]
    except ValueError:
        raise SystemExit('--frac: attese percentuali separate da virgola, non %r' % testo)
    fuori = [x for x in v if not 0 < x <= 100]
    if fuori:
        raise SystemExit('--frac: percentuali fuori da (0, 100]: %s' % fuori)
    return v or [100.0]


def sottoinsieme_training(turbine, pct, seed):
    """Le prime round(pct%) turbine di un ordine fissato dal seed.

    Prendere un prefisso di un ordine unico rende i sottoinsiemi annidati: il 20%
    e' contenuto nel 40%, che e' contenuto nel 60%.  Cosi' due punti vicini della
    curva differiscono solo per le turbine aggiunte.
    """
    ordine = list(turbine)
    np.random.default_rng(0 if seed is None else seed).shuffle(ordine)
    n = max(1, int(round(len(ordine) * pct / 100.0)))
    return sorted(ordine[:n])


def prepara_split(A):
    """{'train': [...], 'val': [...], 'test': [...]} con i numeri di turbina."""
    val = elenco_turbine(A.turbine_val, '--turbine-val')
    test = elenco_turbine(A.turbine_test, '--turbine-test')
    doppie = set(val) & set(test)
    if doppie:
        raise SystemExit('turbine sia in val che in test: %s' % sorted(doppie))
    train = [t for t in range(1, 15) if t not in val and t not in test]
    if not train:
        raise SystemExit('nessuna turbina rimasta per il training')
    return {'train': train, 'val': val, 'test': test}


def ingressi_grasso(dati, turbine):
    """(n_turbine, n_passi, 2) = inverso del carico e temperatura."""
    i = [t - 1 for t in turbine]
    return np.dstack((1.0 / dati['carico'][i], dati['temp'][i])).astype(DTYPE)


def bersaglio_grasso(dati, turbine, ispezioni):
    """(n_turbine, 6, 1): il danno del grasso misurato alle sei ispezioni."""
    i = [t - 1 for t in turbine]
    return dati['grasso'][i][:, ispezioni][:, :, None].astype(DTYPE)


# =============================================================================
#   passi 1-3: il grasso
# =============================================================================

def metriche_regressione(vero, previsto):
    """Le quattro metriche con cui valutiamo qualunque ramo.

    rmse e mae sono in scala assoluta: confrontabili fra run sullo stesso
    dataset, muti presi da soli.  rmse_rel li rende leggibili normalizzando
    sull'escursione del valore vero, e r2 dice quanta della sua varianza e'
    spiegata (1 = perfetto, 0 = tanto vale prevedere la media).
    """
    vero = np.asarray(vero, dtype='float64').ravel()
    previsto = np.asarray(previsto, dtype='float64').ravel()
    err = previsto - vero
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((vero - vero.mean()) ** 2))
    escursione = float(np.ptp(vero))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {'rmse': rmse,
            'mae': float(np.mean(np.abs(err))),
            'rmse_rel': rmse / escursione if escursione > 0 else float('nan'),
            'r2': 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')}


def costruisci_mlp_grasso(piano):
    """MLP del passo 2: (Dkappa, 1/P, T) -> incremento di danno normalizzato."""
    ingressi = piano[['Dkappa', 'dynamicLoads', 'bearingTemp']]
    minimi = ingressi.min(axis=0)
    ampiezza = ingressi.max(axis=0) - minimi
    return Sequential([
        getScalingDenseLayer(minimi, ampiezza),
        Dense(40, activation='sigmoid'),
        Dense(20, activation='elu'),
        Dense(10, activation='elu'),
        Dense(5, activation='elu'),
        Dense(1, activation='sigmoid'),
    ], name='mlp_grasso')


def perdita_mascherata(ispezioni):
    """MSE sulle sole 6 ispezioni (eq. 10): il resto della serie non e' misurato."""
    idx = tf.constant(ispezioni, dtype='int32')

    def perdita(y_true, y_pred):
        return tf.keras.losses.mean_squared_error(
            y_true, tf.gather(y_pred, idx, axis=1))
    return perdita


def costruisci_rnn_grasso(mlp, forma_ingressi, basso, alto, ispezioni, lr):
    """Passo 3: l'MLP diventa la cella di una RNN che accumula il danno.

    L'oggetto ``mlp`` e' condiviso fra le RNN di train, val e test: i tre modelli
    hanno batch_input_shape diversi (numero di turbine diverso) ma gli stessi
    pesi, quindi valutare val e test non richiede di ricopiare niente.
    """
    d0 = np.zeros((forma_ingressi[0], 1), dtype=DTYPE)
    segnaposto = Input(shape=(forma_ingressi[2] + 1,))   # stato + ingressi
    riscalata = Lambda(lambda x: x * (alto - basso) + basso)(mlp(segnaposto))
    cella = CumulativeDamageCell(model=Model(inputs=[segnaposto], outputs=[riscalata]),
                                 batch_input_shape=forma_ingressi,
                                 dtype=DTYPE, initial_damage=d0)
    modello = Sequential([RNN(cell=cella, return_sequences=True,
                              return_state=False, batch_input_shape=forma_ingressi)])
    perdita = perdita_mascherata(ispezioni)
    modello.compile(loss=perdita, optimizer=RMSprop(lr), metrics=[perdita])
    return modello


class ValutaSplit(Callback):
    """Aggiunge val_loss e test_loss a ogni epoca valutando i modelli gemelli."""

    def __init__(self, gemelli):
        super().__init__()
        self.gemelli = gemelli    # [(nome, modello, X, y), ...]

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        for nome, modello, X, y in self.gemelli:
            logs['%s_loss' % nome] = float(modello.evaluate(X, y, verbose=0)[0])


def storia(fase, hist):
    """history di Keras -> righe di loss_history.csv."""
    df = pd.DataFrame({'fase': fase, 'epoch': hist.epoch})
    for col, chiave in (('train_loss', 'loss'), ('val_loss', 'val_loss'),
                        ('test_loss', 'test_loss')):
        df[col] = hist.history.get(chiave, np.nan)
    return df


def addestra_grasso(A, dati, split, out, righe_loss):
    """Passi 1-3.  Restituisce il Dkappa previsto per tutte e 14 le turbine."""
    piano = pd.read_csv(os.path.join(
        DATA, 'random_plane_set_500_adv_case%d.csv' % A.case))
    colonne = ['Dkappa', 'dynamicLoads', 'bearingTemp']

    print('\n[2/4] MLP sul piano del caso %d  (%d epoche)' % (A.case, A.mlp_epochs))
    mlp = costruisci_mlp_grasso(piano)
    mlp.compile(loss='mse', optimizer=RMSprop(1e-2), metrics=['mae'])
    uscite = piano[['delDkappa']]
    basso, alto = float(uscite.min().iloc[0]), float(uscite.max().iloc[0])
    # run02 valida il passo 2 contro il DOE vero: stesso set, stessa normalizzazione
    vero_doe = pd.read_csv(os.path.join(DATA, 'true_set_500_adv.csv'))
    h = mlp.fit(piano[colonne], (uscite - basso) / (alto - basso),
                validation_data=(vero_doe[colonne],
                                 (vero_doe[['delDkappa']] - basso) / (alto - basso)),
                epochs=A.mlp_epochs, verbose=2,
                callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.85, min_lr=1e-15,
                                             patience=30, verbose=0, mode='min')])
    righe_loss.append(storia('mlp_piano', h))
    mlp.save(os.path.join(out, 'models', 'mlp_grasso.keras'))

    ispezioni = indici_ispezione(dati['n_passi'])
    forma = {s: ingressi_grasso(dati, t) for s, t in split.items() if t}
    bersaglio = {s: bersaglio_grasso(dati, t, ispezioni)
                 for s, t in split.items() if t}
    modelli = {s: costruisci_rnn_grasso(mlp, X.shape, np.asarray([basso]),
                                        np.asarray([alto]), ispezioni, 5e-4)
               for s, X in forma.items()}

    prima = modelli['train'].predict(forma['train'], verbose=0)

    print('\n[3/4] rifinitura dentro la RNN di accumulo  (%d epoche)' % A.rnn_epochs)
    gemelli = [(s, modelli[s], forma[s], bersaglio[s])
               for s in ('val', 'test') if s in modelli]
    h = modelli['train'].fit(
        forma['train'], bersaglio['train'], epochs=A.rnn_epochs, verbose=2,
        steps_per_epoch=1,
        callbacks=[ValutaSplit(gemelli),
                   ReduceLROnPlateau(monitor='loss', factor=0.7, min_lr=1e-15,
                                     patience=5, verbose=0, mode='min')])
    righe_loss.append(storia('rnn_grasso', h))
    modelli['train'].save_weights(os.path.join(out, 'models', 'rnn_grasso'))

    # previsioni finali, split per split
    righe, metriche = [], {}
    for s, modello in modelli.items():
        y = modello.predict(forma[s], verbose=0)[:, ispezioni, 0]
        v = bersaglio[s][:, :, 0]
        for k, turbina in enumerate(split[s]):
            for j, passo in enumerate(ispezioni):
                righe.append({'split': s, 'turbina': turbina, 'ispezione': j + 1,
                              'passo': int(passo), 'reale': float(v[k, j]),
                              'previsto': float(y[k, j])})
        for nome, valore in metriche_regressione(v, y).items():
            metriche['grasso_%s_%s' % (s, nome)] = valore
    metriche['grasso_train_rmse_prima'] = float(np.sqrt(np.mean(
        (prima[:, ispezioni, 0] - bersaglio['train'][:, :, 0]) ** 2)))
    pd.DataFrame(righe).to_csv(os.path.join(out, 'grease_predictions.csv'), index=False)

    grafico_grasso(righe, out)

    # il passo 4 puo' chiedere una turbina qualsiasi: prevedo tutte e 14 in blocco
    tutte = ingressi_grasso(dati, list(range(1, 15)))
    completo = costruisci_rnn_grasso(mlp, tutte.shape, np.asarray([basso]),
                                     np.asarray([alto]), ispezioni, 5e-4)
    return completo.predict(tutte, verbose=0)[:, :, 0], ispezioni, metriche


# =============================================================================
#   passo 4: il cuscinetto
# =============================================================================

def disponi_tabella(tabella):
    """Tabella del catalogo -> griglia, estremi e forma per TableInterpolation.

    Come arrange_table di basic/models_and_functions.py, ma senza costruire
    l'array ragged [righe, colonne]: numpy >= 1.24 lo rifiuta quando la tabella
    non e' quadrata, ed erano comunque serviti solo i minimi e i massimi.
    """
    dati = np.transpose(np.asarray(np.transpose(tabella))[1:])
    if dati.shape[1] == 1:
        dati = np.repeat(dati, 2, axis=1)
    dati = np.expand_dims(np.expand_dims(dati, 0), -1)
    righe = np.asarray(tabella.iloc[:, 0], dtype='float64')
    colonne = np.asarray([float(c) for c in tabella.columns[1:]])
    estremi = np.asarray([[righe.min(), colonne.min()],
                          [righe.max(), colonne.max()]])
    return {'data': dati, 'bounds': estremi, 'table_shape': dati.shape}


def ingressi_cuscinetto(dati, dkappa, turbina):
    """(1, n_passi, 4) = Dkappa previsto, cicli, log10(carico), temperatura."""
    if turbina > N_FLOTTA:
        raise SystemExit('--turbina %d: i cicli esistono solo per le turbine 1..%d'
                         % (turbina, N_FLOTTA))
    i = turbina - 1
    return np.dstack((dkappa[i:i + 1], dati['cicli'][i:i + 1],
                      np.log10(dati['carico'][i:i + 1]),
                      dati['temp'][i:i + 1])).astype(DTYPE)


def cuscinetto_fisica(ingressi):
    """Catena SKF del catalogo: nessun parametro da addestrare."""
    a1, C, Pu = 1.0, 6000.0, 750.0
    a = -10 / 3                                                 # pendenza curva SN
    b = (10 / 3) * np.log10(C) + np.log10(1e6) + np.log10(a1)   # intercetta
    tab = {n: disponi_tabella(pd.read_csv(os.path.join(TABLES, '%s.csv' % n)))
           for n in ('aSKF', 'kappa', 'etac')}
    d0 = np.zeros((ingressi.shape[0], 1), dtype=DTYPE)
    return create_pinn_model(a, b, Pu,
                             tab['aSKF']['data'], tab['aSKF']['bounds'], tab['aSKF']['table_shape'],
                             tab['kappa']['data'], tab['kappa']['bounds'], tab['kappa']['table_shape'],
                             tab['etac']['data'], tab['etac']['bounds'], tab['etac']['table_shape'],
                             d0, ingressi.shape,
                             selectdKappa=[1], selectCycle=[2], selectLoad=[3], selectBTemp=[4],
                             myDtype=DTYPE, return_sequences=True)


def cuscinetto_mlp(ingressi, incremento_max, lr):
    """L'MLP che sostituisce la catena SKF, dentro la stessa cella di accumulo.

    Stessi ingressi della catena (piu' lo stato) e stessa uscita: l'incremento di
    danno del passo.  La sigmoide lo tiene in [0, incremento_max], cosi' il danno
    accumulato resta dell'ordine di grandezza giusto anche a inizio training.
    """
    piatti = ingressi.reshape(-1, ingressi.shape[2])
    minimi = np.concatenate(([0.0], piatti.min(axis=0))).astype('float64')
    ampiezza = np.concatenate(([1.0], piatti.max(axis=0) - piatti.min(axis=0))).astype('float64')
    ampiezza[ampiezza == 0] = 1.0

    segnaposto = Input(shape=(ingressi.shape[2] + 1,))
    x = getScalingDenseLayer(minimi, ampiezza)(segnaposto)
    for n, att in ((40, 'sigmoid'), (20, 'elu'), (10, 'elu'), (5, 'elu')):
        x = Dense(n, activation=att)(x)
    x = Dense(1, activation='sigmoid')(x)
    x = Lambda(lambda y: y * incremento_max)(x)
    interno = Model(inputs=[segnaposto], outputs=[x], name='mlp_cuscinetto')

    d0 = np.zeros((ingressi.shape[0], 1), dtype=DTYPE)
    cella = CumulativeDamageCell(model=interno, batch_input_shape=ingressi.shape,
                                 dtype=DTYPE, initial_damage=d0)
    modello = Sequential([RNN(cell=cella, return_sequences=True,
                              return_state=False, batch_input_shape=ingressi.shape)])
    modello.compile(loss='mse', optimizer=RMSprop(lr), metrics=['mae'])
    return modello, interno


def addestra_cuscinetto(A, dati, dkappa, out, righe_loss):
    ingressi = ingressi_cuscinetto(dati, dkappa, A.turbina)
    vero = dati['danno_vero']
    scala = float(vero[-1])   # normalizza la curva: il danno vero e' ~1e-2

    if A.physics:
        print('\n[4/4] catena SKF  (nessun addestramento)')
        modello = cuscinetto_fisica(ingressi)
        previsto = modello.predict(ingressi, verbose=0)[0, :, 0]
    else:
        print('\n[4/4] MLP al posto della catena SKF  (%d epoche)' % A.bearing_epochs)
        # in unita' normalizzate il danno finale vale 1: l'incremento medio per
        # passo e' 1/n_passi, e ne lasciamo un margine di 4x al modello.
        modello, interno = cuscinetto_mlp(ingressi, 4.0 / dati['n_passi'], 5e-4)
        h = modello.fit(ingressi, (vero / scala)[None, :, None].astype(DTYPE),
                        epochs=A.bearing_epochs, verbose=2, steps_per_epoch=1,
                        callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.7,
                                                     min_lr=1e-15, patience=5,
                                                     verbose=0, mode='min')])
        righe_loss.append(storia('rnn_cuscinetto', h))
        interno.save(os.path.join(out, 'models', 'mlp_cuscinetto.keras'))
        modello.save_weights(os.path.join(out, 'models', 'rnn_cuscinetto'))
        previsto = modello.predict(ingressi, verbose=0)[0, :, 0] * scala

    pd.DataFrame({'passo': np.arange(len(vero)), 'turbina': A.turbina,
                  'reale': vero, 'previsto': previsto}
                 ).to_csv(os.path.join(out, 'bearing_predictions.csv'), index=False)
    grafico_cuscinetto(vero, previsto, A.physics, out)
    m = {'cuscinetto_turbina': A.turbina,
         'cuscinetto_in_sample': not A.physics,
         'cuscinetto_danno_finale_vero': float(vero[-1]),
         'cuscinetto_danno_finale_previsto': float(previsto[-1]),
         'cuscinetto_errore_relativo_finale': float((previsto[-1] - vero[-1]) / vero[-1])}
    for nome, valore in metriche_regressione(vero, previsto).items():
        m['cuscinetto_curva_%s' % nome] = valore
    return m


# =============================================================================
#   grafici
# =============================================================================

COLORI = {'train': '0.55', 'val': 'C0', 'test': 'C3'}


def grafico_grasso(righe, out):
    df = pd.DataFrame(righe)
    plt.figure(figsize=(5, 4.5))
    lo, hi = df[['reale', 'previsto']].min().min(), df[['reale', 'previsto']].max().max()
    for s, g in df.groupby('split'):
        plt.plot(g['reale'], g['previsto'], 'o', c=COLORI.get(s, 'k'),
                 label='%s (%d turbine)' % (s, g['turbina'].nunique()))
    plt.plot([lo, hi], [lo, hi], '--k')
    plt.xlabel('danno del grasso misurato')
    plt.ylabel('previsto')
    plt.title('Grasso: le 6 ispezioni')
    plt.grid(which='both'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out, 'plots', 'grasso_previsto_vs_reale.png'), dpi=120)
    plt.close()


def grafico_loss(righe_loss, out):
    df = pd.concat(righe_loss, ignore_index=True)
    fasi = list(dict.fromkeys(df['fase']))
    fig, assi = plt.subplots(1, len(fasi), figsize=(4.5 * len(fasi), 3.8), squeeze=False)
    for ax, fase in zip(assi[0], fasi):
        g = df[df['fase'] == fase]
        for col, nome in (('train_loss', 'train'), ('val_loss', 'val'), ('test_loss', 'test')):
            if col in g and g[col].notna().any():
                ax.semilogy(g['epoch'], g[col], c=COLORI[nome], label=nome)
        ax.set_title(fase); ax.set_xlabel('epoca'); ax.set_ylabel('loss')
        ax.grid(which='both'); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'plots', 'loss.png'), dpi=120)
    plt.close()
    return df


def grafico_cuscinetto(vero, previsto, fisica, out):
    plt.figure(figsize=(6, 4.5))
    plt.plot(vero, 'k-', label='danno reale')
    plt.plot(previsto, '--', c='C0' if fisica else 'C3',
             label='previsto (%s)' % ('catena SKF' if fisica else 'MLP'))
    plt.xlabel('passo (10 minuti)')
    plt.ylabel('danno di fatica del cuscinetto')
    plt.title('Cuscinetto: %s' % ('catena SKF' if fisica else 'MLP al posto della catena SKF'))
    plt.grid(which='both'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out, 'plots', 'cuscinetto_danno.png'), dpi=120)
    plt.close()


# =============================================================================
#   main
# =============================================================================

def esegui_uno(A, dati, split, pct, out):
    """Una pipeline completa (passi 1-4) su una frazione delle turbine di training."""
    os.makedirs(os.path.join(out, 'models'), exist_ok=True)
    os.makedirs(os.path.join(out, 'plots'), exist_ok=True)

    print('\n[1/4] caso %d, ramo cuscinetto: %s%s' %
          (A.case, 'catena SKF' if A.physics else 'MLP',
           '' if pct == 100 else '  |  %g%% dei dati di training' % pct))
    print('      %d passi | train %s | val %s | test %s' %
          (dati['n_passi'], split['train'], split['val'] or '-', split['test'] or '-'))

    t0 = time.time()
    righe_loss = []
    dkappa, ispezioni, metriche = addestra_grasso(A, dati, split, out, righe_loss)
    metriche.update(addestra_cuscinetto(A, dati, dkappa, out, righe_loss))
    metriche['secondi'] = round(time.time() - t0, 1)

    grafico_loss(righe_loss, out).to_csv(
        os.path.join(out, 'loss_history.csv'), index=False)
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump({**vars(A), 'metodo': 'physics' if A.physics else 'nophysics',
                   'n_passi': dati['n_passi'], 'split': split,
                   'train_pct': pct, 'n_turbine_train': len(split['train']),
                   'ispezioni': ispezioni.tolist()}, f, indent=2)
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump(metriche, f, indent=2)

    print('\n--- fatto in %.1fs -> %s' % (metriche['secondi'], out))
    for k, v in metriche.items():
        print('    %-38s %s' % (k, v))
    return metriche


def grafico_data_efficiency(df, out):
    colonne = [c for c in ('grasso_train_rmse', 'grasso_val_rmse', 'grasso_test_rmse')
               if c in df]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for c in colonne:
        a1.plot(df['train_pct'], df[c], 'o-',
                c=COLORI[c.split('_')[1]], label=c.split('_')[1])
    a1.set_xlabel('% delle turbine di training'); a1.set_ylabel('RMSE del grasso')
    a1.set_title('Grasso alle 6 ispezioni'); a1.grid(which='both'); a1.legend()

    a2.plot(df['train_pct'], 100 * df['cuscinetto_errore_relativo_finale'].abs(), 'ko-')
    a2.set_xlabel('% delle turbine di training')
    a2.set_ylabel('|errore| sul danno finale  [%]')
    a2.set_title('Cuscinetto a 6 mesi'); a2.grid(which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'plots', 'data_efficiency.png'), dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--physics', action='store_true',
                   help='passo 4 con la catena SKF del catalogo')
    g.add_argument('--no-physics', dest='nophysics', action='store_true',
                   help='passo 4 con un MLP al posto della catena SKF')
    ap.add_argument('--case', type=int, default=3, choices=range(1, 11), metavar='N',
                    help='piano di partenza, 1..10 (default: 3)')
    ap.add_argument('--frac', default='100', metavar='P',
                    help='percentuali di turbine di training, separate da virgola: '
                         'un training per percentuale (default: 100)')
    ap.add_argument('--turbine-val', default='11,12', metavar='LISTA',
                    help='turbine di validazione (default: 11,12)')
    ap.add_argument('--turbine-test', default='13,14', metavar='LISTA',
                    help='turbine di test (default: 13,14)')
    ap.add_argument('--turbina', type=int, default=8, choices=range(1, 11), metavar='N',
                    help="turbina con la verita' sul cuscinetto (default: 8)")
    ap.add_argument('--mlp-epochs', type=int, default=500, help='passo 2 (default: 500)')
    ap.add_argument('--rnn-epochs', type=int, default=50, help='passo 3 (default: 50)')
    ap.add_argument('--bearing-epochs', type=int, default=300,
                    help='passo 4, solo con --no-physics (default: 300)')
    ap.add_argument('--max-passi', type=int, default=0, metavar='N',
                    help='accorcia le serie a N passi: serve solo per uno smoke test')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--outdir', default=os.path.join(QUI, 'runs'))
    A = ap.parse_args()

    if A.seed is not None:
        np.random.seed(A.seed)
        tf.random.set_seed(A.seed)

    frazioni = elenco_frazioni(A.frac)
    split = prepara_split(A)
    dati = carica_flotta(A.max_passi)
    metodo = 'physics' if A.physics else 'nophysics'
    stampa = '%s_case%d_%s' % (datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
                               A.case, metodo)

    if len(frazioni) == 1 and frazioni[0] == 100:
        esegui_uno(A, dati, split, 100.0, os.path.join(A.outdir, stampa))
        return

    radice = os.path.join(A.outdir, stampa + '_sweep')
    os.makedirs(os.path.join(radice, 'plots'), exist_ok=True)
    riassunto = []
    for pct in frazioni:
        turbine = sottoinsieme_training(split['train'], pct, A.seed)
        parziale = dict(split, train=turbine)
        print('\n' + '=' * 70)
        print('  %g%%  ->  %d turbine su %d: %s' %
              (pct, len(turbine), len(split['train']), turbine))
        print('=' * 70)
        m = esegui_uno(A, dati, parziale, pct,
                       os.path.join(radice, 'pct%g' % pct))
        riassunto.append({'train_pct': pct, 'n_turbine_train': len(turbine),
                          'turbine_train': ' '.join(map(str, turbine)), **m})
        # riscritto a ogni giro: uno sweep interrotto conserva i punti gia' fatti
        df = pd.DataFrame(riassunto)
        df.to_csv(os.path.join(radice, 'data_efficiency.csv'), index=False)
        grafico_data_efficiency(df, radice)

    print('\n' + '=' * 70)
    print('  sweep completo -> %s' % radice)
    print(df[['train_pct', 'n_turbine_train', 'grasso_test_rmse',
              'cuscinetto_errore_relativo_finale']].to_string(index=False))


if __name__ == '__main__':
    main()
