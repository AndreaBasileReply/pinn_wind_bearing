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

NOTA SUI DATI: la verita' sul danno del cuscinetto esiste per una sola turbina,
in True_FatigueDamage_6Months_bsc.csv, e il file non dice quale.  Dando in pasto
alla catena SKF il grasso reale di ognuna delle 10 turbine, il danno finale
combacia solo per la turbina 8 (+0.35%); la piu' vicina dopo di lei e' la 7, che
sbaglia del -8.7%, e la turbina 1 del +43.4%.  Da qui il default --turbina 8.
Quella turbina fa parte anche del training del grasso: la valutazione del ramo
cuscinetto non e' su dati non visti, ed e' l'unica possibile con questo dataset.

Ogni run scrive in runs/<timestamp>_case<N>_<physics|nophysics>/ la
configurazione, le metriche, i pesi e i grafici.
"""
import argparse, datetime, json, os, sys, time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, RNN
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau

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

# I sei mesi campionati ogni 10 minuti: 6*24*30 passi al mese.
PASSI_AL_MESE = 6 * 24 * 30


# =============================================================================
#   dati
# =============================================================================

def indici_ispezione(n_passi):
    """Le 6 ispezioni del grasso, una al mese, l'ultima sull'ultimo passo."""
    if n_passi >= 6 * PASSI_AL_MESE:
        return np.asarray([PASSI_AL_MESE * k for k in range(1, 6)] +
                          [6 * PASSI_AL_MESE - 1])
    # orizzonte accorciato (--max-passi): sei punti equispaziati
    return np.linspace(n_passi / 6.0, n_passi - 1, 6).astype(int)


def csv(nome, **kw):
    return pd.read_csv(os.path.join(DATA, nome), index_col=None, **kw).dropna()


def carica_flotta(max_passi):
    """Serie a 6 mesi delle 10 turbine, come (n_turbine, n_passi)."""
    carico = np.transpose(np.asarray(csv('DynamicLoad_6Months.csv')))
    temp = np.transpose(np.asarray(csv('BearingTemp_6Months.csv')))
    grasso = np.transpose(np.asarray(csv('ViscDamage_6Months.csv')))
    cicli = np.transpose(np.asarray(csv('Cycles_6Months_bsc.csv')))
    # la verita' sul cuscinetto ha un passo in meno di tutto il resto
    danno_vero = np.asarray(csv('True_FatigueDamage_6Months_bsc.csv',
                                header=None)).flatten()

    n = min(carico.shape[1], danno_vero.shape[0])
    if max_passi:
        n = min(n, max_passi)
    return dict(carico=carico[:, :n], temp=temp[:, :n], grasso=grasso[:, :n],
                cicli=cicli[:, :n], danno_vero=danno_vero[:n], n_passi=n)


# =============================================================================
#   passi 1-3: il grasso
# =============================================================================

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


def costruisci_rnn_grasso(mlp, d0, forma_ingressi, basso, alto, ispezioni, lr):
    """Passo 3: l'MLP diventa la cella di una RNN che accumula il danno."""
    segnaposto = Input(shape=(forma_ingressi[2] + 1,))  # stato + ingressi
    uscita = mlp(segnaposto)
    riscalata = Lambda(lambda x: x * (alto - basso) + basso)(uscita)
    cella = CumulativeDamageCell(model=Model(inputs=[segnaposto], outputs=[riscalata]),
                                 batch_input_shape=forma_ingressi,
                                 dtype=DTYPE, initial_damage=d0)
    modello = Sequential([RNN(cell=cella, return_sequences=True,
                              return_state=False, batch_input_shape=forma_ingressi)])
    perdita = perdita_mascherata(ispezioni)
    modello.compile(loss=perdita, optimizer=RMSprop(lr), metrics=[perdita])
    return modello


def addestra_grasso(A, dati, out):
    """Passi 1-3.  Restituisce il Dkappa previsto per l'intera flotta."""
    piano = pd.read_csv(os.path.join(
        DATA, 'random_plane_set_500_adv_case%d.csv' % A.case))

    print('\n[2/4] MLP sul piano del caso %d  (%d epoche)' % (A.case, A.mlp_epochs))
    mlp = costruisci_mlp_grasso(piano)
    mlp.compile(loss='mse', optimizer=RMSprop(1e-2), metrics=['mae'])
    uscite = piano[['delDkappa']]
    basso, alto = float(uscite.min().iloc[0]), float(uscite.max().iloc[0])
    mlp.fit(piano[['Dkappa', 'dynamicLoads', 'bearingTemp']],
            (uscite - basso) / (alto - basso),
            epochs=A.mlp_epochs, verbose=2,
            callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.85, min_lr=1e-15,
                                         patience=30, verbose=0, mode='min')])

    ingressi = np.dstack((1.0 / dati['carico'], dati['temp'])).astype(DTYPE)
    d0 = np.zeros((ingressi.shape[0], 1), dtype=DTYPE)
    ispezioni = indici_ispezione(dati['n_passi'])
    bersaglio = dati['grasso'][:, ispezioni][:, :, None].astype(DTYPE)

    rnn = costruisci_rnn_grasso(mlp, d0, ingressi.shape,
                                np.asarray([basso]), np.asarray([alto]),
                                ispezioni, 5e-4)
    prima = rnn.predict(ingressi, verbose=0)

    print('\n[3/4] rifinitura dentro la RNN di accumulo  (%d epoche)' % A.rnn_epochs)
    rnn.fit(ingressi, bersaglio, epochs=A.rnn_epochs, verbose=2, steps_per_epoch=1,
            callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.7, min_lr=1e-15,
                                         patience=5, verbose=0, mode='min')])
    dopo = rnn.predict(ingressi, verbose=0)
    rnn.save_weights(os.path.join(out, 'models', 'rnn_grasso'))

    mse_prima = float(np.mean((prima[:, ispezioni, 0] - bersaglio[:, :, 0]) ** 2))
    mse_dopo = float(np.mean((dopo[:, ispezioni, 0] - bersaglio[:, :, 0]) ** 2))
    grafico_grasso(bersaglio[:, :, 0], prima[:, ispezioni, 0], dopo[:, ispezioni, 0], out)
    return dopo[:, :, 0], ispezioni, {'grasso_mse_ispezioni_prima': mse_prima,
                                      'grasso_mse_ispezioni_dopo': mse_dopo}


# =============================================================================
#   passo 4: il cuscinetto
# =============================================================================

def ingressi_cuscinetto(dati, dkappa, turbina):
    """(1, n_passi, 4) = Dkappa previsto, cicli, log10(carico), temperatura."""
    i = turbina - 1
    return np.dstack((dkappa[i:i + 1], dati['cicli'][i:i + 1],
                      np.log10(dati['carico'][i:i + 1]),
                      dati['temp'][i:i + 1])).astype(DTYPE)


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


def cuscinetto_fisica(ingressi):
    """Catena SKF del catalogo: nessun parametro da addestrare."""
    a1, C, Pu = 1.0, 6000.0, 750.0
    a = -10 / 3                                             # pendenza curva SN
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

    d0 = np.zeros((ingressi.shape[0], 1), dtype=DTYPE)
    cella = CumulativeDamageCell(model=Model(inputs=[segnaposto], outputs=[x]),
                                 batch_input_shape=ingressi.shape,
                                 dtype=DTYPE, initial_damage=d0)
    modello = Sequential([RNN(cell=cella, return_sequences=True,
                              return_state=False, batch_input_shape=ingressi.shape)])
    modello.compile(loss='mse', optimizer=RMSprop(lr), metrics=['mae'])
    return modello


def addestra_cuscinetto(A, dati, dkappa, out):
    ingressi = ingressi_cuscinetto(dati, dkappa, A.turbina)
    vero = dati['danno_vero']
    scala = float(vero[-1])   # normalizza la curva: il danno vero e' ~1e-6

    if A.physics:
        print('\n[4/4] catena SKF  (nessun addestramento)')
        modello = cuscinetto_fisica(ingressi)
        previsto = modello.predict(ingressi, verbose=0)[0, :, 0]
    else:
        print('\n[4/4] MLP al posto della catena SKF  (%d epoche)' % A.bearing_epochs)
        # in unita' normalizzate il danno finale vale ~1: l'incremento medio per
        # passo e' 1/n_passi, e ne lasciamo un margine di 4x al modello.
        modello = cuscinetto_mlp(ingressi, 4.0 / dati['n_passi'], 5e-4)
        modello.fit(ingressi, (vero / scala)[None, :, None].astype(DTYPE),
                    epochs=A.bearing_epochs, verbose=2, steps_per_epoch=1,
                    callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.7, min_lr=1e-15,
                                                 patience=5, verbose=0, mode='min')])
        modello.save_weights(os.path.join(out, 'models', 'mlp_cuscinetto'))
        previsto = modello.predict(ingressi, verbose=0)[0, :, 0] * scala

    grafico_cuscinetto(vero, previsto, A.physics, out)
    return {'cuscinetto_danno_finale_vero': float(vero[-1]),
            'cuscinetto_danno_finale_previsto': float(previsto[-1]),
            'cuscinetto_errore_relativo_finale': float((previsto[-1] - vero[-1]) / vero[-1]),
            'cuscinetto_rmse_curva': float(np.sqrt(np.mean((previsto - vero) ** 2)))}


# =============================================================================
#   grafici
# =============================================================================

def grafico_grasso(vero, prima, dopo, out):
    plt.figure(figsize=(5, 4.5))
    lo = min(vero.min(), prima.min(), dopo.min())
    hi = max(vero.max(), prima.max(), dopo.max())
    plt.plot(vero.flatten(), prima.flatten(), 'o', c='0.6', label='prima del training')
    plt.plot(vero.flatten(), dopo.flatten(), 'ro', label='dopo il training')
    plt.plot([lo, hi], [lo, hi], '--k')
    plt.xlabel('danno del grasso misurato')
    plt.ylabel('previsto')
    plt.title('Grasso: le 6 ispezioni, 10 turbine')
    plt.grid(which='both'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out, 'plots', 'grasso_previsto_vs_reale.png'), dpi=120)
    plt.close()


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

    metodo = 'physics' if A.physics else 'nophysics'
    nome = '%s_case%d_%s' % (datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
                             A.case, metodo)
    out = os.path.join(A.outdir, nome)
    os.makedirs(os.path.join(out, 'models'), exist_ok=True)
    os.makedirs(os.path.join(out, 'plots'), exist_ok=True)

    print('\n[1/4] caso %d, ramo cuscinetto: %s' %
          (A.case, 'catena SKF' if A.physics else 'MLP'))
    dati = carica_flotta(A.max_passi)
    print('      %d turbine x %d passi' % (dati['carico'].shape[0], dati['n_passi']))

    t0 = time.time()
    dkappa, ispezioni, m_grasso = addestra_grasso(A, dati, out)
    metriche = dict(m_grasso)
    metriche.update(addestra_cuscinetto(A, dati, dkappa, out))
    metriche['secondi'] = round(time.time() - t0, 1)

    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump({**vars(A), 'metodo': metodo, 'n_passi': dati['n_passi'],
                   'ispezioni': ispezioni.tolist()}, f, indent=2)
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump(metriche, f, indent=2)

    print('\n--- fatto in %.1fs -> %s' % (metriche['secondi'], out))
    for k, v in metriche.items():
        print('    %-38s %s' % (k, v))


if __name__ == '__main__':
    main()
