# -*- coding: utf-8 -*-
"""Data efficiency comparison: MSE in funzione della percentuale di dati di training.

    python compare_data_efficiency.py SORGENTE_A [SORGENTE_B] [opzioni]

Ogni SORGENTE e' una delle due cose, riconosciuta da sola:

  un loss_on_percentage.csv   il file cumulativo che train.py aggiorna a ogni run
  una cartella runs/          i run vengono letti dai loro metrics.json

Se la prima sorgente contiene i due metodi (il caso tipico di un
loss_on_percentage.csv scritto con lo stesso --outdir per fisica e non-fisica),
la seconda si puo' omettere: lo script divide le curve da se'.

    python compare_data_efficiency.py runs/loss_on_percentage.csv
    python compare_data_efficiency.py A/loss_on_percentage.csv B/loss_on_percentage.csv
    python compare_data_efficiency.py runs_physics runs_nophysics

cioe' PINN (catena SKF al passo 4) contro l'MLP che la sostituisce.

DA DOVE VIENE LA PERCENTUALE (nell'ordine, il primo che risponde vince):

  1. config.json, una chiave fra train_frac / train_fraction / train_pct /
     data_pct / frazione / percentuale / pct  (<=1 e' letto come frazione)
  2. il nome della cartella del run: ..._pct40, ..._p40, ..._frac0.4, ..._dati40
  3. il numero di turbine presenti in grasso_predictions_train.csv, rapportato al
     dataset intero (10 turbine, vedi --full-turbines)

DA DOVE VIENE LA MSE:

  grasso      media di errore**2 su grasso_predictions_test.csv (o _train.csv con
              --split train): 6 ispezioni x turbine di validazione
  cuscinetto  media di errore**2 su cuscinetto_predictions.csv, sull'intera curva
              giornaliera (--bearing-metric curva) o sui soli 6 giorni di
              ispezione (--bearing-metric ispezioni)

Se i CSV mancano si ripiega su metrics.json (mse_test, rmse_curva**2, ...).

NOTA: nella pipeline del paper il ramo del grasso e' identico nei due comandi, e
con --physics il ramo cuscinetto non usa nemmeno un dato.  Sono le due cose che
questo grafico deve rendere visibili: la curva del grasso si sovrappone, quella
del cuscinetto resta piatta per il PINN e peggiora per l'MLP quando i dati calano.
"""
import argparse, glob, json, os, re, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLU, ARA, INK, GRID, MUTO = '#2a78d6', '#eb6834', '#0f141a', '#dce3ea', '#55636f'
PCT_TICKS = [20, 40, 60, 80, 100]
MARCATORI = ('o', 's')
GIORNI_ISP = [30, 60, 90, 120, 150, 180]        # i 6 giorni di ispezione
CHIAVI_PCT = ('train_frac', 'train_fraction', 'train_pct', 'train_percent',
              'data_frac', 'data_pct', 'frazione', 'percentuale', 'pct', 'frac')
RE_PCT = (re.compile(r'(?:pct|perc|frac|dati|data|p)[-_]?(\d+(?:\.\d+)?)', re.I),
          re.compile(r'(\d+(?:\.\d+)?)[-_]?(?:pct|perc|%)', re.I))

ETICHETTE = {'physics': 'PINN — formule SKF',
             'nophysics': 'MLP al posto delle formule SKF'}
COLORI    = {'physics': BLU, 'nophysics': ARA}


def lista_casi(v):
    """--caso 4  oppure  --caso 1,2,3,10  ->  set di interi."""
    try:
        return {int(x) for x in str(v).replace(' ', '').split(',') if x}
    except ValueError:
        raise argparse.ArgumentTypeError('casi non interi: %r' % v)


def _num_pct(v):
    """Normalizza a percentuale: 0.4 -> 40, 40 -> 40.  None se non interpretabile."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x <= 0:
        return None
    return x*100 if x <= 1 else x


def percentuale(run, cfg, met, full_turbine):
    """Percentuale di dati di training del run, e da dove e' stata ricavata."""
    for k in CHIAVI_PCT:
        if k in cfg:
            p = _num_pct(cfg[k])
            if p is not None:
                return p, 'config.json[%s]' % k
    nome = os.path.basename(os.path.normpath(run))
    for rx in RE_PCT:
        m = rx.search(nome)
        if m:
            p = _num_pct(m.group(1))
            if p is not None:
                return p, 'nome cartella'
    f = os.path.join(run, 'grasso_predictions_train.csv')
    if os.path.isfile(f):
        n = pd.read_csv(f).turbina.nunique()
        return 100.0*n/full_turbine, '%d/%d turbine' % (n, full_turbine)
    n = (met.get('grasso') or {}).get('osservazioni')
    if n:
        return 100.0*n/(6*full_turbine), '%d/%d osservazioni' % (n, 6*full_turbine)
    return None, 'ignota'


def mse_grasso(run, met, split):
    f = os.path.join(run, 'grasso_predictions_%s.csv' % split)
    if os.path.isfile(f):
        return float(np.mean(pd.read_csv(f).errore.values**2))
    v = (met.get('grasso') or {}).get('mse_%s' % split)
    return float(v) if v is not None else None


def mse_cuscinetto(run, met, quale):
    f = os.path.join(run, 'cuscinetto_predictions.csv')
    if os.path.isfile(f):
        d = pd.read_csv(f)
        if quale == 'ispezioni':
            d = d[d.giorno.isin(GIORNI_ISP)]
        if len(d):
            return float(np.mean(d.errore.values**2))
    v = (met.get('cuscinetto') or {}).get('rmse_curva' if quale == 'curva'
                                          else 'rmse_ispezioni')
    return float(v)**2 if v is not None else None


def leggi_run(run, A):
    """Una riga per run, o None se il run e' incompleto."""
    fm = os.path.join(run, 'metrics.json')
    if not os.path.isfile(fm):
        return None
    met = json.load(open(fm))
    fc = os.path.join(run, 'config.json')
    cfg = json.load(open(fc)) if os.path.isfile(fc) else {}
    pct, fonte = percentuale(run, cfg, met, A.full_turbine)
    return dict(run=os.path.basename(os.path.normpath(run)), pct=pct, fonte_pct=fonte,
                metodo=cfg.get('metodo') or met.get('metodo') or '?',
                caso=cfg.get('case', met.get('caso')),
                grasso=mse_grasso(run, met, A.split),
                cuscinetto=mse_cuscinetto(run, met, A.bearing_metric))


def raccogli_csv(f, A):
    """Righe da un loss_on_percentage.csv, quello che train.py aggiorna a ogni run."""
    d = pd.read_csv(f)
    manca = {'percentuale', 'metodo'} - set(d.columns)
    if manca:
        sys.exit('%s non e\' un loss_on_percentage.csv: manca %s'
                 % (f, ', '.join(sorted(manca))))
    if A.caso and 'caso' in d.columns:
        d = d[d.caso.isin(A.caso)]
        if d.empty:
            sys.exit('nessuna riga con caso in %s in %s' % (sorted(A.caso), f))
    col_b = ('mse_danno_ispezioni' if A.bearing_metric == 'ispezioni'
             else 'mse_danno_curva')
    if A.split == 'train' and A.target != 'cuscinetto':
        print('  ! %s riporta solo la MSE del grasso su test: --split train ignorato'
              % os.path.basename(f))
    prendi = lambda r, c: (float(r[c]) if c in d.columns and pd.notna(r[c]) else None)
    return [dict(run=str(r.get('run', '?')), pct=_num_pct(r['percentuale']),
                 fonte_pct='loss_on_percentage.csv', metodo=r['metodo'],
                 caso=r.get('caso'), grasso=prendi(r, 'mse_grasso_test'),
                 cuscinetto=prendi(r, col_b))
            for r in d.to_dict('records') if _num_pct(r['percentuale']) is not None]


def raccogli_sorgente(percorso, A):
    """Un loss_on_percentage.csv oppure una cartella di run: decide da sola."""
    if os.path.isfile(percorso):
        return raccogli_csv(percorso, A)
    return raccogli(percorso, A)


def raccogli(cartella, A):
    """Tutti i run dentro `cartella` (che puo' anche essere un singolo run)."""
    if not os.path.isdir(cartella):
        sys.exit('cartella inesistente: %s' % cartella)
    figli = sorted(d for d in glob.glob(os.path.join(cartella, '*')) if os.path.isdir(d))
    candidati = [cartella] if os.path.isfile(os.path.join(cartella, 'metrics.json')) else figli
    righe, scartati = [], []
    for c in candidati:
        r = leggi_run(c, A)
        (righe.append(r) if r else scartati.append(os.path.basename(os.path.normpath(c))))
    if scartati:
        print('  ! ignorati (nessun metrics.json): %s' % ', '.join(scartati))
    if not righe:
        sys.exit('nessun run valido in %s' % cartella)
    if A.caso:
        righe = [r for r in righe if r['caso'] in A.caso]
        if not righe:
            sys.exit('nessun run con caso in %s in %s' % (sorted(A.caso), cartella))
    senza = [r['run'] for r in righe if r['pct'] is None]
    if senza:
        print('  ! percentuale non ricavabile, run esclusi: %s' % ', '.join(senza))
    return [r for r in righe if r['pct'] is not None]


def aggrega(righe, campo, stat='media'):
    """Statistica centrale per percentuale, con banda di dispersione.

    stat='media'   -> media aritmetica, banda min-max
    stat='mediana' -> mediana, banda interquartile (robusta agli outlier: con 10 casi
                      di inizializzazione il paper stesso ne classifica 3 come scadenti)
    """
    d = pd.DataFrame([r for r in righe if r[campo] is not None])
    if d.empty:
        return d
    g = d.groupby('pct')[campo].agg(
        centro=('median' if stat == 'mediana' else 'mean'),
        lo=((lambda x: x.quantile(.25)) if stat == 'mediana' else 'min'),
        hi=((lambda x: x.quantile(.75)) if stat == 'mediana' else 'max'),
        count='count').reset_index()
    return g.sort_values('pct')


def stile(ax, xl, yl, ti, log):
    ax.grid(axis='y', color=GRID, lw=.7); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTO, length=0, labelsize=9)
    ax.set_xlabel(xl, fontsize=10, color=MUTO); ax.set_ylabel(yl, fontsize=10, color=MUTO)
    ax.set_title(ti, fontsize=11, loc='left', fontweight='bold', color=INK, pad=8)
    ticks = sorted(set(PCT_TICKS))
    ax.set_xticks(ticks); ax.set_xticklabels(['%d%%' % t for t in ticks])
    ax.set_xlim(min(ticks)-8, max(ticks)+8)
    if log:
        ax.set_yscale('log')


def coincidono(gruppi, campo, stat='media'):
    """True se le due curve hanno gli stessi valori su tutte le x in comune."""
    ds = [aggrega(g['righe'], campo, stat) for g in gruppi]
    if len(ds) != 2 or any(d.empty for d in ds):
        return False
    a, b = (d.set_index('pct')['centro'] for d in ds)
    comune = a.index.intersection(b.index)
    return len(comune) > 0 and np.allclose(a[comune], b[comune], rtol=1e-9, atol=0)


def disegna(ax, gruppi, campo, titolo, log, ylab='MSE', etichette_fine=True, stat='media'):
    """Una curva per gruppo, marcatore diverso; barre min-max dove i run sono piu' di uno."""
    # se le curve sono identiche la seconda va tratteggiata, altrimenti sparisce sotto
    sovrapposte = coincidono(gruppi, campo, stat)
    punti, vuoto = [], True
    for i, g in enumerate(gruppi):
        d = aggrega(g['righe'], campo, stat)
        if d.empty:
            continue
        vuoto = False
        c, mk = g['colore'], MARCATORI[i % len(MARCATORI)]
        ls = '--' if (sovrapposte and i) else '-'
        rip = d[d['count'] > 1]
        if len(rip):                              # dispersione fra run allo stesso livello
            ax.vlines(rip.pct, rip['lo'], rip['hi'], color=c, lw=1.4, alpha=.55, zorder=2)
        ax.plot(d.pct, d['centro'], color=c, lw=2.2, ls=ls, marker=mk, ms=7.5,
                mfc=c, mec='white', mew=2, label=g['etichetta'], zorder=3+i, clip_on=False)
        u = d.iloc[-1]
        punti.append((float(u.pct), float(u['centro']), c))
    if vuoto:
        ax.text(.5, .5, 'metrica non disponibile', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color=MUTO)
    stile(ax, 'Percentuale di dati di training utilizzati', ylab, titolo, log)

    # etichette a fine curva, allontanate fra loro se finirebbero sovrapposte
    dy = [0]*len(punti)
    if len(punti) == 2 and punti[0][0] == punti[1][0]:
        ya, yb = (ax.transLimits.transform((x, y))[1] for x, y, _ in punti)
        if abs(ya-yb) < .035:
            dy = [7, -7] if ya >= yb else [-7, 7]
    for (x, y, c), off in zip(punti, dy):
        if not etichette_fine:
            break
        ax.annotate('%.3g' % y, (x, y), textcoords='offset points', xytext=(9, off),
                    va='center', fontsize=9, color=c, fontweight='bold',
                    annotation_clip=False)
    if sovrapposte and not vuoto:
        ax.set_title(titolo, fontsize=11, loc='left', fontweight='bold', color=INK, pad=24)
        ax.annotate('curve coincidenti (ramo identico nei due metodi)',
                    xy=(0, 1), xycoords='axes fraction', xytext=(0, 7),
                    textcoords='offset points', fontsize=8.5, color=MUTO)
    return not vuoto


def per_caso(A, gruppi, masked_b):
    """Un pannello per caso, assi y condivisi cosi' i casi sono confrontabili a vista."""
    campo = 'grasso' if A.target == 'grasso' else 'cuscinetto'
    if A.target == 'entrambi':
        print("  ! --per-caso mostra una metrica sola: uso 'cuscinetto'")
    casi = sorted({r['caso'] for g in gruppi for r in g['righe']
                   if r['caso'] is not None and r[campo] is not None})
    if not casi:
        sys.exit('nessun caso con la metrica %s' % campo)

    # limiti y comuni: senza questo i pannelli non sono confrontabili a occhio
    vals = [r[campo] for g in gruppi for r in g['righe'] if r[campo] is not None]
    lo, hi = min(vals), max(vals)
    marg = (hi/lo)**.08 if not A.linear and lo > 0 else 1
    ylim = (lo/marg, hi*marg) if marg != 1 else (lo-.05*(hi-lo), hi+.05*(hi-lo))
    if A.ylim:
        ylim = A.ylim

    # 4 casi stanno meglio in 2x2 che in 3+1
    nc = 2 if len(casi) in (2, 4) else min(3, len(casi))
    nr = -(-len(casi)//nc)
    # un pannello solo: figura piu' larga, altrimenti titolo e legenda escono dal bordo
    solo = len(casi) == 1
    f, axs = plt.subplots(nr, nc, figsize=(6.9 if solo else 4.6*nc, 4.8 if solo else 3.9*nr),
                          squeeze=False)
    ylab = 'masked MSE' if (campo == 'grasso' or masked_b) else 'MSE'
    for k, ax in enumerate(axs.ravel()):
        if k >= len(casi):
            ax.axis('off'); continue
        c = casi[k]
        sotto = [dict(g, righe=[r for r in g['righe'] if r['caso'] == c]) for g in gruppi]
        disegna(ax, sotto, campo, '' if solo else 'Caso %d' % c, not A.linear,
                ylab if k % nc == 0 else '', etichette_fine=solo, stat=A.stat)
        ax.set_ylim(*ylim)
        if k // nc != nr-1:
            ax.set_xlabel('')
        if k % nc:
            ax.tick_params(labelleft=False)

    f.suptitle('Masked MSE — caso %d' % casi[0] if solo
               else 'Masked MSE per caso di inizializzazione',
               fontsize=15, fontweight='bold', color=INK, x=.012, ha='left',
               y=.975 if solo else .985)
    h, l = axs[0][0].get_legend_handles_labels()
    if h:
        lg = f.legend(h, l, frameon=False, fontsize=9.5, ncol=len(h),
                      loc='upper left', bbox_to_anchor=(.008, .95))
        [t.set_color(INK) for t in lg.get_texts()]
    f.tight_layout(rect=(0, 0, .96 if solo else 1, .87 if solo else (.93 if nr > 1 else .86)))
    d = os.path.dirname(os.path.abspath(A.out))
    if d:
        os.makedirs(d, exist_ok=True)
    f.savefig(A.out, dpi=160); plt.close(f)

    tab = pd.concat([pd.DataFrame(g['righe']).assign(gruppo=g['etichetta'])
                     for g in gruppi], ignore_index=True)
    csv = os.path.splitext(A.out)[0] + '.csv'
    tab.sort_values(['gruppo', 'caso', 'pct']).to_csv(csv, index=False)
    print('\n-> %s\n-> %s' % (A.out, csv))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sorgente_a', help='loss_on_percentage.csv oppure runs/ del primo metodo')
    ap.add_argument('sorgente_b', nargs='?',
                    help='la seconda sorgente; omettila se la prima contiene i due metodi')
    ap.add_argument('--caso', type=lista_casi, metavar='N[,N...]',
                    help='tieni solo questi casi, separati da virgola (utile se il CSV '
                         'ne accumula piu\' di uno): --caso 4  oppure  --caso 1,2,3,10')
    ap.add_argument('--target', choices=('grasso', 'cuscinetto', 'entrambi'),
                    default='cuscinetto',
                    help='quale MSE mostrare (default: cuscinetto)')
    ap.add_argument('--split', choices=('test', 'train'), default='test',
                    help='split su cui calcolare la MSE del grasso (default: test)')
    ap.add_argument('--bearing-metric', choices=('curva', 'ispezioni'), default='ispezioni',
                    help='MSE del cuscinetto: curva giornaliera o 6 ispezioni (default: ispezioni)')
    ap.add_argument('--label-a', help='etichetta della prima curva')
    ap.add_argument('--label-b', help='etichetta della seconda curva')
    ap.add_argument('--full-turbine', type=int, default=10,
                    help='turbine del dataset completo, per il calcolo di ripiego (default: 10)')
    ap.add_argument('--stat', choices=('media', 'mediana'), default='media',
                    help='statistica su cui aggregare i casi: media con banda min-max, '
                         'oppure mediana con banda interquartile (default: media)')
    ap.add_argument('--per-caso', action='store_true',
                    help='un pannello per caso invece di una curva media: mostra la '
                         'dispersione fra inizializzazioni senza nasconderla in una media')
    ap.add_argument('--ylim', metavar='LO,HI',
                    help='limiti fissi dell\'asse y (es. 3e-7,4e-5): indispensabile per '
                         'confrontare immagini generate separatamente')
    ap.add_argument('--linear', action='store_true', help='asse y lineare invece che log')
    ap.add_argument('--out', default='data_efficiency_comparison.png',
                    help='PNG di uscita; la tabella va nel .csv con lo stesso nome')
    A = ap.parse_args()
    if A.ylim:
        try:
            A.ylim = tuple(float(x) for x in A.ylim.split(','))
            assert len(A.ylim) == 2 and A.ylim[0] < A.ylim[1]
        except (ValueError, AssertionError):
            sys.exit('--ylim vuole LO,HI con LO < HI (es. 3e-7,4e-5)')

    # una sorgente sola che contiene i due metodi si divide da se'
    una = A.sorgente_b is None or (os.path.exists(A.sorgente_b) and
                                   os.path.samefile(A.sorgente_a, A.sorgente_b))
    if una:
        print('%s:' % A.sorgente_a)
        tutte = raccogli_sorgente(A.sorgente_a, A)
        ordine = [m for m in ('physics', 'nophysics')
                  if any(r['metodo'] == m for r in tutte)]
        ordine += sorted({r['metodo'] for r in tutte} - set(ordine))
        if len(ordine) < 2:
            sys.exit('una sorgente sola con un solo metodo (%s): passa anche il secondo '
                     'percorso' % ', '.join(map(str, ordine)))
        if len(ordine) > 2:
            print('  ! piu\' di due metodi (%s): uso i primi due'
                  % ', '.join(map(str, ordine)))
        blocchi = [(A.sorgente_a, lab, [r for r in tutte if r['metodo'] == m])
                   for m, lab in zip(ordine[:2], (A.label_a, A.label_b))]
    else:
        blocchi = []
        for src, lab in ((A.sorgente_a, A.label_a), (A.sorgente_b, A.label_b)):
            print('%s:' % src)
            blocchi.append((src, lab, raccogli_sorgente(src, A)))

    gruppi = []
    for src, lab, righe in blocchi:
        metodi = sorted({str(r['metodo']) for r in righe})
        metodo = metodi[0] if len(metodi) == 1 else None
        gruppi.append(dict(
            cartella=src, righe=righe, metodo=metodo,
            etichetta=lab or ETICHETTE.get(metodo) or os.path.basename(os.path.normpath(src)),
            colore=COLORI.get(metodo)))
        for r in sorted(righe, key=lambda r: r['pct']):
            print('  %5.1f%%  %-38s  grasso %-11s cuscinetto %-11s  [%s]' % (
                r['pct'], r['run'],
                '%.4g' % r['grasso'] if r['grasso'] is not None else '-',
                '%.4g' % r['cuscinetto'] if r['cuscinetto'] is not None else '-',
                r['fonte_pct']))
        if len(metodi) > 1:
            print('  ! la sorgente mescola metodi diversi (%s)' % ', '.join(metodi))

    # colori: la convenzione del progetto se il metodo e' noto, altrimenti per ordine
    for g, fallback in zip(gruppi, (BLU, ARA)):
        if g['colore'] is None:
            g['colore'] = fallback
    if gruppi[0]['colore'] == gruppi[1]['colore']:
        gruppi[0]['colore'], gruppi[1]['colore'] = BLU, ARA

    pcts = sorted({r['pct'] for g in gruppi for r in g['righe']})
    if len(pcts) < 2:
        print('\n! un solo livello di dati (%s): la curva e\' un punto.\n'
              '  Servono run a 20/40/60/80/100%% di dati di training in ogni cartella.'
              % ', '.join('%g%%' % p for p in pcts))

    masked_b = A.bearing_metric == 'ispezioni'
    if A.per_caso:
        return per_caso(A, gruppi, masked_b)

    ti_b = ('masked MSE sulle 6 ispezioni del danno' if masked_b
            else 'MSE sulla curva giornaliera del danno')
    # il grasso e' sempre valutato sulle sole ispezioni: e' la masked MSE dell'eq. 10
    yl_g, yl_b = 'masked MSE', 'masked MSE' if masked_b else 'MSE'
    campi = [('grasso', 'MSE grasso (%s, 6 ispezioni x turbina)' % A.split, yl_g)] \
            if A.target == 'grasso' else \
            [('cuscinetto', ti_b, yl_b)] \
            if A.target == 'cuscinetto' else \
            [('grasso', 'Grasso — MSE %s' % A.split, yl_g), ('cuscinetto', ti_b, yl_b)]

    f, axs = plt.subplots(1, len(campi), figsize=(6.9*len(campi), 4.8), squeeze=False)
    n_casi = len({r['caso'] for g in gruppi for r in g['righe'] if r['caso'] is not None})
    for ax, (campo, ti, yl) in zip(axs[0], campi):
        if n_casi > 1:
            ti = '%s — %s su %d casi' % (ti, A.stat, n_casi)
        disegna(ax, gruppi, campo, ti, not A.linear, yl, stat=A.stat)
        if A.ylim:
            ax.set_ylim(*A.ylim)
    # titolo e legenda incolonnati a sinistra: non si scontrano a nessuna larghezza
    f.suptitle('Data efficiency comparison', fontsize=15, fontweight='bold',
               color=INK, x=.012, ha='left', y=.975)
    h, l = axs[0][0].get_legend_handles_labels()
    if h:
        lg = f.legend(h, l, frameon=False, fontsize=9.5, ncol=len(h),
                      loc='upper left', bbox_to_anchor=(.008, .93))
        [t.set_color(INK) for t in lg.get_texts()]
    # il margine destro lascia posto alle etichette di fine curva, fuori dagli assi
    f.tight_layout(rect=(0, 0, .96, .87))
    d = os.path.dirname(os.path.abspath(A.out))
    if d:
        os.makedirs(d, exist_ok=True)
    f.savefig(A.out, dpi=160); plt.close(f)

    tab = pd.concat([pd.DataFrame(g['righe']).assign(gruppo=g['etichetta'],
                                                     cartella=g['cartella'])
                     for g in gruppi], ignore_index=True)
    csv = os.path.splitext(A.out)[0] + '.csv'
    tab.sort_values(['gruppo', 'pct']).to_csv(csv, index=False)
    print('\n-> %s\n-> %s' % (A.out, csv))


if __name__ == '__main__':
    main()
