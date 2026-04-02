"""
hERG Cardiotoxicity Prediction API
For deployment on Render.com — no CLI args needed.
Place all pkl files in the same directory as this file.
"""

import pickle, warnings, os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Draw import rdMolDraw2D
import base64

from mordred import Calculator, descriptors as mordred_descriptors

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))

def load_pkl(name):
    with open(os.path.join(BASE, name), 'rb') as f:
        return pickle.load(f)

rf_clf        = load_pkl('rf_classifier.pkl')
xgb_clf       = load_pkl('xgb_classifier.pkl')
meta_clf      = load_pkl('meta_classifier.pkl')
scaler        = load_pkl('scaler.pkl')
lasso         = load_pkl('lasso_selector.pkl')
FEAT_NAMES    = list(load_pkl('feat_names.pkl'))
FEAT_FILTERED = list(load_pkl('feat_filtered.pkl'))

print(f'Models loaded — {len(FEAT_NAMES)} selected features')

_remover   = SaltRemover()
_uncharger = rdMolStandardize.Uncharger()
_tautomer  = rdMolStandardize.TautomerEnumerator()
_mordred   = Calculator(mordred_descriptors, ignore_3D=True)


def curate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Cannot parse SMILES: {smiles}')
    mol = _remover.StripMol(mol)
    mol = _uncharger.uncharge(mol)
    mol = _tautomer.Canonicalize(mol)
    return mol


def mordred_dict(mol):
    d = _mordred(mol).asdict()
    out = {}
    for k, v in d.items():
        try:    out[str(k)] = float(v)
        except: out[str(k)] = np.nan
    return out


def build_feature_vector(desc_dict):
    x_filt   = np.array([[desc_dict.get(f, 0.0) for f in FEAT_FILTERED]], dtype=np.float64)
    x_filt   = np.nan_to_num(x_filt, nan=0.0)
    x_scaled = scaler.transform(x_filt)
    filt_idx = {name: i for i, name in enumerate(FEAT_FILTERED)}
    col_idx  = [filt_idx[f] for f in FEAT_NAMES if f in filt_idx]
    return x_scaled[:, col_idx]


def mol_to_svg(mol):
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 220)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return base64.b64encode(drawer.GetDrawingText().encode()).decode()


def rdkit_props(mol):
    return {
        'Molecular weight' : round(Descriptors.MolWt(mol), 2),
        'LogP'             : round(Descriptors.MolLogP(mol), 3),
        'H-bond donors'    : rdMolDescriptors.CalcNumHBD(mol),
        'H-bond acceptors' : rdMolDescriptors.CalcNumHBA(mol),
        'Rotatable bonds'  : rdMolDescriptors.CalcNumRotatableBonds(mol),
        'Aromatic rings'   : rdMolDescriptors.CalcNumAromaticRings(mol),
        'TPSA'             : round(Descriptors.TPSA(mol), 2),
    }


@app.route('/', methods=['GET'])
def index():
    html_path = os.path.join(BASE, 'herg_ui.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        return f.read(), 200, {'Content-Type': 'text/html'}


@app.route('/predict', methods=['POST'])
def predict():
    data   = request.get_json()
    smiles = (data or {}).get('smiles', '').strip()
    if not smiles:
        return jsonify({'error': 'No SMILES provided'}), 400
    try:
        mol = curate_smiles(smiles)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    desc  = mordred_dict(mol)
    X     = build_feature_vector(desc)

    p_rf  = rf_clf.predict_proba(X)[0, 1]
    p_xgb = xgb_clf.predict_proba(X)[0, 1]
    meta_input = np.column_stack([[p_rf], [p_xgb]])
    prob  = float(meta_clf.predict_proba(meta_input)[0, 1])
    label = int(prob >= 0.5)

    return jsonify({
        'probability' : round(prob, 4),
        'label'       : label,
        'p_rf'        : round(float(p_rf),  4),
        'p_xgb'       : round(float(p_xgb), 4),
        'svg'         : mol_to_svg(mol),
        'properties'  : rdkit_props(mol),
        'smiles'      : Chem.MolToSmiles(mol),
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'n_features': len(FEAT_NAMES)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
