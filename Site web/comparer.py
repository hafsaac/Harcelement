import sys
import joblib
import os

BASE = '/var/www/html/safetext/models'

modeles = [
    {
        'nom': 'Régression Logistique',
        'bin': os.path.join(BASE, 'modele_reglog_binaire.pkl'),
        'cat': os.path.join(BASE, 'modele_reglog_multiclasse.pkl'),
        'vec_bin': None,
        'vec_cat': None,
        'type': 'pipeline'
    },
    {
        'nom': 'TF-IDF',
        'bin': os.path.join(BASE, 'modele_tfidf_binaire.pkl'),
        'cat': os.path.join(BASE, 'modele_tfidf_multiclasse.pkl'),
        'vec_bin': os.path.join(BASE, 'vectorizer_tfidf.pkl'),
        'vec_cat': os.path.join(BASE, 'vectorizer_tfidf.pkl'),
        'type': 'vectorizer'
    },
    {
        'nom': 'Bag of Words',
        'bin': os.path.join(BASE, 'modele_bow_binaire.pkl'),
        'cat': os.path.join(BASE, 'modele_bow_multiclasse.pkl'),
        'vec_bin': None,
        'vec_cat': None,
        'type': 'pipeline'
    },
    {
        'nom': 'Random Forest',
        'bin': os.path.join(BASE, 'modele_rf_binaire.pkl'),
        'cat': os.path.join(BASE, 'modele_rf_multiclasse.pkl'),
        'vec_bin': os.path.join(BASE, 'vectorizer_rf_binaire.pkl'),
        'vec_cat': os.path.join(BASE, 'vectorizer_rf_multiclasse.pkl'),
        'type': 'vectorizer'
    },
    {
        'nom': 'Toxic BERT',
        'bin': os.path.join(BASE, 'model_binary'),
        'cat': os.path.join(BASE, 'model_category'),
        'vec_bin': None,
        'vec_cat': None,
        'type': 'bert'
    },
]

texte = sys.argv[1] if len(sys.argv) > 1 else ""

if not texte.strip():
    print("ERREUR:texte vide")
    sys.exit(1)

for m in modeles:
    try:
        if m['type'] == 'pipeline':
            model_bin = joblib.load(m['bin'])
            pred_bin = model_bin.predict([texte])[0]
            proba_bin = model_bin.predict_proba([texte])[0]
            confiance = round(float(max(proba_bin)) * 100, 1)
            if str(pred_bin) in ['Haineux', '1', 1]:
                model_cat = joblib.load(m['cat'])
                pred_cat = model_cat.predict([texte])[0]
            else:
                pred_cat = 'Non applicable'

        elif m['type'] == 'vectorizer':
            vec_bin = joblib.load(m['vec_bin'])
            model_bin = joblib.load(m['bin'])
            X_bin = vec_bin.transform([texte])
            pred_bin = model_bin.predict(X_bin)[0]
            proba_bin = model_bin.predict_proba(X_bin)[0]
            confiance = round(float(max(proba_bin)) * 100, 1)
            if str(pred_bin) in ['Haineux', '1', 1]:
                vec_cat = joblib.load(m['vec_cat'])
                model_cat = joblib.load(m['cat'])
                X_cat = vec_cat.transform([texte])
                pred_cat = model_cat.predict(X_cat)[0]
            else:
                pred_cat = 'Non applicable'

        elif m['type'] == 'bert':
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            tokenizer_bin = AutoTokenizer.from_pretrained(m['bin'])
            model_bin_bert = AutoModelForSequenceClassification.from_pretrained(m['bin'])
            pipe_bin = pipeline("text-classification", model=model_bin_bert, tokenizer=tokenizer_bin)
            result_bin = pipe_bin(texte[:512])[0]
            score = result_bin['score']
            label = result_bin['label']
            if label in ['LABEL_1', '1', 'Haineux', 'POSITIVE']:
                pred_bin = 'Haineux'
            else:
                pred_bin = 'Non haineux'
            confiance = round(score * 100, 1)
            if pred_bin == 'Haineux':
                tokenizer_cat = AutoTokenizer.from_pretrained(m['cat'])
                model_cat_bert = AutoModelForSequenceClassification.from_pretrained(m['cat'])
                pipe_cat = pipeline("text-classification", model=model_cat_bert, tokenizer=tokenizer_cat)
                result_cat = pipe_cat(texte[:512])[0]
                pred_cat = result_cat['label']
            else:
                pred_cat = 'Non applicable'

        label_bin = 'Haineux' if str(pred_bin) in ['Haineux', '1', 1] else 'Non haineux'
        print(f"{m['nom']}|{label_bin}|{pred_cat}|{confiance}")

    except Exception as e:
        print(f"{m['nom']}|Erreur|Erreur|0")