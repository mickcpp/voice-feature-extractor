import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
import joblib
import argparse
import os
import json

# ============================================
# MODEL CONFIGURATIONS
# ============================================
MODEL_CONFIGS = {
    'random_forest': {
        'class': RandomForestClassifier,
        'param_name': 'n_estimators',
        'default_params': {
            'random_state': 42,
            'n_jobs': -1,
        },
        'default_grid': [100, 150, 200, 300],
        'display_name': 'Random Forest'
    },
    'gradient_boosting': {
        'class': GradientBoostingClassifier,
        'param_name': 'n_estimators',
        'default_params': {
            'random_state': 42,
            'learning_rate': 0.05,
        },
        'default_grid': [50, 75, 100, 150],
        'display_name': 'Gradient Boosting'
    },
    'svm': {
        'class': SVC,
        'param_name': 'C',
        'default_params': {
            'random_state': 42,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True
        },
        'default_grid': [0.1, 0.5, 1.0, 5.0, 10.0],
        'display_name': 'Support Vector Machine'
    },
    'logistic_regression': {
        'class': LogisticRegression,
        'param_name': 'C',
        'default_params': {
            'random_state': 42,
            'max_iter': 5000,
            'n_jobs': -1,
        },
        'default_grid': [0.01, 0.1, 0.5, 1.0, 5.0],
        'display_name': 'Logistic Regression'
    },
    'mlp': {
        'class': MLPClassifier,
        'param_name': 'alpha',
        'default_params': {
            'random_state': 42,
            'max_iter': 2000,
            'hidden_layer_sizes': (20,)
        },
        'default_grid': [0.1, 0.5, 1.0, 2.0],
        'display_name': 'Multi-Layer Perceptron'
    }
}


def mc_cv_balanced_accuracy(estimator, X_tr, y_tr, test_size=0.20, n_splits=50, seed=42):
    """
    Monte-Carlo CV: esegue n_splits split 80–20 stratificati sul TRAIN,
    applica SMOTE SOLO sullo split di train, allena e valuta balanced accuracy sul 20% di validazione.
    Ritorna l'array degli score.
    """
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    scores = []
    for tr_idx, val_idx in sss.split(X_tr, y_tr):
        X_tr_i, X_val_i = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
        y_tr_i, y_val_i = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]

        # --- SMOTE solo sul TRAIN di questo split ---
        sm = SMOTE(random_state=seed)
        X_tr_bal, y_tr_bal = sm.fit_resample(X_tr_i, y_tr_i)

        # --- Fit & eval ---
        est = clone(estimator)
        est.fit(X_tr_bal, y_tr_bal)
        y_hat = est.predict(X_val_i)
        scores.append(balanced_accuracy_score(y_val_i, y_hat))

    scores = np.array(scores, dtype=float)
    print(f"\n[Monte-Carlo {n_splits}× (80–20 strat.)] balanced acc:"
          f" mean={scores.mean():.3f} std={scores.std():.3f}"
          f" p10={np.percentile(scores,10):.3f} p90={np.percentile(scores,90):.3f}"
          f" min={scores.min():.3f} max={scores.max():.3f}")
    return scores


def main(
    dataset_custom_path,
    dataset_egemaps_path,
    dataset_index_path,
    mc_summary_csv,
    model_path,
    test_set_path,
    test_target_path,
    x_columns_path,
    grid_param_list=None,
    model_type='random_forest',
    merged_dataset_path=None,
    target_column='Tipo soggetto'
):
    """
    Main training function with support for multiple models.
    
    Parameters:
    -----------
    model_type : str
        Type of model to train. Options: 'random_forest', 'gradient_boosting',
        'svm', 'logistic_regression', 'mlp'
    grid_param_list : list, optional
        List of parameter values to test in Monte-Carlo CV.
        If None, uses default grid for the selected model.
    merged_dataset_path : str, optional
        Path to a pre-merged dataset containing only features and target column.
        If provided, bypasses the merge process.
    target_column : str
        Name of the target column in the dataset.
    """
    # Validate model type
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Model type '{model_type}' not supported. "
                         f"Available options: {list(MODEL_CONFIGS.keys())}")
    
    model_config = MODEL_CONFIGS[model_type]
    print(f"\n{'='*60}")
    print(f"Training with: {model_config['display_name']}")
    print(f"{'='*60}\n")
    
    # Use default grid if not specified
    if grid_param_list is None:
        grid_param_list = model_config['default_grid']
    
    param_name = model_config['param_name']
    
    # Converti i path relativi in assoluti
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    def make_absolute(path):
        if not os.path.isabs(path):
            return os.path.join(script_dir, path)
        return path
    
    model_path = make_absolute(model_path)
    test_set_path = make_absolute(test_set_path)
    test_target_path = make_absolute(test_target_path)
    x_columns_path = make_absolute(x_columns_path)
    mc_summary_csv = make_absolute(mc_summary_csv)
    
    # ======================
    # Load the data
    # ======================
    if merged_dataset_path is not None:
        # Carica il dataset già merged
        print(f"Loading merged dataset from: {merged_dataset_path}")
        merged_df = pd.read_csv(merged_dataset_path, delimiter=';')
        
        # Features and target (tutte le colonne tranne il target sono feature)
        X = merged_df.drop(columns=[target_column])
        y = merged_df[target_column]
    else:
        # Processo di merge originale
        dataset_custom = pd.read_csv(dataset_custom_path, delimiter=';')
        dataset_egemaps = pd.read_csv(dataset_egemaps_path, delimiter=';')
        dataset_index = pd.read_excel(dataset_index_path)
        
        # Standardize filenames (remove everything after "Italian")
        dataset_custom['filename_standard'] = dataset_custom['filename'].apply(
            lambda x: x.split('Italian')[0] + 'Italian'
        )
        dataset_egemaps['filename_standard'] = dataset_egemaps['filename'].apply(
            lambda x: x.split('Italian')[0] + 'Italian'
        )
        
        # Merge dataset.csv and extracted_features_eGeMAPS.csv on standardized filename
        merged_df = pd.merge(
            dataset_egemaps,
            dataset_custom,
            left_on='filename_standard',
            right_on='filename_standard'
        )
        
        # Merge with dataset_index on subjectId and ID
        merged_df = pd.merge(
            merged_df,
            dataset_index,
            left_on='filename_x',
            right_on='FileName'
        )
        
        # Filter rows where Tipo audio is 'Free'
        merged_df = merged_df[merged_df['Tipo audio'] == 'Free']
        
        # Features and target
        X = merged_df.drop(
            columns=[
                target_column,
                'filename_standard',
                'filename_x',
                'filename_y',
                'subjectId_x',
                'subjectId_y',
                'ID',
                'FileName',
                'Tipo audio',
            ]
        )
        y = merged_df[target_column]
    
    # Encoding categorical variables (if present in the dataset)
    le = LabelEncoder()
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        X[col] = le.fit_transform(X[col])
    
    # Split data into 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Scaling condizionale
    # Modelli che beneficiano dello scaling dei dati
    models_needing_scaling = ['logistic_regression', 'svm', 'mlp']
    use_scaling = model_type in models_needing_scaling
    scaler = None
    
    if use_scaling:
        print(f"\n⚙ Scaling features per {model_config['display_name']}...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Converti di nuovo in DataFrame per mantenere i nomi delle colonne
        X_train = pd.DataFrame(
            X_train_scaled,
            columns=X_train.columns,
            index=X_train.index
        )
        X_test = pd.DataFrame(
            X_test_scaled,
            columns=X_test.columns,
            index=X_test.index
        )
        print(f"  ✓ Feature scaling completato")
    
    # SMOTE sul train (dopo eventuale scaling)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # ============================================
    # MONTE-CARLO CV (Stratified 80–20 ripetuto)
    # ============================================
    os.makedirs(os.path.dirname(mc_summary_csv) or ".", exist_ok=True)
    print(f"\nTesting {param_name} values: {grid_param_list}")
    
    mc_results = []
    for param_value in grid_param_list:
        # Create model with current parameter value
        model_params = model_config['default_params'].copy()
        model_params[param_name] = param_value
        model = model_config['class'](**model_params)
        
        print(f"\n--- Testing {param_name}={param_value} ---")
        scores = mc_cv_balanced_accuracy(
            model,
            X_train,
            y_train,
            test_size=0.20,
            n_splits=50,
            seed=42
        )
        
        mc_results.append({
            param_name: param_value,
            "mean": scores.mean(),
            "std": scores.std(),
            "p10": np.percentile(scores, 10),
            "p90": np.percentile(scores, 90)
        })
    
    mc_df = pd.DataFrame(mc_results).sort_values("mean", ascending=False)
    print(f"\n=== Monte-Carlo CV — riepilogo per {param_name} ===")
    print(mc_df.to_string(index=False) + "\n")
    mc_df.to_csv(mc_summary_csv, index=False, sep=";")
    
    # sceglie il miglior parametro
    best_row = mc_df.iloc[0]
    best_param_value = best_row[param_name]
    
    # Converte al tipo corretto in base al parametro
    if param_name in ['n_estimators']:
        best_param_value = int(best_param_value)
    elif param_name in ['C']:
        best_param_value = float(best_param_value)
    # hidden_layer_sizes è già una tupla, non serve conversione
    
    print(f"Miglior {param_name} da Monte-Carlo: {best_param_value} "
          f"(mean balanced acc = {best_row['mean']:.3f})")
    
    # Salva i nomi delle colonne
    X_columns = X_train.columns.tolist()
    os.makedirs(os.path.dirname(x_columns_path) or ".", exist_ok=True)
    with open(x_columns_path, 'w') as f:
        json.dump(X_columns, f)
    
    # ==========================
    # Train final model
    # ==========================
    final_params = model_config['default_params'].copy()
    final_params[param_name] = best_param_value
    final_model = model_config['class'](**final_params)
    final_model.fit(X_train_res, y_train_res)
    
    # Make predictions and evaluate
    y_pred = final_model.predict(X_test)
    print("\n" + "="*60)
    print(f"FINAL {model_config['display_name'].upper()} EVALUATION")
    print("="*60)
    print(classification_report(y_test, y_pred))
    balanced_ac = balanced_accuracy_score(y_test, y_pred)
    print(f"\nBalanced accuracy (TEST): {balanced_ac:.3f}")
    
    # Save the model and the test set
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(final_model, model_path)
    X_test.to_csv(test_set_path, index=False, sep=";")
    y_test.to_csv(test_target_path, index=False, sep=";")
    
    # Salva lo scaler se è stato usato
    if scaler is not None:
        scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"✓ Scaler saved to: {scaler_path}")
    
    print(f"\n✓ Model saved to: {model_path}")
    print(f"✓ Test set saved to: {test_set_path}")
    print(f"✓ Test target saved to: {test_target_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training Machine Learning models on custom + eGeMAPS dataset with Monte-Carlo CV."
    )
    
    # Path di input
    parser.add_argument(
        "--dataset-custom",
        type=str,
        required=False,
        help="Percorso al CSV datasrt custom."
    )
    parser.add_argument(
        "--dataset-egemaps",
        type=str,
        required=False,
        help="Percorso al CSV delle feature eGeMAPS."
    )
    parser.add_argument(
        "--dataset-index",
        type=str,
        required=False,
        help="Percorso al file dataset_index."
    )
    parser.add_argument(
        "--merged-dataset",
        type=str,
        default=None,
        help="Percorso al dataset già merged con solo feature e target. Se fornito, bypassa il merge."
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="Tipo soggetto",
        help="Nome della colonna target nel dataset."
    )
    
    # Path di output
    parser.add_argument(
        "--mc-summary-csv",
        type=str,
        default=r"risultati\mc_cv_summary.csv",
        help="Percorso del CSV di riepilogo Monte-Carlo CV."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=r"models\trained_model.pkl",
        help="Percorso dove salvare il modello addestrato."
    )
    parser.add_argument(
        "--test-set-path",
        type=str,
        default=r"models\test_set.csv",
        help="Percorso dove salvare X_test."
    )
    parser.add_argument(
        "--test-target-path",
        type=str,
        default=r"models\test_target.csv",
        help="Percorso dove salvare y_test."
    )
    parser.add_argument(
        "--x-columns-path",
        type=str,
        default=r"models\X_columns.json",
        help="Percorso dove salvare la lista dei nomi delle feature (colonne di X)."
    )
    
    # Model selection
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=list(MODEL_CONFIGS.keys()),
        help="Tipo di modello da addestrare."
    )
    
    # Parameter grid
    parser.add_argument(
        "--grid-param",
        type=str,
        nargs="+",
        default=None,
        help="Lista di valori del parametro da testare in Monte-Carlo (automatico in base al modello se non specificato)."
    )
    
    args = parser.parse_args()
    
    # Validazione: se non c'è merged_dataset, i tre path originali sono obbligatori
    if args.merged_dataset is None:
        if not all([args.dataset_custom, args.dataset_egemaps, args.dataset_index]):
            parser.error("Se --merged-dataset non è fornito, --dataset-custom, --dataset-egemaps e --dataset-index sono obbligatori.")
    
    # Convert grid_param to appropriate type based on model
    grid_param_list = None
    if args.grid_param is not None:
        model_config = MODEL_CONFIGS[args.model_type]
        param_name = model_config['param_name']
        
        # Convert based on parameter type
        if param_name in ['n_estimators']:
            grid_param_list = [int(x) for x in args.grid_param]
        elif param_name in ['C']:
            grid_param_list = [float(x) for x in args.grid_param]
        elif param_name == 'hidden_layer_sizes':
            # Parse tuples like "(100,50)" or "100"
            grid_param_list = []
            for x in args.grid_param:
                if ',' in x:
                    grid_param_list.append(tuple(int(i) for i in x.strip('()').split(',')))
                else:
                    grid_param_list.append((int(x),))
    
    main(
        dataset_custom_path=args.dataset_custom,
        dataset_egemaps_path=args.dataset_egemaps,
        dataset_index_path=args.dataset_index,
        mc_summary_csv=args.mc_summary_csv,
        model_path=args.model_path,
        test_set_path=args.test_set_path,
        test_target_path=args.test_target_path,
        x_columns_path=args.x_columns_path,
        grid_param_list=grid_param_list,
        model_type=args.model_type,
        merged_dataset_path=args.merged_dataset,
        target_column=args.target_column
    )
