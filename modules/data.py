import pandas as pd
import typing as tp
from IPython.display import display
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

class Dataset:

    def __init__(
            self, 
            train: pd.DataFrame,
            treatment_field: str,
            target_field: str,
            info_fields: tp.Optional[tp.List] = None,
            numeric_features: tp.Optional[tp.List] = None,
            cat_features: tp.Optional[tp.List] = None,
            valid: tp.Optional[pd.DataFrame] = None,
            test: tp.Optional[pd.DataFrame] = None,  
            show_stats: bool = True
    ) -> None:
        """
        Initializes a Dataset object for uplift modeling.

        Parameters:
            train (pd.DataFrame): The training dataset containing features, treatment and target variable.
            treatment_field (str): The name of the column indicating the treatment assignment (e.g., control or treatment group).
            target_field (str): The name of the column representing the target variable (outcome) to be predicted.
            info_fields (list, optional): A list of columns that serve as unique identifiers for observations or are other service information that should not be used in modeling. Defaults to an empty list.
            numeric_features (list, optional): A list of columns that are numeric features to be included in the modeling. Defaults to an empty list.
            cat_features (list, optional): A list of columns that are categorical features to be included in the modeling. Defaults to an empty list.
            valid (pd.DataFrame, optional): An optional validation dataset for model evaluation. Defaults to None.
            test (pd.DataFrame, optional): An optional test dataset for final model evaluation. Defaults to None.
            show_stats (bool): Whether to display statistics about the provided samples.
        """

        if not isinstance(train, pd.DataFrame):
            raise ValueError("The 'train' parameter must be a pandas DataFrame.")
        
        if treatment_field not in train.columns:
            raise ValueError(f"The 'treatment_field' '{treatment_field}' is not present in the training DataFrame.")

        if target_field not in train.columns:
            raise ValueError(f"The 'target_field' '{target_field}' is not present in the training DataFrame.")
        
        self.samples = ['train']
        self.__setattr__('train', train)
        self.treatment_field = treatment_field
        self.target_field = target_field
        self.info_fields = info_fields
        
        if info_fields is None:
            info_fields = []
        possible_features = [col for col in train.columns if col not in info_fields]
        if treatment_field in possible_features:
            possible_features.remove(treatment_field)
        if target_field in possible_features:
            possible_features.remove(target_field)
        
        if cat_features is None:
            self.cat_features = train[possible_features].select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            self.cat_features = cat_features

        if numeric_features is None:
            self.numeric_features = [col for col in possible_features if col not in self.cat_features]
        else:
            self.numeric_features = numeric_features
        
        self.feature_list = self.numeric_features + self.cat_features

        if valid is not None:
            if not isinstance(valid, pd.DataFrame):
                raise ValueError("The 'valid' parameter must be a pandas DataFrame if provided.")
            else:
                self.samples.append('valid')
                self.__setattr__('valid', valid)

        if test is not None:
            if not isinstance(test, pd.DataFrame):
                raise ValueError("The 'test' parameter must be a pandas DataFrame if provided.")
            else:
                self.samples.append('test')
                self.__setattr__('test', test)

        if show_stats:
            self.__display_data_stats()

    def __getitem__(self, sample):
        if sample not in self.samples:
            raise KeyError('Unknown sample')
        return getattr(self, sample)

    def __display_sample_stats(self, sample, sample_name) -> None:
        print(f'{sample_name} stats', end='')
        stata = sample.groupby(self.treatment_field).agg({self.target_field: ['count', 'mean']})
        stata.columns = stata.columns.droplevel()
        stata = stata.rename(columns={'count': 'n obs', 'mean':'mean target'})
        stata[r'% of population'] = 100*(stata['n obs']/stata['n obs'].sum())
        display(stata[['n obs', '% of population', 'mean target']])


    def __display_data_stats(self) -> None:
        numeric_print_fix = ''
        if len(self.numeric_features) > 10:
            numeric_print_fix = '\b' + ', ...]'
        cat_print_fix = ''
        if len(self.cat_features) > 10:
            cat_print_fix = '\b' + ', ...]'
        print(f'\033[1m Num features: \033[0m {len(self.numeric_features)} / {len(self.feature_list)} {self.numeric_features[:10]}', end='')
        print(numeric_print_fix)
        print(f'\033[1m Cat features: \033[0m {len(self.cat_features)} / {len(self.feature_list)} {self.cat_features[:10]}', end='')
        print(cat_print_fix, end='\n\n')

        for sample in self.samples:
            self.__display_sample_stats(getattr(self, sample), sample)

# TECHNICAL METHODS ######################################################################

def _decrease_feature_lists(dataset: Dataset, drop_features: list) -> None:

    dataset.feature_list = [feature for feature in dataset.feature_list if feature not in drop_features]

    dataset.cat_features = list(set(dataset.cat_features) & set(dataset.feature_list))
    dataset.numeric_features = list(set(dataset.numeric_features) & set(dataset.feature_list))

    print(f'{len(drop_features)} features were removed.')
    print(f'The remaining number of factors is {len(dataset.feature_list)}')

# FEATURE FILTERING ######################################################################

def delete_nan_features(dataset: Dataset, nan_threshold: float = 0.8) -> None:
    """
    Removes from consideration factors with a share of omissions greater than nan_treshold.
    """
    features_nan_proportion = dataset['train'][dataset.feature_list].isna().mean()
    mask = features_nan_proportion > nan_threshold
    nan_features = features_nan_proportion[mask].index.to_list()

    _decrease_feature_lists(dataset, nan_features)


def delete_high_cardinality_cat_features(
        dataset: Dataset, 
        n_unique_values: tp.Optional[int] = None,
        mean_unique_values: tp.Optional[float] = None,
    ) -> None:

    """
    Removes categorical features with high cardinality. It is necessary to pass one of the parameters
    n_unique_values or mean_unique_values.

    Parameters:
        dataset (Dataset): Dataset instance
        n_unique_values (int, optional): Number of unique values in a feature to remove it.
        mean_unique_values (float, optional): If unique values of a feature occur in no more than mean_unique_values number of observations, 
            then it is removed
    """
    
    if n_unique_values is None and mean_unique_values is None:
        raise ValueError('It is necessary to pass one of the parameters n_unique_values or mean_unique_values.')
    if n_unique_values is not None and mean_unique_values is not None:
        raise ValueError('It is necessary to pass one of the parameters n_unique_values or mean_unique_values.')
    
    if n_unique_values is not None:
        feature_nunique_values = dataset['train'][dataset.cat_features].nunique()
        mask = feature_nunique_values > n_unique_values
        high_cardinality_features = feature_nunique_values[mask].index.to_list()

    if mean_unique_values is not None:
        high_cardinality_features = []
        for feature in dataset.cat_features:
            if dataset['train'][feature].value_counts(True)[0] < mean_unique_values:
                high_cardinality_features.append(feature)

    _decrease_feature_lists(dataset, high_cardinality_features)
    

def delete_duplicated_features(dataset: Dataset) -> None:
    """
    Removes duplicate columns. Only unique columns will be kept.
    """
    duplicated_features = []

    for i, feature_1 in tqdm(enumerate(dataset.feature_list)):
        if feature_1 in duplicated_features:
            continue
        col1 = dataset['train'][feature_1]
        for feature_2 in dataset.feature_list[i+1:]:
            if feature_2 in duplicated_features:
                continue
            col2 = dataset['train'][feature_2]
            if col1.equals(col2):
                duplicated_features.append(feature_2)

    _decrease_feature_lists(dataset, duplicated_features)


def delete_high_corr_features(dataset: Dataset, corr_threshold: float = 0.95) -> None:
    """
    Removes numeric features that correlate with others more than corr_threshold.
    """
    cor_matrix = dataset['train'][dataset.numeric_features].corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    corr_features = [feature for feature in upper_tri.columns if any(upper_tri[feature] > corr_threshold)]

    _decrease_feature_lists(dataset, corr_features)


def delete_constant_features(dataset: Dataset):
    """
    Removes features that have one unique value.
    """
    feature_unique_values = dataset['train'][dataset.feature_list].nunique()
    mask = feature_unique_values == 1
    constant_features = feature_unique_values[mask].index.to_list()

    _decrease_feature_lists(dataset, constant_features)
    
# ADVERSARIAL VALIDATION ######################################################################

def adversarial_tg_cg(dataset: Dataset, sample: str, folds: int = 4) -> None:
    estimator = LGBMClassifier(n_jobs=1, verbose=0)
    cv = StratifiedKFold(n_splits=folds, random_state=42, shuffle=True)

    X = dataset[sample].loc[:, dataset.feature_list]
    for cat_feature in dataset.cat_features:
        X[cat_feature] = X[cat_feature].astype('category')
    y = dataset[sample].loc[:, dataset.treatment_field]

    AUCs = []
    feat_imp = pd.Series(0, index=dataset.feature_list)
    for train_idx, test_idx in tqdm(cv.split(X, y)):
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx], categorical_feature=dataset.cat_features)
        pred = estimator.predict_proba(X.iloc[test_idx])[:, 1]
        feat_imp = feat_imp + estimator.feature_importances_
        AUCs.append(roc_auc_score(y.iloc[test_idx], pred))
    feat_imp = feat_imp / folds

    print(f'ROC-AUC in {sample} TG/CG: {round(np.mean(AUCs), 4)}')
    display(feat_imp.nlargest(30).plot(kind='barh', figsize=(8,10), title='Adversarial feature importance'))