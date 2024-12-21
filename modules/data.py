import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from causalml.feature_selection.filters import FilterSelect
from IPython.display import display
from lightgbm import LGBMClassifier
from pandas.api.types import is_numeric_dtype
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm



class Dataset:

    def __init__(
            self, 
            train: pd.DataFrame,
            treatment_field: str,
            target_field: str,
            info_fields: tp.Optional[tp.List] = None,
            numeric_features: tp.Optional[tp.List] = None,
            cat_features: tp.Optional[tp.List] = None,
            control_group: tp.Any = 0,
            treatment_group: tp.Any = 1,
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
            control_group (any): The index of the control group. Defaults to 0.
            treatment_group (any): The index of the treatment group. Defaults to 1.
            cat_features (list, optional): A list of columns that are categorical features to be included in the modeling. Defaults to an empty list.
            valid (pd.DataFrame, optional): An optional validation dataset for model evaluation. Defaults to None.
            test (pd.DataFrame, optional): An optional test dataset for final model evaluation. Defaults to None.
            show_stats (bool): Whether to display statistics about the provided samples.
        """

        if not isinstance(train, pd.DataFrame):
            raise ValueError("The 'train' parameter must be a pandas DataFrame.")
        
        if treatment_field not in train.columns:
            raise ValueError(f"The 'treatment_field' '{treatment_field}' is not present in the training DataFrame.")

        if train[treatment_field].nunique() != 2:
            raise ValueError(f"The 'treatment_field' '{treatment_field}' must have exactly two unique values.")

        if target_field not in train.columns:
            raise ValueError(f"The 'target_field' '{target_field}' is not present in the training DataFrame.")
        
        self.samples = ['train']
        self.__setattr__('train', train)
        self.treatment_field = treatment_field
        self.treatment_group = treatment_group
        self.control_group = control_group
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

def F_filter(
        dataset: Dataset, 
        order: int = 1, 
        p_value: float = 0.05,
        fillna_value: float = 0
    ) -> None:
    """
    Removes features that do not pass the F-test.
    The function uses the CausalML FilterSelect class to calculate the importance of each feature using the F-test.
    It then removes the features that have a p-value greater than the specified threshold.

    Paper: https://arxiv.org/pdf/2005.03447
    """
    filter_method = FilterSelect()
    f_imp = filter_method.get_importance(
        dataset['train'].fillna(fillna_value), 
        dataset.numeric_features, 
        dataset.target_field, 
        'F', 
        experiment_group_column = dataset.treatment_field,
        control_group = dataset.control_group,
        treatment_group = dataset.treatment_group,
        order = order
    )
    removed_features = f_imp[f_imp['p_value'] > p_value]['feature'].to_list()

    _decrease_feature_lists(dataset, removed_features)


def LR_filter(
        dataset: Dataset, 
        order: int = 1, 
        p_value: float = 0.05,
        fillna_value: float = 0
    ) -> None:
    """
    Removes features that do not pass the Logistic Regression test.
    The function uses the CausalML FilterSelect class to calculate the importance of each feature using 
    the Logistic Regression test. It then removes the features that have a p-value greater than the specified threshold.

    Paper: https://arxiv.org/pdf/2005.03447
    """
    filter_method = FilterSelect()
    f_imp = filter_method.get_importance(
        dataset['train'].fillna(fillna_value), 
        dataset.numeric_features, 
        dataset.target_field, 
        'LR', 
        experiment_group_column = dataset.treatment_field,
        control_group = dataset.control_group,
        treatment_group = dataset.treatment_group,
        order = order
    )
    removed_features = f_imp[f_imp['p_value'] > p_value]['feature'].to_list()

    _decrease_feature_lists(dataset, removed_features)

def KL_filter(
        dataset: Dataset,
        n_bins: int = 10,
        score: float = 1e-5,
        fillna_value: float = 0
) -> None:
    """
    Removes features that do not pass the Kullback-Leibler divergence test. 
    The function uses the CausalML FilterSelect class to calculate the importance of each feature 
    using the Kullback-Leibler divergence test. It then removes the features that have a score lower 
    than the specified threshold.

    Paper: https://arxiv.org/pdf/2005.03447
    """
    
    filter_method = FilterSelect()
    f_imp = filter_method.get_importance(
        dataset['train'].fillna(fillna_value), 
        dataset.numeric_features, 
        dataset.target_field, 
        'KL', 
        experiment_group_column = dataset.treatment_field,
        control_group = dataset.control_group,
        treatment_group = dataset.treatment_group,
        n_bins = n_bins
    )

    removed_features = f_imp[f_imp['score'] < score]['feature'].to_list()

    _decrease_feature_lists(dataset, removed_features)
    
# ADVERSARIAL VALIDATION ######################################################################

def adversarial_tg_cg(dataset: Dataset, sample: str, folds: int = 4) -> None:
    """
    A test to check the quality of splitting into control and treatment groups. The meta-classifier is 
    trained on the specified sample. The features used are those that have been 
    selected to date. The target is the treatment group flag. The average value of the ROC-AUC is returned 
    based on cross-validation and the top features of the meta-classifier.

    Parameters:
        dataset (Dataset): Dataset instance
        sample (str): Name of the sample to be tested.
        folds (int): Number of folders for cross-validation. Default 4
    """
    estimator = LGBMClassifier(n_jobs=1, verbose=0)
    cv = StratifiedKFold(n_splits=folds, random_state=42, shuffle=True)

    X = dataset[sample].loc[:, dataset.feature_list]
    for cat_feature in dataset.cat_features:
        X[cat_feature] = X[cat_feature].astype('category')
    y = dataset[sample].loc[:, dataset.treatment_field]
    if not is_numeric_dtype(y):
        y = LabelEncoder().fit_transform(y)

    AUCs = []
    feat_imp = pd.Series(0, index=dataset.feature_list)
    for train_idx, test_idx in tqdm(cv.split(X, y)):
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx], categorical_feature=dataset.cat_features)
        pred = estimator.predict_proba(X.iloc[test_idx])[:, 1]
        feat_imp = feat_imp + estimator.feature_importances_
        AUCs.append(roc_auc_score(y.iloc[test_idx], pred))
    feat_imp = feat_imp / folds

    print(f'ROC-AUC in {sample} TG/CG: {round(np.mean(AUCs), 4)}')

    for_plotting = feat_imp.nlargest(30)
    sns.barplot(x=for_plotting, y=for_plotting.index, orient='h',)
    plt.title('Feature importance')
    plt.show()

def adversarial_split_quality(dataset: Dataset, sample_1: str, sample_2: str, folds: int = 4) -> None:
    """
    They will check two samples for independence of partition. The meta-classifier is trained by 
    the features used to determine which observations are from the first sample and which are from the second.
    The average value of the ROC-AUC is returned based on cross-validation and the top features of the meta-classifier.

    Parameters:
        dataset (Dataset): Dataset instance
        sample_1 (str): Name of the first sample.
        sample_2 (str): Name of the second sample.
        folds (int): Number of folders for cross-validation. Default 4
    """
    estimator = LGBMClassifier(n_jobs=1, verbose=0)
    cv = StratifiedKFold(n_splits=folds, random_state=42, shuffle=True)

    X = pd.concat(
        [
            dataset[sample_1].loc[:, dataset.feature_list],
            dataset[sample_2].loc[:, dataset.feature_list]
        ]
    )
    for cat_feature in dataset.cat_features:
        X[cat_feature] = X[cat_feature].astype('category')

    y = np.array([0]*len(dataset[sample_1]) + [1]*len(dataset[sample_2]))

    AUCs = []
    feat_imp = pd.Series(0, index=dataset.feature_list)
    for train_idx, test_idx in tqdm(cv.split(X, y)):
        estimator.fit(X.iloc[train_idx], y[train_idx], categorical_feature=dataset.cat_features)
        pred = estimator.predict_proba(X.iloc[test_idx])[:, 1]
        feat_imp = feat_imp + estimator.feature_importances_
        AUCs.append(roc_auc_score(y[test_idx], pred))
    feat_imp = feat_imp / folds

    print(f'ROC-AUC in {sample_1}/{sample_2} split: {round(np.mean(AUCs), 4)}')
    
    for_plotting = feat_imp.nlargest(30)
    sns.barplot(x=for_plotting, y=for_plotting.index, orient='h',)
    plt.title('Feature importance')
    plt.show()

# STATISTICAL CHECKS OF A/B TEST ######################################################################

def two_proprotions_confint(dataset: Dataset, significance: float = 0.05, sample: tp.Optional[str] = None):
    """
    A/B test for two proportions;
    given a success a trial size of group A and B compute its confidence interval.

    Parameters:
        dataset (Dataset): Dataset instance
        
        significance (float): Often denoted as alpha. Governs the chance of a false positive.
        A significance level of 0.05 means that there is a 5% chance of a false positive. In other words, 
        our confidence level is 1 - 0.05 = 0.95
        
        sample (str, optional): The name of the sample for which the check is to be performed. If nothing is 
        transferred, the calculation will be performed on all available data.

    Source: https://stackoverflow.com/questions/47570903/confidence-interval-for-the-difference-between-two-proportions-in-python
    """

    if sample is not None:
        data = dataset[sample]
    else:
        data = pd.concat([dataset[sample] for sample in dataset.samples])
    
    group_a = data[data[dataset.treatment_field] == dataset.control_group]
    size_a = group_a.shape[0]
    prop_a = group_a[dataset.target_field].mean()

    group_b = data[data[dataset.treatment_field] == dataset.treatment_group]
    prop_b = group_b[dataset.target_field].mean()
    size_b = group_b.shape[0]

    var = prop_a * (1 - prop_a) / size_a + prop_b * (1 - prop_b) / size_b
    se = np.sqrt(var)
    z = norm.ppf(1 - significance/2)
    prop_diff = prop_b - prop_a

    print(f'Proportions difference: {prop_diff}')
    print(f'Confidence interval: {(prop_diff - z*se, prop_diff + z*se)}')
    print(f'0 in confidence interval: {prop_diff - z*se < 0 < prop_diff + z*se}')

def psi_cg_tg(
        dataset: Dataset,
        sample: tp.Optional[str] = None,
        num_bins: int = 15,
        save_path: tp.Optional[str] = None,
    ) -> None:
    """
    Calculates the PSI (Population Stability Index) for each feature in the dataset.
    """
    if sample is not None:
        data = dataset[sample]
    else:
        data = pd.concat([dataset[sample] for sample in dataset.samples])

    res_psi = pd.DataFrame(
        {
            'feature': dataset.feature_list,
            'psi': [0. for _ in range(len(dataset.feature_list))]
        }
    )

    def make_distribution(full_data, clear_data, nan_bin):
        bins = nan_bin + [str(bucket) for bucket in sorted(clear_data.unique())]
        full_data = pd.concat([clear_data, full_data[full_data.isna()]]) if full_data.isna().any() else clear_data
        return full_data.astype(str).value_counts(True)[bins]

    for feature in tqdm(dataset.feature_list):
        control_group = data[data[dataset.treatment_field] == dataset.control_group][feature].fillna(np.nan)
        treatment_group = data[data[dataset.treatment_field] == dataset.treatment_group][feature].fillna(np.nan)

        control_group_clear = control_group.dropna()
        treatment_group_clear = treatment_group.dropna()

        nan_bin = ['nan'] if control_group.isna().any() or treatment_group.isna().any() else []

        if feature not in dataset.cat_features and control_group_clear.nunique() > num_bins:
            bins = [-np.inf] + list(np.histogram(control_group_clear, bins=num_bins)[1]) + [np.inf]
            control_group_clear = pd.cut(control_group_clear, bins=bins)
            treatment_group_clear = pd.cut(treatment_group_clear, bins=bins)

        control_group = make_distribution(control_group, control_group_clear, nan_bin)
        treatment_group = make_distribution(treatment_group, treatment_group_clear, nan_bin)

        psi = 0
        for bucket in sorted(set(control_group.index).union(set(treatment_group.index))):
            a_value = control_group.get(bucket, 1e-10)
            b_value = treatment_group.get(bucket, 1e-10)
            psi += (a_value - b_value) * np.log(a_value / b_value)
        res_psi.loc[res_psi['feature'] == feature, 'psi'] = psi

    res_psi = res_psi.sort_values(by='psi', ascending=False)
    if save_path is not None:
        res_psi.to_csv(save_path, index=False)
    display(res_psi)


