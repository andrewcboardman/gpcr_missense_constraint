import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from scipy.interpolate import interp1d
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, average_precision_score, precision_recall_curve
from sklearn import datasets
from patsy import dmatrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def run_roc_analysis(df, metrics, level_col, levels_positive, iter = 10000):
    outputs = []
    for metric in metrics:
        # select 
        y_true = df[level_col].isin(levels_positive).astype(int).values
        y_pred = (-df[metric]).fillna(-100).values
        # Run ROC analysis
        roc_analysis = roc_bootstrap_cis(y_true, y_pred, iter=10000)
        roc_analysis['metric'] = metric
        roc_analysis['levels'] = levels_positive
        outputs.append(roc_analysis)
    return outputs

def auroc_table(rocs):
    df = []
    for roc in rocs:
        df.append({
            'metric': roc['metric'],
            'level':roc['levels'],
            'auroc':f"{roc['auroc']['mean']:.2f} ({roc['auroc']['lower_bound']:.2f} - {roc['auroc']['upper_bound']:.2f})"
        })
    return pd.DataFrame(df)

def plot_rocs(rocs, metric_names, show_ci = False, ax=None):
    if ax is None:
        ax = plt.gca()

    for i, metric_name in enumerate(metric_names):
        print(f"Plotting {metric_name} using column {rocs[i]['metric']}")
        auroc_label = f"{rocs[i]['auroc']['mean']:.2f} ({rocs[i]['auroc']['lower_bound']:.2f} - {rocs[i]['auroc']['upper_bound']:.2f})"
        ax.plot(rocs[i]['roc']['fpr'], rocs[i]['roc']['tpr']['mean'], label = metric_name + '\n' + auroc_label)
        if show_ci:
            ax.fill_between(rocs[i]['roc']['fpr'], rocs[i]['roc']['tpr']['lower_bound'], rocs[i]['roc']['tpr']['upper_bound'], alpha=0.5)

    ax.plot((0,1),(0,1),'r--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.legend(loc='lower right')
    ax.set_aspect("equal")
    return ax


def average_precision_bootstrap_ci(y_true, y_pred, alpha = 0.05, N_iter = 10000):
    N_actives = y_true.sum()
    N_inactives = len(y_true) - y_true.sum()
    bootstrap_y_true = np.concatenate((np.ones(N_actives),np.zeros(N_inactives)))

    ap = np.zeros(N_iter)

    if len(y_true) != len(y_pred):
        raise Exception("Input lengths do not match")
    for i in range(N_iter):
        bootstrap_actives = y_pred[y_true==1].sample(n=N_actives, replace=True).values
        bootstrap_inactives = y_pred[y_true==0].sample(n=N_inactives, replace=True).values
        bootstrap_y_pred = np.concatenate((bootstrap_actives,bootstrap_inactives))
        ap[i] = average_precision_score(bootstrap_y_true,bootstrap_y_pred)
    return np.quantile(ap, alpha), np.quantile(ap,1-alpha)

def average_precision_bootstrap_ci_contrast(y_true, y_pred_1, y_pred_2, alpha = 0.05, N_iter = 10000):
    N_actives = y_true.sum()
    N_inactives = len(y_true) - y_true.sum()
    bootstrap_y_true = np.concatenate((np.ones(N_actives),np.zeros(N_inactives)))

    d_ap = np.zeros(N_iter)

    if len(y_true) != len(y_pred_1) or len(y_true) != len(y_pred_2):
        raise Exception("Input lengths do not match")
    for i in range(N_iter):
        bootstrap_actives_1 = y_pred_1[y_true==1].sample(n=N_actives, replace=True).values
        bootstrap_inactives_1 = y_pred_1[y_true==0].sample(n=N_inactives, replace=True).values
        bootstrap_y_pred_1 = np.concatenate((bootstrap_actives_1,bootstrap_inactives_1))

        bootstrap_actives_2 = y_pred_2[y_true==1].sample(n=N_actives, replace=True).values
        bootstrap_inactives_2 = y_pred_2[y_true==0].sample(n=N_inactives, replace=True).values
        bootstrap_y_pred_2 = np.concatenate((bootstrap_actives_2,bootstrap_inactives_2))
        
        d_ap[i] = average_precision_score(bootstrap_y_true,bootstrap_y_pred_1) - \
            average_precision_score(bootstrap_y_true, bootstrap_y_pred_2)
    return np.quantile(d_ap, alpha), np.quantile(d_ap,1-alpha)



def roc_bootstrap_cis(y_test, y_predicted, alpha=0.05, iter= 10000):
    
    actives = y_predicted[np.where(y_test==1)[0]]
    inactives = y_predicted[np.where(y_test==0)[0]]
    y_test_bootstrap = np.concatenate((np.ones_like(actives), np.zeros_like(inactives)))

    fpr, tpr, _ = roc_curve(y_test, y_predicted)
    auroc = auc(fpr, tpr)
    tpr_bootstraps = np.zeros((iter, len(fpr)))
    auroc_bootstraps = np.zeros(iter)

    for i in range(iter):
        # choose bootstrap samples and construct dataset
        bootstrap_actives = np.random.choice(actives.flatten(), size=actives.shape[0])
        bootstrap_inactives = np.random.choice(inactives.flatten(), size=inactives.shape[0])
        y_predicted_bootstrap = np.concatenate((bootstrap_actives, bootstrap_inactives))

        # calculate the ROC for this case        
        fpr_bootstrap_, tpr_bootstrap_, _ = roc_curve(y_test_bootstrap, y_predicted_bootstrap)
        tpr_bootstraps[i, :] = interp1d(fpr_bootstrap_,tpr_bootstrap_)(fpr)

        # calculate the mean AUROC for this case
        auroc_bootstraps[i] = (bootstrap_actives > bootstrap_inactives[:,np.newaxis]).mean()

    # Extract ROC CI using interpolation and quantile
    tpr_lower = np.quantile(tpr_bootstraps,alpha, axis=0)
    tpr_upper = np.quantile(tpr_bootstraps,(1-alpha), axis=0)

    # Extract AUROC CI
    auroc_lower_bound = np.quantile(auroc_bootstraps,alpha)
    auroc_upper_bound = np.quantile(auroc_bootstraps,(1-alpha))

    # format output
    output = {
        'auroc':{'mean':auroc,'lower_bound':auroc_lower_bound, 'upper_bound':auroc_upper_bound},
        'roc':{'fpr':fpr,'tpr':{'mean':tpr,'lower_bound':tpr_lower,'upper_bound':tpr_upper}}
    }
    return output

def auroc_contrast_cis(y_test, y_predicted_1, y_predicted_2, alpha=0.05, iter= 10000):
    
    fpr, tpr, _ = roc_curve(y_test, y_predicted_1)
    auroc_1 = auc(fpr, tpr)
    fpr, tpr, _ =  roc_curve(y_test, y_predicted_2)
    auroc_2 = auc(fpr, tpr)
    print(auroc_1,auroc_2)
    d_auroc = auroc_1 - auroc_2
    d_auroc_bootstraps = np.zeros(iter)

    actives_1 = y_predicted_1[np.where(y_test==1)[0]]
    inactives_1 = y_predicted_1[np.where(y_test==0)[0]]

    actives_2 = y_predicted_2[np.where(y_test==1)[0]]
    inactives_2 = y_predicted_2[np.where(y_test==0)[0]]

    for i in range(iter):
        # choose bootstrap samples and construct dataset for case 1
        bootstrap_actives_1 = np.random.choice(actives_1.flatten(), size=actives_1.shape[0])
        bootstrap_inactives_1 = np.random.choice(inactives_1.flatten(), size=inactives_1.shape[0])

        # choose bootstrap samples and construct dataset for case 2
        bootstrap_actives_2 = np.random.choice(actives_2.flatten(), size=actives_2.shape[0])
        bootstrap_inactives_2 = np.random.choice(inactives_2.flatten(), size=inactives_2.shape[0])

        # calculate the AUROC difference
        auroc_bootstrap_1 = (bootstrap_actives_1 > bootstrap_inactives_1[:,np.newaxis]).mean()
        auroc_bootstrap_2 = (bootstrap_actives_2 > bootstrap_inactives_2[:,np.newaxis]).mean()
        d_auroc_bootstraps[i] = auroc_bootstrap_1 - auroc_bootstrap_2

    # Extract AUROC CI
    d_auroc_lower_bound = np.quantile(d_auroc_bootstraps,alpha)
    d_auroc_upper_bound = np.quantile(d_auroc_bootstraps,(1-alpha))

    # format output
    output = {'mean':d_auroc,'lower_bound':d_auroc_lower_bound, 'upper_bound':d_auroc_upper_bound, 'samples':d_auroc_bootstraps}
    return output


def calculate_rocs(data, target, target_levels, variant_class, input='oe_upper'):
    regdf = data[data.variant_class==variant_class]
    regdf = regdf[~regdf[input].isna() & ~np.isinf(regdf[input])]
    X = regdf[input]
    y = dmatrix('C(' + target + ',levels=target_levels)', data=regdf)

        
    outputs = []
    for i, level in enumerate(target_levels[1:]):
        fpr, tpr, thresholds = roc_curve(y[:,i+1],-X)
        outputs.append({
            'target':target,
            'variant_class':variant_class,
            'level':level,
            'fpr':fpr,
            'tpr':tpr,
            'thresholds':thresholds
        })

    return pd.DataFrame(outputs)

def f1_score_thresholds(data,target, target_levels, variant_class,input='oe_upper', thresholds=np.arange(0.1,1.3,0.1)):
    regdf = data[data.variant_class==variant_class]
    regdf = regdf[~regdf[input].isna() & ~np.isinf(regdf[input])]
    X = regdf[input]
    y = dmatrix('C(' + target + ',levels=target_levels)', data=regdf)

        
    outputs = []
    for i, level in enumerate(target_levels[1:]):
        for threshold in thresholds:
            x_ = 1* (X < threshold)
            outputs.append({
                'target':target,
                'variant_class':variant_class,
                'level':level,
                'threshold':threshold,
                'f1_score':f1_score(y[:,i+1],x_)
            })

    return pd.DataFrame(outputs)



def get_aurocs(rocs):
    outputs = []
    for _, roc in rocs.iterrows():
        outputs.append({
            'target':roc['target'],
            'variant_class':roc['variant_class'],
            'level':roc['level'],
            'auroc':auc(roc['fpr'], roc['tpr'])
        })
    return pd.DataFrame(outputs)


def plot_roc(fpr, tpr, label, ax_):
    ax_.plot(fpr,tpr,label=label)
    #ax_.set_title(annotation)    
    ax_.plot((0,1),(0,1),'r--')
    ax_.set_xlabel('FPR')
    ax_.set_ylabel('TPR')
    return ax_


def plot_target_level(data, ax):
    variant_class_dict = {
        'lof_hc': 'pLoF',
        'mis_pphen':'pPM missense',
        'mis_non_pphen':'pBM missense'
    }

    for _, roc in data.iterrows():
        plot_roc(
            roc['fpr'],
            roc['tpr'],
            variant_class_dict[roc['variant_class']],
            ax
            )
    ax.legend()
    return ax



def roc_stats(fpr, tpr, thresholds, threshold_):
    sensitivity = max(tpr[thresholds > threshold_])
    specificity = 1-max(fpr[thresholds > threshold_])
    lr_plus = sensitivity/(1-specificity)
    lr_minus = (1-sensitivity)/specificity

    output = {
        'sensitivity':sensitivity,
        'specificity':specificity,
        'lr_plus':lr_plus,
        'lr_minus':lr_minus,
        
    }
    return pd.DataFrame(output,index=[0])

def get_roc_stats(roc, thresholds):
    outputs = []
    for threshold in thresholds:
        output = roc_stats(roc['fpr'],roc['tpr'],roc['thresholds'],threshold)
        output['threshold'] = threshold
        outputs.append(output)
    outputs = pd.concat(outputs,ignore_index=True)
   
    return outputs

# functions for testing ROC CIs
def auroc_iterate(y_test_predicted, y_test):
    actives = np.where(y_test==1)[0]
    inactives = np.where(y_test==0)[0]
    aurocs = []
    for a in actives:
        aurocs.append((y_test_predicted[a] > y_test_predicted[inactives]).mean())
    return aurocs

def auroc_cis(aurocs,alpha=0.33):   
    aurocs_sorted = np.sort(aurocs)
    auroc_median = aurocs_sorted[len(aurocs) // 2]

    auroc_lower_bound = aurocs_sorted[int(len(aurocs)*alpha)]
    auroc_upper_bound = aurocs_sorted[int(len(aurocs)*(1-alpha))]
    return (auroc_median, auroc_lower_bound, auroc_upper_bound)

def auc_resample(y_predicted, y_test, iter= 10000):
    results = np.zeros(iter)
    random_idx = lambda x: np.random.choice(np.where(y_test==x)[0])
    for i in range(iter):
        positive_score = y_predicted[random_idx(1)]
        negative_score = y_predicted[random_idx(0)]
        results[i] = positive_score > negative_score
    return results.mean()

def test_auroc_cis():
    X, y = datasets.make_classification(n_samples=100,class_sep=0.7)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
    model = LogisticRegression()
    model.fit(X_train,y_train)
    y_train_predicted = model.predict_proba(X_train)[:,1]
    y_test_predicted = model.predict_proba(X_test)[:,1]


def plot_rocs(rocs, metric_names, show_ci = False, ax=None):
    if ax is None:
        ax = plt.gca()

    for i, metric_name in enumerate(metric_names):
        print(f"Plotting {metric_name} using column {rocs[i]['metric']}")
        auroc_label = ''#f"{rocs[i]['auroc']['mean']:.2f} ({rocs[i]['auroc']['lower_bound']:.2f} - {rocs[i]['auroc']['upper_bound']:.2f})"
        ax.plot(rocs[i]['roc']['fpr'], rocs[i]['roc']['tpr']['mean'], label = metric_name + '\n' + auroc_label)
        if show_ci:
            ax.fill_between(rocs[i]['roc']['fpr'], rocs[i]['roc']['tpr']['lower_bound'], rocs[i]['roc']['tpr']['upper_bound'], alpha=0.5)

    ax.plot((0,1),(0,1),'r--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.legend(loc='lower right')
    ax.set_aspect("equal")
    return ax

def plot_precision_recall(y_true, y_pred, label='', color='k'):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    avprc = average_precision_score(y_true,y_pred)
    plt.plot(recall[:-6], precision[:-6], label = label, color=color) 
    return None

def resample(target, predictors, fns, iter=100, n_sample = None, multiple=False):
    # Indices of positive and negative examples
    plus = np.where(target==1)[0]
    minus = np.where(target==0)[0]
    # Count positive and negative examples    
    if n_sample is None:
        n_plus = len(plus)
        n_minus = len(minus)
    # if number of samples specified, keep them in same ratio
    else:
        n_plus = int(n_sample * len(plus) / len(target))
        n_minus = int(n_sample * len(minus) / len(target))
        
    

    for i in tqdm(range(iter)):
        # select same number of positive and negative examples with replacement
        plus_select = np.random.choice(plus, size = n_plus)
        minus_select = np.random.choice(minus, size = n_minus)

        
        if not multiple:
            results = [[] for f in fns]
            x_sample = np.concatenate((predictors[plus_select], predictors[minus_select]),axis=0)
            y_sample = np.concatenate((target[plus_select], target[minus_select]),axis=0)
            for j, f in enumerate(fns):
                results[j].append(f(y_sample, x_sample))
        else:
            results = [[[] for p in predictors] for f in fns]
            for k, x_ in enumerate(predictors):
                x_sample = np.concatenate((x_[plus_select], x_[minus_select]),axis=0)
                y_sample = np.concatenate((target[plus_select], target[minus_select]),axis=0)
                for j, f in enumerate(fns):
                    results[j][k].append(f(y_sample, x_sample))

    return np.array(results)


def empirical_ci(samples, alpha):
    if len(samples.shape) > 1:
        raise ValueError('Input more than 1-d')
    samples_ = np.sort(samples)
    n = len(samples_)
    return (
        samples_[int(n * alpha/2 )],
        samples_[int(n * (1 - alpha/2))]
    )

def normal_ci(samples, alpha):
    if len(samples.shape) > 1:
        raise ValueError('Input more than 1-d')
    zscale = norm().ppf
    mean = np.mean(samples)
    sd = np.std(samples)
    return (
        mean + sd * zscale(alpha/2), 
        mean + sd * zscale(1- alpha/2)
    )


def plot_precision_recall(y_true, y_pred, label='',ax=None,dp=2):
    if ax is None:
        ax = plt.gca()
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true,y_pred)
    ax.plot(recall, precision, label = label + f': AP={round(ap,dp)}')
    return ap

def plot_roc(y_true, y_pred, label='', ax=None, dp=2):
    if ax is None:
        ax = plt.gca()
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auroc = roc_auc_score(y_true,y_pred)
    ax.plot(fpr, tpr, label = label + f': AUROC={round(auroc,dp)}')
    return auroc


def plot_comparison(samples, sample_labels, metric_label, bins=20, figsize=(6,2.5), legend_loc = 'upper right'):
    x1 = samples[:,0]
    x2 = samples[:,1]
    d = x1 - x2

    print(empirical_ci(x1, 0.05))
    print(empirical_ci(x2, 0.05))
    print(empirical_ci(d, 0.05))
    
    fig, ax = plt.subplots(1, 2, figsize = figsize, sharey=True)

    _ = ax[0].hist(x1,alpha=0.5,bins=bins,label=sample_labels[0])
    _ = ax[0].hist(x2,alpha=0.5,bins=bins,label=sample_labels[1])
    ax[0].set_xlabel(metric_label)
    ax[0].set_ylabel('Frequency')
    ax[0].legend(loc=legend_loc)

    _ = ax[1].hist(d,alpha=0.5,bins=bins, color='k')
    ax[1].set_xlabel(f'{metric_label} difference \n({sample_labels[0]} - {sample_labels[1]})')
    ax[1].set_ylabel('Frequency')
    return ax